import math
from typing import Callable, Literal

import torch
import torch.nn as nn

from energies import BaseEnergy
from models.modules import BaseModule
from utils.misc_utils import maybe_compile
from utils.train_utils import binary_search_smoothing, ess

logtwopi = math.log(2 * math.pi)


def cos_sq_fn_step_scheme(n_steps, s=0.008, noise_scale=6.0, dtype=torch.float32):
    pre_phase = torch.linspace(0, 1, n_steps + 1, dtype=dtype)
    phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5
    dts = torch.cos(phase) ** 4
    dts_out = dts / dts.sum()
    return dts_out * noise_scale


class GFN(nn.Module):
    def __init__(
        self,
        energy: BaseEnergy,
        module: BaseModule,
        num_steps: int | None = None,
        reference_process: Literal["pinned_brownian", "ou"] = "pinned_brownian",
        # --- Pinned Brownian Args --- #
        t_scale: float | None = None,
        # --- OU Args --- #
        init_std: float | None = None,
        noise_scale: float | None = None,
        # --- SubTB Args --- #
        partial_energy: bool = False,
        learn_beta: bool = False,
        device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.energy = energy
        self.pred_module = module
        self.device = device
        self.num_steps = num_steps
        self.dt = torch.tensor(1.0 / num_steps, device=self.device)
        self.times = torch.arange(num_steps + 1, device=self.device) * self.dt

        self.reference_process = reference_process
        self.t_scale = self.init_std = self.noise_scale = None

        if reference_process == "pinned_brownian":
            assert t_scale is not None
            self.t_scale = t_scale
            self.sample_initial_state = lambda bsz: torch.zeros(
                (bsz, self.energy.ndim), device=self.device
            )
            self.initial_logprob = lambda s: torch.zeros((s.shape[0],), device=self.device)
            self.alpha_fn = self.lambda_fn = None
        elif reference_process == "ou":
            assert init_std is not None and noise_scale is not None
            self.init_std = init_std
            self.noise_scale = noise_scale
            self.initial_dist = torch.distributions.Normal(
                torch.zeros((self.energy.ndim,), device=self.device),
                torch.full((self.energy.ndim,), init_std, device=self.device),
            )
            self.sample_initial_state = lambda bsz: self.initial_dist.sample(
                sample_shape=torch.Size((bsz,))
            )
            self.initial_logprob = lambda s: self.initial_dist.log_prob(s).sum(-1)
            alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale).to(self.device)
            self.alpha_fn = lambda step: alphas[step]
            self.lambda_fn = lambda step: alphas[step]
        else:
            raise ValueError(f"Invalid reference process: {reference_process}")

        self.partial_energy = partial_energy
        self.beta_model = None
        if learn_beta:
            self.beta_model = torch.nn.Parameter(
                torch.cat([torch.ones(self.num_steps, device=self.device)])
            )
            self.softplus = nn.Softplus()

    def forward_step(
        self,
        s: torch.Tensor,  # state at time t
        s_next: torch.Tensor | None,  # state at time t + \Delta t; if None, we sample
        step: int,  # step at time t; not used here
        pf_mean: torch.Tensor,
        pf_logvar_correction: torch.Tensor,
        detach: bool = True,  # for PIS
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_process == "pinned_brownian":
            fwd_mean = s + self.dt * pf_mean
            fwd_std = (self.dt * self.t_scale).sqrt()
        elif self.reference_process == "ou":
            sqrt_at = torch.clamp(self.alpha_fn(step).sqrt(), 0.0, 1.0)
            sqrt_1_minus_at = (1 - sqrt_at**2).sqrt()
            fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * pf_mean
            fwd_std = sqrt_at * self.init_std
        else:
            raise ValueError(f"Invalid reference process: {self.reference_process}")

        fwd_std = fwd_std * (pf_logvar_correction / 2).exp()

        if s_next is None:
            s_next = fwd_mean + fwd_std * torch.randn_like(s, device=s.device)
        s_next = s_next.detach() if detach else s_next
        noise = (s_next - fwd_mean) / fwd_std
        log_pfs = -0.5 * (logtwopi + 2 * fwd_std.log() + noise**2).sum(1)
        return s_next, log_pfs

    def backward_step(
        self,
        s: torch.Tensor | None,  # state at time t; if None, we sample
        s_next: torch.Tensor,  # state at time t + \Delta t
        step: int,  # step at time t
        pb_mean_correction: torch.Tensor,
        pb_var_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_process == "pinned_brownian":
            t_next = (step + 1) / self.num_steps

            bwd_mean = s_next - s_next * self.dt / t_next
            bwd_std = (self.t_scale * self.dt * (t_next - self.dt) / t_next).sqrt()
        elif self.reference_process == "ou":
            sqrt_at = torch.clamp(self.alpha_fn(step).sqrt(), 0.0, 1.0)
            sqrt_1_minus_at = (1 - sqrt_at**2).sqrt()
            bwd_mean = sqrt_1_minus_at * s_next
            bwd_std = sqrt_at * self.init_std
        else:
            raise ValueError(f"Invalid reference process: {self.reference_process}")

        bwd_mean = bwd_mean * pb_mean_correction
        bwd_std = bwd_std * pb_var_correction.sqrt()

        if self.reference_process == "pinned_brownian" and step == 0:
            s = torch.zeros_like(s_next)
            log_pbs = torch.zeros_like(bwd_mean[:, 0])
        else:
            if s is None:
                s = bwd_mean + bwd_std * torch.randn_like(s_next)
            s = s.detach()
            noise = (s - bwd_mean) / bwd_std
            log_pbs = -0.5 * (logtwopi + 2 * bwd_std.log() + noise**2).sum(1)
        return s, log_pbs

    @maybe_compile
    def get_partial_energy(
        self,
        states: torch.Tensor,  # (bsz, T', ndim)
        steps: torch.Tensor,  # (T')
    ) -> torch.Tensor:
        assert self.partial_energy
        bsz = states.shape[0]

        ts = steps / self.num_steps

        if self.beta_model is not None:
            betas = self.softplus(self.beta_model).cumsum(0)
            betas = betas / betas[-1]
            betas = torch.cat([torch.zeros(1, device=self.device), betas], dim=0)
            betas = betas[steps]
        else:
            betas = ts

        ts = ts.unsqueeze(0)
        betas = betas.unsqueeze(0)

        if self.reference_process == "pinned_brownian":
            ref_log_var = (self.t_scale * ts).log().unsqueeze(2)  # (1, T', 1)
            log_p_ref = -0.5 * (logtwopi + ref_log_var + (-ref_log_var).exp() * (states**2)).sum(-1)
        else:
            log_p_ref = self.initial_logprob(states.reshape(-1, self.energy.ndim)).view(bsz, -1)

        partial_energy = (1 - betas) * log_p_ref + betas * self.energy.log_reward(
            states.reshape(-1, self.energy.ndim)
        ).view(bsz, -1).detach()
        return partial_energy  # (bsz, T')

    @maybe_compile
    def traj_step_fwd(
        self,
        s: torch.Tensor,  # state at time t
        t: torch.Tensor,  # (bsz,) time t
        step: int,  # step at time t
        detach: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pf_mean, pf_logvar, flow = self.pred_module.forward(s, t, self.energy.grad_log_reward)
        s_next, log_pf = self.forward_step(s, None, step, pf_mean, pf_logvar, detach=detach)

        t_next = t + self.dt
        mean_correction, var_correction = self.pred_module.backward(s_next, t_next)
        _, log_pb = self.backward_step(s, s_next, step, mean_correction, var_correction)
        return s_next, log_pf, log_pb, flow

    @maybe_compile
    def traj_step_bwd(
        self,
        s_next: torch.Tensor,  # state at time t + \Delta t
        t_next: torch.Tensor,  # (bsz,) time t + \Delta t
        step: int,  # step at time t
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_correction, var_correction = self.pred_module.backward(s_next, t_next)
        s, log_pb = self.backward_step(None, s_next, step, mean_correction, var_correction)

        t = t_next - self.dt
        pf_mean, pf_logvar, flow = self.pred_module.forward(s, t, self.energy.grad_log_reward)
        _, log_pf = self.forward_step(s, s_next, step, pf_mean, pf_logvar, detach=True)
        return s, log_pf, log_pb, flow

    def get_trajectory_fwd(
        self,
        batch_size: int,
        detach=True,
        subtraj_len: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = batch_size
        s = self.sample_initial_state(bsz)
        init_log_probs = self.initial_logprob(s)

        log_pfs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_pbs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_fs = torch.zeros((bsz, self.num_steps + 1), device=self.device)
        states = torch.zeros((bsz, self.num_steps + 1, self.energy.ndim), device=self.device)
        states[:, 0] = s

        for i in range(self.num_steps):  # from step 0 to self.num_steps - 1
            s = s.detach() if detach else s

            s_, log_pf, log_pb, flow = self.traj_step_fwd(s, self.times[i].expand(bsz), i, detach)

            if self.pred_module.conditional_flow_model and i > 0 and i % subtraj_len == 0:
                log_fs[:, i] = flow  # for i == 0, we use log Z + init_log_probs
            log_pfs[:, i] = log_pf
            log_pbs[:, i] = log_pb

            s = s_
            states[:, i + 1] = s

        if self.partial_energy:
            log_fs[:, subtraj_len:-1:subtraj_len] += self.get_partial_energy(
                states[:, subtraj_len:-1:subtraj_len],
                torch.arange(subtraj_len, self.num_steps, subtraj_len, device=self.device),
            )

        # Assign the terminal reward
        # Set terminal reward based on whether we need gradients for PIS loss
        with torch.no_grad() if detach else torch.enable_grad():
            log_fs[:, -1] = self.energy.log_reward(states[:, -1])

        return states, log_pfs, log_pbs, log_fs, init_log_probs

    def get_trajectory_bwd(
        self,
        s: torch.Tensor,
        log_r: torch.Tensor,  # (bsz,)
        subtraj_len: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]

        log_pfs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_pbs = torch.zeros((bsz, self.num_steps), device=self.device)
        log_fs = torch.zeros((bsz, self.num_steps + 1), device=self.device)
        states = torch.zeros((bsz, self.num_steps + 1, self.energy.ndim), device=self.device)
        log_fs[:, -1] = log_r
        states[:, -1] = s
        for i in range(self.num_steps - 1, -1, -1):  # from step T - 1 to 0
            s = s.detach()

            s_, log_pf, log_pb, flow = self.traj_step_bwd(s, self.times[i + 1].expand(bsz), i)

            if self.pred_module.conditional_flow_model and i > 0 and i % subtraj_len == 0:
                log_fs[:, i] = flow  # for i == 0, we use log Z + init_log_probs
            log_pfs[:, i] = log_pf
            log_pbs[:, i] = log_pb

            s = s_
            states[:, i] = s

        if self.partial_energy:
            log_fs[:, subtraj_len:-1:subtraj_len] += self.get_partial_energy(
                states[:, subtraj_len:-1:subtraj_len],
                torch.arange(subtraj_len, self.num_steps, subtraj_len, device=self.device),
            )

        init_log_probs = self.initial_logprob(s)

        return states, log_pfs, log_pbs, log_fs, init_log_probs

    @torch.no_grad()
    def get_trajectory_fwd_smc(
        self,
        batch_size: int,
        subtraj_len: int,
        sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        resample_threshold: float,
        target_ess: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates trajectories using Sequential Monte Carlo (SMC).
        The process is broken into subtrajectories. After each subtrajectory,
        importance weights are updated and resampling is performed if ESS is low.
        Args:
            batch_size (int): The number of particles (trajectories) to sample.
            subtraj_len (int): The number of steps in each SMC segment.
            sampling_func (Callable[[torch.Tensor, int, bool], torch.Tensor]): The sampling
            function to use for resampling.
            resample_threshold (float): The normalized ESS threshold to trigger resampling.
            target_ess (float): The target ESS.
        Returns:
            A tuple containing:
            - final_states (torch.Tensor): States of particles at the final step.
            - final_log_iws (torch.Tensor): Unnormalized log importance weights of final particles.
            - subtraj_log_fs (torch.Tensor): Log flows for intermediate steps.
        """
        bsz = batch_size

        # --- Initialization ---
        assert self.pred_module.conditional_flow_model
        assert self.num_steps % subtraj_len == 0, "num_steps must be divisible by subtraj_len"
        num_subtrajs = self.num_steps // subtraj_len

        s = self.sample_initial_state(bsz)
        init_log_probs = self.initial_logprob(s)

        # Tensors to store data in chunks
        subtraj_states = torch.zeros(
            (num_subtrajs, bsz, subtraj_len + 1, self.energy.ndim),
            device=self.device,
        )
        subtraj_log_pfs = torch.zeros((num_subtrajs, bsz, subtraj_len), device=self.device)
        subtraj_log_pbs = torch.zeros((num_subtrajs, bsz, subtraj_len), device=self.device)
        subtraj_log_fs = torch.zeros((num_subtrajs, bsz, subtraj_len + 1), device=self.device)
        subtraj_log_fs[0, :, 0] = init_log_probs

        # SMC variables
        log_iws = torch.full((bsz,), -math.log(bsz), device=self.device)
        logZ_est = torch.tensor(0.0, device=self.device)

        # --- Main Loop over Subtrajectories ---
        for i in range(num_subtrajs):
            start_step = i * subtraj_len
            end_step = start_step + subtraj_len

            # --- Inner Loop (Simulate one subtrajectory) ---
            for j in range(subtraj_len):
                s = s.detach()

                step = start_step + j
                subtraj_states[i, :, j, :] = s

                s_next, log_pfs, log_pbs, flow = self.traj_step_fwd(
                    s, self.times[step].expand(bsz), step, True
                )

                if j > 0:
                    subtraj_log_fs[i, :, j] = flow

                subtraj_log_pfs[i, :, j] = log_pfs
                subtraj_log_pbs[i, :, j] = log_pbs
                s = s_next

            subtraj_states[i, :, subtraj_len, :] = s
            if end_step == self.num_steps:
                next_log_f = self.energy.log_reward(s)
            else:
                t = self.times[end_step].expand(bsz)
                _, _, next_log_f = self.pred_module.forward(s, t, self.energy.grad_log_reward)
                if self.partial_energy:
                    next_log_f += self.get_partial_energy(
                        s.unsqueeze(1),
                        torch.tensor([end_step], device=self.device, dtype=torch.long),
                    ).squeeze(1)
            subtraj_log_fs[i, :, subtraj_len] = next_log_f

            # --- Update Importance Weights ---
            log_iws += (
                subtraj_log_fs[i, :, subtraj_len]
                + subtraj_log_pbs[i].sum(dim=1)
                - subtraj_log_fs[i, :, 0]
                - subtraj_log_pfs[i].sum(dim=1)
            ).detach()
            logZ_est += torch.logsumexp(log_iws, dim=0)

            # --- Resampling ---
            normalized_ess = ess(log_weights=log_iws.unsqueeze(1)).item() / bsz
            if end_step < self.num_steps and normalized_ess < resample_threshold:
                tempered_log_iws, temp = binary_search_smoothing(log_iws.unsqueeze(1), target_ess)
                tempered_log_iws = tempered_log_iws.squeeze(1)
                temp = temp.item()
                log_iws = log_iws * (1 - 1 / temp)
                indices = sampling_func(
                    tempered_log_iws.softmax(dim=0), log_iws.shape[0], replacement=True
                )
            else:
                indices = torch.arange(bsz, device=self.device)

            s = s[indices]
            log_iws = log_iws[indices]
            if i < num_subtrajs - 1:
                subtraj_log_fs[i + 1, :, 0] = subtraj_log_fs[i, indices, subtraj_len]
            log_iws = log_iws.log_softmax(dim=0)  # normalise

        # Final update to logZ and combine with log_iws
        final_log_iws = log_iws + logZ_est + math.log(bsz)

        return s, final_log_iws, subtraj_log_fs[-1, :, -1]
