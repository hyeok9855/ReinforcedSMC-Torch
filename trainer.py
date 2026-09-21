import warnings
from typing import Callable, Literal

import torch

from buffers import TerminalStateBuffer
from energies import ALDP, BaseEnergy, IntermediateEnergy
from losses import cal_subtb_coef_matrix, get_loss
from mcmcs import BaseMCMC
from models import GFN
from utils.eval_utils import density_metrics, distribution_distance_metrics
from utils.misc_utils import linear_annealing, logmeanexp
from utils.plot_utils import visualize
from utils.train_utils import CompositeOptimizer, CompositeScheduler


class Trainer:
    def __init__(
        self,
        energy: BaseEnergy,
        gfn_model: GFN,
        optimizer: CompositeOptimizer,
        scheduler: CompositeScheduler | None,
        clip_grad_norm: float,
        clip_logZ_grad_norm_ratio: float,
        loss_type: Literal["tb", "logvar", "db", "subtb", "tb-subtb", "rev_kl", "mle"],
        logZ_huber_delta: float,
        subtb_lambda: float,
        subtb_n_chunks: int,
        n_epochs: int,
        bwd_to_fwd_ratio: float,
        buffer: TerminalStateBuffer | None,
        prefill_epochs: int,
        batch_size: int,
        smc: bool,
        smc_sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        smc_resample_threshold: float,
        smc_target_ess: float,
        smc_every: int,
        mcmc: BaseMCMC | None,
        mcmc_freq: int,
        mcmc_batch_size: int,
        invtemp: float,
        invtemp_anneal: bool,
        init_log_Z: Literal["iw_elbo", "iw"] | float,
        eval_batch_size: int,
        plot_gt: bool,
        plot_t_idx: list[int],
    ):

        self.energy = energy
        self.gfn_model = gfn_model
        self.device = gfn_model.device

        # Optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.clip_logZ_grad_norm_ratio = clip_logZ_grad_norm_ratio

        # Loss
        self.loss_type = loss_type
        self.subtb_chunk_size = (
            gfn_model.num_steps // subtb_n_chunks
            if self.loss_type in ["subtb", "tb-subtb"] and subtb_n_chunks > 0
            else 1
        )
        self.subtb_coef_matrix = None
        self.logZ_huber_delta = logZ_huber_delta
        if loss_type == "subtb" and subtb_n_chunks == 0:
            self.subtb_coef_matrix = cal_subtb_coef_matrix(
                subtb_lambda, self.gfn_model.num_steps
            ).to(self.device)

        # Training parameters and buffer
        self.n_epochs = n_epochs
        if bwd_to_fwd_ratio < 1:
            self.bwd_to_fwd_ratio = None
            self.fwd_to_bwd_ratio = round(1 / bwd_to_fwd_ratio)
        else:
            self.bwd_to_fwd_ratio = int(bwd_to_fwd_ratio)
            self.fwd_to_bwd_ratio = None
        self.buffer = buffer
        self.prefill_epochs = prefill_epochs if self.buffer is not None else 0

        # Sampling parameters
        self.batch_size = batch_size

        # SMC
        self.smc = smc
        self.smc_sampling_func = smc_sampling_func
        self.smc_resample_threshold = smc_resample_threshold
        self.smc_target_ess = smc_target_ess
        self.smc_every = smc_every
        self.bwd_count = 0

        # MCMC
        self.mcmc = mcmc
        self.mcmc_freq = mcmc_freq
        self.mcmc_batch_size = mcmc_batch_size

        # Misc
        self._invtemp = invtemp
        self.invtemp_anneal = invtemp_anneal

        self.init_log_Z = init_log_Z

        # Eval and Plot
        self.eval_batch_size = eval_batch_size
        self.plot_gt = plot_gt
        self.plot_t_idx = plot_t_idx

    def get_invtemp(self, it: int) -> float:
        if not self.invtemp_anneal:
            return self._invtemp
        return linear_annealing(
            it, int(0.5 * self.n_epochs), self._invtemp, 1.0, descending=False, avoid_zero=True
        )

    def prefill(self) -> None:
        # Prefill the buffer at the beginning of training
        assert self.prefill_epochs > 0 and self.buffer is not None
        with torch.no_grad():
            for _ in range(self.prefill_epochs):
                if not self.smc:
                    states, log_pfs, log_pbs, log_fs, init_log_probs = (
                        self.gfn_model.get_trajectory_fwd(
                            self.batch_size,
                            detach=self.loss_type != "rev_kl",
                            subtraj_len=self.subtb_chunk_size,
                        )
                    )
                    xs = states[:, -1]
                    log_rs = log_fs[:, -1]
                    log_iws = log_rs + log_pbs.sum(-1) - (log_pfs.sum(-1) + init_log_probs)
                else:
                    xs, log_iws, log_rs = self.gfn_model.get_trajectory_fwd_smc(
                        self.batch_size,
                        self.subtb_chunk_size,
                        self.smc_sampling_func,
                        self.smc_resample_threshold,
                        self.smc_target_ess,
                    )
                self.buffer.add(xs=xs, log_rs=log_rs, log_iws=log_iws)

    def initialize_log_Z(self) -> None:
        if isinstance(self.init_log_Z, float):
            self.gfn_model.pred_module.set_log_Z(self.init_log_Z)
        else:
            # sample one batch to initialize log_Z
            with torch.no_grad():
                _, log_pfs, log_pbs, log_fs, init_log_probs = self.gfn_model.get_trajectory_fwd(
                    self.batch_size,
                    detach=self.loss_type != "rev_kl",
                    subtraj_len=self.subtb_chunk_size,
                )

            log_rs = log_fs[:, -1] * self.get_invtemp(1)
            log_iws = log_rs + log_pbs.sum(-1) - (log_pfs.sum(-1) + init_log_probs)

            if self.init_log_Z == "iw_elbo":
                init_log_Z_val = logmeanexp(log_iws).item()
            elif self.init_log_Z == "elbo":
                init_log_Z_val = log_iws.mean().item()
            else:
                raise ValueError(f"Invalid init_log_Z: {self.init_log_Z}")
            self.gfn_model.pred_module.set_log_Z(init_log_Z_val)

    def train_step(self, it: int) -> float:
        self.gfn_model.train()
        if it == 0:
            if self.prefill_epochs > 0:
                self.prefill()
            self.initialize_log_Z()

        if self.loss_type == "mle":
            loss = self.bwd_train_step(it)
        elif self.buffer is None:
            loss = self.fwd_train_step(it)
        else:  # self.buffer is not None
            if (self.bwd_to_fwd_ratio is not None and it % (self.bwd_to_fwd_ratio + 1) == 0) or (
                self.fwd_to_bwd_ratio is not None and it % (self.fwd_to_bwd_ratio + 1) != 0
            ):
                loss = self.fwd_train_step(it)
            else:
                loss = self.bwd_train_step(it)

        # MCMC buffer augmentation
        if self.mcmc is not None and it % self.mcmc_freq == 0:
            self.perform_mcmc()

        if loss.isinf() or loss > 1e28:
            print(f"Loss is inf or too large: {loss.item()}; skipping this batch...")
            return loss.item()
        elif loss.isnan():
            raise ValueError("Loss is NaN")

        loss.backward()
        logZ_abs = abs(self.gfn_model.pred_module.log_Z.item())
        try:
            self.optimizer.clip_grad_norm_(
                self.clip_grad_norm or float("inf"),
                self.clip_logZ_grad_norm_ratio * logZ_abs or float("inf"),
            )
        except RuntimeError:
            print(f"Non-finite gradients at it={it} (loss={loss.item():.4g}); skipping update...")
            self.gfn_model.zero_grad()
            return loss.item()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.gfn_model.zero_grad()

        return loss.item()

    def fwd_train_step(self, it: int) -> torch.Tensor:
        # Forward sampling
        states, log_pfs, log_pbs, log_fs, init_log_probs = self.gfn_model.get_trajectory_fwd(
            self.batch_size, detach=self.loss_type != "rev_kl", subtraj_len=self.subtb_chunk_size
        )

        # Compute losses
        losses = get_loss(
            self.loss_type,
            log_pfs,
            log_pbs,
            log_fs,
            init_log_probs,
            log_Z=self.gfn_model.pred_module.log_Z,
            invtemp=self.get_invtemp(it),
            subtb_coef_matrix=self.subtb_coef_matrix,
            subtb_chunk_size=self.subtb_chunk_size,
            logZ_huber_delta=self.logZ_huber_delta,
            ndim=self.energy.ndim,
        )

        # Add data to buffer
        if self.buffer is not None:
            self.buffer.add(
                xs=states[:, -1],
                log_rs=log_fs[:, -1],
                log_iws=log_fs[:, -1] + log_pbs.sum(-1) - (log_pfs.sum(-1) + init_log_probs),
                losses=losses,
            )

        loss = losses.mean()
        return loss

    def bwd_train_step(self, it: int) -> torch.Tensor:
        if self.loss_type == "mle":
            gt_xs, gt_log_rewards = self.energy.cached_sample(self.batch_size, seed=it)
            _, log_pfs, log_pbs, log_fs, init_log_probs = self.gfn_model.get_trajectory_bwd(
                gt_xs, gt_log_rewards, subtraj_len=self.subtb_chunk_size
            )
            # mle over trajectories
            loss = -log_pfs.sum(-1).mean()

        else:  # self.loss_type != "mle"
            assert self.buffer is not None
            assert self.subtb_chunk_size > 0

            # SMC sampling and adding to buffer if smc is True
            if self.smc and self.bwd_count % self.smc_every == 0:
                xs, log_iws, log_rs = self.gfn_model.get_trajectory_fwd_smc(
                    self.batch_size,
                    self.subtb_chunk_size,
                    self.smc_sampling_func,
                    self.smc_resample_threshold,
                    self.smc_target_ess,
                )
                self.buffer.add(xs=xs, log_rs=log_rs, log_iws=log_iws, losses=None)
            self.bwd_count += 1

            # Buffer sampling
            buf_xs, buf_log_rs, indices = self.buffer.sample(self.batch_size)
            # each with shape (bs,)

            # Construct complete trajectories
            _, log_pfs, log_pbs, log_fs, init_log_probs = self.gfn_model.get_trajectory_bwd(
                buf_xs, buf_log_rs, subtraj_len=self.subtb_chunk_size
            )

            losses = get_loss(
                self.loss_type,
                log_pfs,
                log_pbs,
                log_fs,
                init_log_probs,
                log_Z=self.gfn_model.pred_module.log_Z,
                invtemp=self.get_invtemp(it),
                subtb_coef_matrix=self.subtb_coef_matrix,
                subtb_chunk_size=self.subtb_chunk_size,
                logZ_huber_delta=self.logZ_huber_delta,
                ndim=self.energy.ndim,
            )

            if self.buffer.prioritization == "loss":
                self.buffer.update(indices, losses=losses)

            loss = losses.mean()

        return loss

    def perform_mcmc(self) -> None:
        # TODO: support for intermediate states
        assert self.mcmc is not None
        assert isinstance(self.buffer, TerminalStateBuffer) and len(self.buffer) > 0
        assert self.buffer.prioritization != "loss"

        buf_xs, _, indices = self.buffer.sample(self.mcmc_batch_size)

        # Augment the buffer with samples from MCMC
        mcmc_xs, mcmc_log_rs = self.mcmc.sample(buf_xs)
        indices = indices.unsqueeze(0).repeat(mcmc_xs.shape[0], 1)

        mcmc_xs = mcmc_xs.reshape(-1, self.energy.ndim)
        mcmc_log_rs = mcmc_log_rs.reshape(-1)
        indices = indices.reshape(-1)

        if isinstance(self.energy, ALDP):
            ind_L = self.energy.get_lform_indices(mcmc_xs)
            mcmc_xs = mcmc_xs[ind_L]
            mcmc_log_rs = mcmc_log_rs[ind_L]
            indices = indices[ind_L]

        if mcmc_xs.isnan().any():  # This happens rarely when mcmc is MD
            valid_mask = ~mcmc_xs.isnan().any(1)
            mcmc_xs = mcmc_xs[valid_mask]
            mcmc_log_rs = mcmc_log_rs[valid_mask]
            indices = indices[valid_mask]

        self.buffer.add(
            xs=mcmc_xs,
            log_rs=mcmc_log_rs,
            log_iws=self.buffer.priority_dataset.data[indices],
        )

    def eval_and_plot(
        self,
        data_size: int,
        full_eval: bool,
        final_eval: bool = False,
        plot: bool = False,
    ) -> dict:
        metrics, model_trajs, buffer_xs = self.eval_step(data_size, full_eval, final_eval)
        if plot:
            images = self.plot_step(model_trajs, buffer_xs, self.plot_gt or final_eval)
            metrics.update(images)
            self.plot_gt = False  # disable plotting gt after first plot
        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        data_size: int,
        full_eval: bool = False,
        final_eval: bool = False,
    ) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
        self.gfn_model.eval()

        metrics = {}

        eval_batch_size = min(self.eval_batch_size, data_size)

        with torch.no_grad():
            divisible = data_size % eval_batch_size == 0
            n_epochs = data_size // eval_batch_size + (1 if not divisible else 0)

            model_trajs, log_pfs, log_pbs, log_rs, init_log_probs = [], [], [], [], []

            for i in range(n_epochs):
                _model_trajs, _log_pfs, _log_pbs, _log_fs, _init_log_probs = (
                    self.gfn_model.get_trajectory_fwd(
                        self.eval_batch_size,
                        detach=self.loss_type != "rev_kl",
                        subtraj_len=self.subtb_chunk_size,
                    )
                )
                model_trajs.append(_model_trajs)
                log_pfs.append(_log_pfs)
                log_pbs.append(_log_pbs)
                log_rs.append(_log_fs[:, -1])
                init_log_probs.append(_init_log_probs)
            model_trajs = torch.cat(model_trajs, dim=0)[:data_size]
            sample_xs = model_trajs[:, -1]
            log_pfs = torch.cat(log_pfs, dim=0)[:data_size]
            log_pbs = torch.cat(log_pbs, dim=0)[:data_size]
            log_rs = torch.cat(log_rs, dim=0)[:data_size]
            init_log_probs = torch.cat(init_log_probs, dim=0)[:data_size]

            try:
                gt_xs, gt_log_rs = self.energy.cached_sample(data_size)
                gt_log_pfs, gt_log_pbs, gt_init_log_probs = [], [], []
                for i in range(n_epochs):
                    gt_xs_batch = gt_xs[i * eval_batch_size : (i + 1) * eval_batch_size]
                    gt_log_rewards_batch = gt_log_rs[
                        i * eval_batch_size : (i + 1) * eval_batch_size
                    ]
                    _, _log_pfs, _log_pbs, _, _init_log_probs = self.gfn_model.get_trajectory_bwd(
                        gt_xs_batch, gt_log_rewards_batch, subtraj_len=self.subtb_chunk_size
                    )
                    gt_log_pfs.append(_log_pfs)
                    gt_log_pbs.append(_log_pbs)
                    gt_init_log_probs.append(_init_log_probs)
                gt_log_pfs = torch.cat(gt_log_pfs, dim=0)[:data_size]
                gt_log_pbs = torch.cat(gt_log_pbs, dim=0)[:data_size]
                gt_init_log_probs = torch.cat(gt_init_log_probs, dim=0)[:data_size]
            except NotImplementedError:
                gt_xs = gt_log_rs = gt_log_pfs = gt_log_pbs = gt_init_log_probs = None

        try:
            gt_log_Z = self.energy.gt_logz()
        except NotImplementedError:
            gt_log_Z = None

        metrics.update(
            density_metrics(
                log_pfs,
                log_pbs,
                log_rs,
                init_log_probs,
                gt_log_pfs=gt_log_pfs,
                gt_log_pbs=gt_log_pbs,
                gt_log_rewards=gt_log_rs,
                gt_init_log_probs=gt_init_log_probs,
                gt_log_Z=gt_log_Z,
            )
        )

        if gt_xs is not None and full_eval:
            # "1-Wasserstein", "2-Wasserstein", "Linear_MMD", "Poly_MMD", "RBF_MMD",
            # "Mean_MSE", "Mean_L2", "Mean_L1", "Median_MSE", "Median_L2", "Median_L1"
            metrics.update(distribution_distance_metrics(sample_xs, gt_xs))

        metrics = {f"eval/{k}": v for k, v in metrics.items()}

        buffer_xs = None
        if self.buffer is not None and len(self.buffer) > 0:
            assert gt_xs is not None
            buffer_xs, _, _ = self.buffer.sample(data_size)
            if full_eval:
                metrics_b = distribution_distance_metrics(buffer_xs, gt_xs)
                metrics_b = {f"eval_buffer/{k}": v for k, v in metrics_b.items()}
                metrics.update(metrics_b)

        if final_eval:
            metrics = {k.replace("eval", "final_eval"): v for k, v in metrics.items()}

        return metrics, model_trajs, buffer_xs

    @torch.no_grad()
    def sample_terminal(self, n: int) -> torch.Tensor:
        """Draw ``n`` terminal states in eval-sized batches, keeping only x_T on the CPU.

        ``eval_step`` concatenates whole trajectories, which is (n, T+1, ndim) and reaches
        several GB for the high-dimensional targets; this keeps only what is worth saving.
        """
        self.gfn_model.eval()
        chunks, drawn = [], 0
        while drawn < n:
            bsz = min(self.eval_batch_size, n - drawn)
            trajs, *_ = self.gfn_model.get_trajectory_fwd(
                bsz, detach=True, subtraj_len=self.subtb_chunk_size
            )
            chunks.append(trajs[:, -1].detach().cpu())
            drawn += bsz
        return torch.cat(chunks, dim=0)[:n]

    @torch.no_grad()
    def plot_step(
        self,
        model_trajs: torch.Tensor,
        buffer_xs: torch.Tensor | None = None,
        plot_gt: bool = False,
    ) -> dict:
        xs = model_trajs[:, -1]
        images = visualize(self.energy, xs)
        if buffer_xs is not None:
            images.update(visualize(self.energy, buffer_xs, suffix="_buffer"))
        if plot_gt:
            try:
                gt_xs, _ = self.energy.cached_sample(model_trajs.shape[0])
                images.update(visualize(self.energy, gt_xs, suffix="_gt"))
            except NotImplementedError:
                warnings.warn(
                    f"Ground-truth samples are not available for {self.energy.__class__.__name__}."
                    "Skipping plotting of ground-truth samples."
                )

        # Plot intermediate states
        if len(self.plot_t_idx) > 0:
            assert self.gfn_model.pred_module.conditional_flow_model
            for t_idx in self.plot_t_idx:
                inter_states = model_trajs[:, t_idx]
                inter_energy = IntermediateEnergy(self.energy, self.gfn_model, t_idx)
                images.update(
                    visualize(
                        inter_energy,
                        inter_states,
                        suffix=f"-t{round(t_idx / self.gfn_model.num_steps, 3)}",
                    )
                )

        return images
