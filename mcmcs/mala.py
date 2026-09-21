from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import trange

from mcmcs.base import BaseMCMC

if TYPE_CHECKING:
    from energies import BaseEnergy


class MALA(BaseMCMC):
    def __init__(
        self,
        energy: "BaseEnergy",
        burn_in: int = 100,
        n_steps: int = 1000,
        thinning: int = 1,
        step_size: float = 0.1,
        ld_schedule: bool = True,
        target_acceptance_rate: float = 0.574,
        **kwargs,
    ):
        super().__init__(energy)
        assert thinning >= 1, f"thinning must be >= 1, got {thinning}"
        self.burn_in = burn_in
        self.n_steps = n_steps
        self.thinning = thinning
        self.step_size = step_size
        self.ld_schedule = ld_schedule
        self.target_acceptance_rate = target_acceptance_rate

    def adjust_ld_step(
        self,
        ld_step: float,
        acceptance_rate: float,
        adjustment_factor: float = 0.01,
    ):
        if acceptance_rate > self.target_acceptance_rate:
            return ld_step + adjustment_factor * ld_step
        else:
            return ld_step - adjustment_factor * ld_step

    def sample(self, x, return_indices: bool = False):
        accepted_samples = []
        accepted_logr = []
        accepted_idx = []
        acceptance_rate_lst = []
        chain_idx = torch.arange(x.shape[0], device=x.device)
        log_r_original = self.energy.log_reward(x)
        acceptance_count = 0
        acceptance_rate = 0
        total_proposals = 0

        ld_step = self.step_size
        for i in trange(self.n_steps, desc="[MALA]", dynamic_ncols=True):
            x = x.requires_grad_(True)

            log_rs = self.energy.log_reward(x)
            r_grad_original = torch.autograd.grad(log_rs.sum(), x)[0].detach()
            if self.ld_schedule and i > 0:
                ld_step = self.adjust_ld_step(ld_step, acceptance_rate)

            new_x = x + ld_step * r_grad_original + np.sqrt(2 * ld_step) * torch.randn_like(x)
            log_r_new = self.energy.log_reward(new_x)
            r_grad_new = torch.autograd.grad(log_r_new.sum(), new_x)[0].detach()

            with torch.no_grad():
                log_q_fwd = -(
                    torch.norm(new_x - x - ld_step * r_grad_original, p=2, dim=1) ** 2
                ) / (4 * ld_step)
                log_q_bwd = -(torch.norm(x - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (
                    4 * ld_step
                )

                log_accept = (log_r_new - log_r_original) + log_q_bwd - log_q_fwd
                accept_mask = torch.rand(x.shape[0], device=x.device) < torch.exp(
                    torch.clamp(log_accept, max=0)
                )
                acceptance_count += accept_mask.sum().item()
                total_proposals += x.shape[0]

                x = x.detach()
                # After burn-in, keep every ``thinning``-th step
                if i > self.burn_in and (i - self.burn_in - 1) % self.thinning == 0:
                    accepted_samples.append(new_x[accept_mask].detach())
                    accepted_logr.append(log_r_new[accept_mask].detach())
                    accepted_idx.append(chain_idx[accept_mask])
                x[accept_mask] = new_x[accept_mask]
                log_r_original[accept_mask] = log_r_new[accept_mask]

                if i % 5 == 0:
                    acceptance_rate = acceptance_count / total_proposals
                    if i > self.burn_in:
                        acceptance_rate_lst.append(acceptance_rate)
                    acceptance_count = 0
                    total_proposals = 0

        xs = torch.cat(accepted_samples, dim=0)
        log_rs = torch.cat(accepted_logr, dim=0)
        if return_indices:
            return xs, log_rs, torch.cat(accepted_idx, dim=0)
        return xs, log_rs
