from typing import Literal

import torch

from buffers import TerminalStateBuffer
from energies import ALDP, BaseEnergy
from losses import get_loss
from mcmcs import MALA, BaseMCMC
from models import GFN
from trainer import Trainer
from utils.eval_utils import distribution_distance_metrics
from utils.plot_utils import visualize
from utils.sampling_utils import multinomial
from utils.train_utils import CompositeOptimizer, CompositeScheduler


class LocalSearchTrainer(Trainer):
    # Abort if this many consecutive local-search rounds accept no proposal while ``ls_buffer``
    # is still empty (gfn-diffusion has no fallback at all and errors on an empty buffer).
    MAX_EMPTY_LS_ROUNDS = 3

    def __init__(
        self,
        energy: BaseEnergy,
        gfn_model: GFN,
        optimizer: CompositeOptimizer,
        scheduler: CompositeScheduler | None,
        clip_grad_norm: float,
        clip_logZ_grad_norm_ratio: float,
        loss_type: Literal["tb", "logvar"],
        logZ_huber_delta: float,
        n_epochs: int,
        bwd_to_fwd_ratio: float,
        buffer: TerminalStateBuffer,
        ls_buffer: TerminalStateBuffer,
        prefill_epochs: int,
        batch_size: int,
        mcmc: BaseMCMC,
        mcmc_freq: int,
        mcmc_batch_size: int,
        invtemp: float,
        invtemp_anneal: bool,
        init_log_Z: Literal["iw_elbo", "elbo"] | float,
        eval_batch_size: int,
        plot_gt: bool,
        plot_t_idx: list[int],
    ) -> None:
        assert loss_type in ("tb", "logvar"), "This baseline supports TB and VarGrad (logvar) only"
        assert mcmc_freq > 0 and mcmc_batch_size > 0
        assert buffer.prioritization != "loss" and ls_buffer.prioritization != "loss"
        assert bwd_to_fwd_ratio >= 1 or prefill_epochs > 0, (
            "bwd_to_fwd_ratio < 1 makes Trainer.train_step start with a backward step, which needs "
            "a non-empty forward buffer; set prefill_epochs >= 1"
        )

        super().__init__(
            energy=energy,
            gfn_model=gfn_model,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_grad_norm=clip_grad_norm,
            clip_logZ_grad_norm_ratio=clip_logZ_grad_norm_ratio,
            loss_type=loss_type,
            logZ_huber_delta=logZ_huber_delta,
            subtb_lambda=0.0,  # unused for tb / logvar
            subtb_n_chunks=0,  # unused for tb / logvar
            n_epochs=n_epochs,
            bwd_to_fwd_ratio=bwd_to_fwd_ratio,
            buffer=buffer,
            prefill_epochs=prefill_epochs,
            batch_size=batch_size,
            # SMC and the parent's in-place MCMC augmentation are disabled
            smc=False,
            smc_sampling_func=multinomial,
            smc_resample_threshold=0.0,
            smc_target_ess=0.0,
            smc_every=1,
            mcmc=None,
            mcmc_freq=mcmc_freq,
            mcmc_batch_size=mcmc_batch_size,
            invtemp=invtemp,
            invtemp_anneal=invtemp_anneal,
            init_log_Z=init_log_Z,
            eval_batch_size=eval_batch_size,
            plot_gt=plot_gt,
            plot_t_idx=plot_t_idx,
        )

        # Local search
        self.ls_buffer = ls_buffer
        self.ls_mcmc = mcmc
        self.last_ls_it: int | None = None
        self.n_empty_ls_rounds = 0  # consecutive LS rounds that accepted no proposal
        self.ls_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Local search                                                       #
    # ------------------------------------------------------------------ #
    def run_mcmc(self, init_xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MCMC from ``init_xs``; return flat states, log-rewards and chain indices."""
        if isinstance(self.ls_mcmc, MALA):  # flat (N, ndim), (N,) + chain index per sample
            xs, log_rs, chain_idx = self.ls_mcmc.sample(init_xs, return_indices=True)
        else:  # MD-style output with a leading chain axis: (T, bsz, ndim), (T, bsz)
            xs, log_rs = self.ls_mcmc.sample(init_xs)
            assert xs.ndim == 3 and xs.shape[1] == init_xs.shape[0], (
                "MCMC output must keep the chain axis so samples can be mapped to their chain"
            )
            chain_idx = torch.arange(init_xs.shape[0], device=init_xs.device).repeat(xs.shape[0])
        return xs.reshape(-1, self.energy.ndim), log_rs.reshape(-1), chain_idx.reshape(-1)

    def perform_local_search(self) -> None:
        """Refine a batch from ``buffer`` with MCMC; push accepted states into ``ls_buffer``."""
        assert self.buffer is not None and len(self.buffer) > 0, "Forward buffer is empty"
        init_xs, _, indices = self.buffer.sample(self.mcmc_batch_size)
        ls_xs, ls_log_rs, chain_idx = self.run_mcmc(init_xs)
        indices = indices[chain_idx]  # ``buffer`` entry each MCMC sample descends from

        if isinstance(self.energy, ALDP):  # keep L-form conformers only, as in ``perform_mcmc``
            ind_L = self.energy.get_lform_indices(ls_xs)
            ls_xs, ls_log_rs, indices = ls_xs[ind_L], ls_log_rs[ind_L], indices[ind_L]

        valid = ~(ls_xs.isnan().any(dim=1) | ls_log_rs.isnan())
        ls_xs, ls_log_rs, indices = ls_xs[valid], ls_log_rs[valid], indices[valid]

        n_new = ls_xs.shape[0]
        self.n_empty_ls_rounds = self.n_empty_ls_rounds + 1 if n_new == 0 else 0
        if n_new > 0:
            # MCMC samples inherit the priority (e.g. importance weight) of their starting point.
            self.ls_buffer.add(
                xs=ls_xs,
                log_rs=ls_log_rs,
                log_iws=self.buffer.priority_dataset.data[indices],
            )

        self.ls_metrics = {
            "local_search/n_accepted": float(n_new),
            "local_search/n_empty_rounds": float(self.n_empty_ls_rounds),
            "local_search/n_accepted_per_chain": n_new / self.mcmc_batch_size,
            "local_search/mean_log_r": ls_log_rs.mean().item() if n_new > 0 else float("nan"),
            "local_search/buffer_size": float(len(self.ls_buffer)),
        }

    def pop_ls_metrics(self) -> dict[str, float]:
        """Return (and clear) the metrics of the most recent local-search round, if any."""
        metrics, self.ls_metrics = self.ls_metrics, {}
        return metrics

    def bwd_train_step(self, it: int) -> torch.Tensor:
        assert self.buffer is not None and len(self.buffer) > 0, "Forward buffer is empty"

        if len(self.ls_buffer) == 0 or (it // self.mcmc_freq != self.last_ls_it // self.mcmc_freq):
            self.last_ls_it = it
            self.perform_local_search()

        if len(self.ls_buffer) > 0:
            buf_xs, buf_log_rs, _ = self.ls_buffer.sample(self.batch_size)
        else:  # no MCMC proposal has been accepted yet; bounded fallback to the forward buffer
            if self.n_empty_ls_rounds >= self.MAX_EMPTY_LS_ROUNDS:
                raise RuntimeError(
                    f"Local search accepted 0 proposals in {self.n_empty_ls_rounds} consecutive "
                    "rounds and the local-search buffer is still empty. Check the MCMC settings "
                    "(e.g. mcmc_step_size); continuing would silently train from the forward "
                    "buffer."
                )
            print(
                f"[it={it}] Local-search buffer is empty ({self.n_empty_ls_rounds} empty "
                "round(s)); sampling from the forward buffer instead."
            )
            buf_xs, buf_log_rs, _ = self.buffer.sample(self.batch_size)

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
            logZ_huber_delta=self.logZ_huber_delta,
            ndim=self.energy.ndim,
        )
        return losses.mean()

    # ------------------------------------------------------------------ #
    # Eval & plot: also report the dedicated local-search buffer         #
    # ------------------------------------------------------------------ #
    def eval_and_plot(
        self,
        data_size: int,
        full_eval: bool,
        final_eval: bool = False,
        plot: bool = False,
    ) -> dict:
        metrics, model_trajs, buffer_xs = self.eval_step(data_size, full_eval, final_eval)

        ls_buffer_xs = None
        if len(self.ls_buffer) > 0:
            with torch.no_grad():
                ls_buffer_xs, _, _ = self.ls_buffer.sample(data_size)
            if full_eval:
                try:
                    gt_xs, _ = self.energy.cached_sample(data_size)
                except NotImplementedError:
                    gt_xs = None
                if gt_xs is not None:
                    prefix = "final_eval_buffer_ls" if final_eval else "eval_buffer_ls"
                    metrics_ls = distribution_distance_metrics(ls_buffer_xs, gt_xs)
                    metrics.update({f"{prefix}/{k}": v for k, v in metrics_ls.items()})

        if plot:
            images = self.plot_step(model_trajs, buffer_xs, self.plot_gt or final_eval)
            if ls_buffer_xs is not None:
                images.update(visualize(self.energy, ls_buffer_xs, suffix="_buffer_ls"))
            metrics.update(images)
            self.plot_gt = False  # disable plotting gt after first plot
        return metrics
