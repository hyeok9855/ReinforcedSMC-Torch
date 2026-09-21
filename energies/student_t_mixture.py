from functools import partial

import numpy as np
import torch
import torch.distributions as D
from matplotlib import pyplot as plt

import wandb
from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.plot_utils import fig_to_image, sliced_log_reward, viz_energy_hist


class StudentTMixture(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device,
        ndim: int = 2,
        num_components: int = 10,
        degree_of_freedom: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__(device=device, ndim=ndim, seed=seed, plot_bound=15)

        try:
            locs = torch.from_numpy(np.load(f"energies/data/mos-{ndim}d_locs.npy"))
        except FileNotFoundError:
            with temp_seed(seed):
                locs = (torch.rand(num_components, ndim) * 2 - 1) * 10  # 10 from Beyond ELBOs
        locs = locs.to(dtype=torch.float32, device=device)

        dofs = torch.ones((num_components, ndim), device=device) * degree_of_freedom
        scales = torch.ones((num_components, ndim), device=device)
        logits = torch.ones(num_components, device=device)  # uniform, default in Beyond ELBOs

        mixture_dist = D.Categorical(logits=logits)
        components_dist = D.Independent(
            D.StudentT(loc=locs, scale=scales, df=dofs),
            reinterpreted_batch_ndims=1,  # type: ignore
        )
        self.distribution = D.MixtureSameFamily(mixture_dist, components_dist)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._log_prob(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            return self.distribution.sample(sample_shape=torch.Size((batch_size,))).to(self.device)

    def gt_logz(self):
        return 0.0

    # ----- Energy-specific methods ----- #
    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)

        log_prob = self.distribution.log_prob(x)

        if not batched:
            log_prob = log_prob.squeeze(0)

        return log_prob

    def visualize(
        self, samples: torch.Tensor, weights: torch.Tensor | None = None, **kwargs
    ) -> dict:
        if self.ndim != 2:
            samples = samples[:, :2]
            logp_func = partial(sliced_log_reward, energy=self, dims=(0, 1))
        else:
            logp_func = self.log_reward

        boarder = [-self.plot_bound, self.plot_bound]
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        x, y = torch.meshgrid(
            torch.linspace(boarder[0], boarder[1], 100), torch.linspace(boarder[0], boarder[1], 100)
        )
        grid = torch.cat([x.ravel().unsqueeze(1), y.ravel().unsqueeze(1)], dim=1)
        pdf_values = torch.exp(logp_func(grid))
        pdf_values = pdf_values.reshape(x.shape)
        ax.contourf(x, y, pdf_values, levels=50)

        samples = samples.detach().cpu()
        weights = weights.detach().cpu() if weights is not None else None
        plt.scatter(
            samples[:, 0],
            samples[:, 1],
            c="r",
            alpha=0.5,
            marker="x",
            s=weights[: len(samples)] * len(samples) * 5 if weights is not None else 5,
        )
        ax.set_xlim(boarder[0], boarder[1])
        ax.set_ylim(boarder[0], boarder[1])

        out_dict = {"visualization/contour01": wandb.Image(fig_to_image(fig))}
        out_dict.update(viz_energy_hist(self, samples))
        return out_dict
