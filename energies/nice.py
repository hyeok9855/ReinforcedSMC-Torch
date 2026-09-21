"""NICE (Dinh et al., 2014) flow densities used as sampling targets.

The flow weights are the VSM-published checkpoints
(https://github.com/DenisBless/variational_sampling_methods), converted from
FLAX to torch state dicts. Ported from 1step_energy_sampler
(`src/energy_sampler/nice.py`), which is where the conversion was done.

The target is the *pushforward density of the trained flow*, so it is exactly
normalized: `gt_logz() == 0`, and exact ground-truth samples are available by
pushing gaussian noise through the inverse flow.
"""

import contextlib
import math
from pathlib import Path

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import wandb
from energies.base import BaseEnergy
from utils.misc_utils import maybe_compile, temp_seed
from utils.plot_utils import fig_to_image, viz_energy_hist

DATA_PATH = Path(__file__).parent / "data" / "nice"

_CHECKPOINTS = {
    "mnist": DATA_PATH / "params_nice_mnist_14x14.pt",
    "fashion_mnist": DATA_PATH / "params_nice_fashion_mnist_28x28.pt",
}
_IMAGE_SIZE = {"mnist": 14, "fashion_mnist": 28}


@contextlib.contextmanager
def matmul_precision(precision: str):
    """Temporarily set `torch.set_float32_matmul_precision`."""
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old)


class NICEFlow(nn.Module):
    """NICE with additive coupling, full-reversal permutation and diagonal scaling.

    `h_depth` counts every `nn.Linear` in a coupling net, whereas the VSM repo
    counts only the hidden ones -- their `h_depth=k` is our `h_depth=k+1`. The
    defaults here match the published checkpoints; do not change them unless
    you also retrain.
    """

    def __init__(self, ndim: int, n_steps: int = 4, h_depth: int = 5, h_dim: int = 1000) -> None:
        super().__init__()
        self.ndim = ndim
        self.split = ndim // 2 + (1 if ndim % 2 == 1 else 0)

        self.register_buffer("part", torch.arange(ndim - 1, -1, -1))
        self.logscale = nn.Parameter(torch.zeros(ndim))

        self.nets = nn.ModuleList()
        for _ in range(n_steps):
            layers: list[nn.Module] = []
            curr_in = self.split
            for j in range(h_depth):
                if j != h_depth - 1:
                    layers.append(nn.Linear(curr_in, h_dim))
                    layers.append(nn.ReLU())
                    curr_in = h_dim
                else:
                    layers.append(nn.Linear(curr_in, ndim - self.split))
            self.nets.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x -> z (inference direction)."""
        for net in self.nets:
            x = x[:, self.part]
            xa, xb = x[:, : self.split], x[:, self.split :]
            x = torch.cat([xa, xb + net(xa)], dim=-1)
        return x

    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """z -> x (generation direction)."""
        for net in reversed(self.nets):
            ya, yb = z[:, : self.split], z[:, self.split :]
            z = torch.cat([ya, yb - net(ya)], dim=-1)[:, self.part]
        return z

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        zs = self.forward(x) * torch.exp(self.logscale)
        log_pz = -0.5 * (zs.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        return log_pz + self.logscale.sum()

    def sample(self, batch_size: int) -> torch.Tensor:
        zs = torch.randn(
            batch_size, self.ndim, device=self.logscale.device, dtype=self.logscale.dtype
        )
        return self.reverse(zs / torch.exp(self.logscale))


class NICE(BaseEnergy):
    def __init__(
        self,
        device: str | torch.device,
        target: str,
        seed: int = 0,
    ) -> None:
        if target not in _CHECKPOINTS:
            raise ValueError(f"target must be one of {sorted(_CHECKPOINTS)} (got {target!r})")
        self.target = target
        self.im_size = _IMAGE_SIZE[target]
        ndim = self.im_size**2
        super().__init__(device=device, ndim=ndim, seed=seed, plot_bound=1.0)

        path = _CHECKPOINTS[target]
        if not path.exists():
            raise FileNotFoundError(
                f"NICE checkpoint {path} is missing. It ships with the repo; if this is a fresh "
                "clone, check that the file was fetched rather than left as a pointer stub."
            )
        self.flow = NICEFlow(ndim)
        self.flow.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        # The flow defines the target: freeze it so `grad_log_reward` differentiates
        # only w.r.t. x.
        self.flow.eval().requires_grad_(False).to(device)

        self._score_precision = "highest"
        if torch.device(self.device).type == "cuda" and torch.get_default_dtype() == torch.float32:
            self._score_precision = "high"

        def _score(x: torch.Tensor) -> torch.Tensor:
            return torch.func.grad(lambda s: self.flow.log_prob(s).sum())(x)

        self._log_prob_fn = maybe_compile(self.flow.log_prob)
        self._score_fn = maybe_compile(_score)

    def grad_log_reward(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        x = x.detach() if batched else x.detach().unsqueeze(0)
        with matmul_precision(self._score_precision):
            grad = self._score_fn(x).detach()
        return grad if batched else grad.squeeze(0)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self._log_prob(x)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            return self.flow.sample(batch_size).to(self.device)

    def gt_logz(self) -> float:
        return 0.0  # the flow density is normalized by construction

    # ----- Energy-specific methods ----- #
    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)

        log_prob = self._log_prob_fn(x)

        if not batched:
            log_prob = log_prob.squeeze(0)

        return log_prob

    def _image_grid(self, samples: torch.Tensor, title: str) -> plt.Figure:
        n_show = min(64, samples.shape[0])
        n_rows = math.ceil(math.sqrt(n_show)) if n_show > 0 else 1
        imgs = samples[:n_show].detach().cpu().reshape(n_show, self.im_size, self.im_size)

        fig, axes = plt.subplots(n_rows, n_rows, figsize=(8, 8), squeeze=False)
        for k, ax in enumerate(axes.ravel()):
            if k < n_show:
                ax.imshow(imgs[k], cmap="gray")
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def visualize(
        self, samples: torch.Tensor, weights: torch.Tensor | None = None, **kwargs
    ) -> dict:
        # An image grid can't express per-sample weights; show the highest-weight
        # samples instead of a uniform prefix so the panel stays representative.
        if weights is not None:
            k = min(weights.shape[0], samples.shape[0])
            samples = samples[torch.argsort(weights[:k], descending=True)]

        gt_xs, _ = self.cached_sample(samples.shape[0])
        out_dict = {
            "visualization/samples": wandb.Image(fig_to_image(self._image_grid(samples, "Model"))),
            "visualization/gt_samples": wandb.Image(
                fig_to_image(self._image_grid(gt_xs, "Ground truth"))
            ),
        }
        out_dict.update(viz_energy_hist(self, samples))
        return out_dict


class NICEMNIST(NICE):
    def __init__(self, device: str | torch.device, seed: int = 0) -> None:
        super().__init__(device=device, target="mnist", seed=seed)


class NICEFashionMNIST(NICE):
    def __init__(self, device: str | torch.device, seed: int = 0) -> None:
        super().__init__(device=device, target="fashion_mnist", seed=seed)
