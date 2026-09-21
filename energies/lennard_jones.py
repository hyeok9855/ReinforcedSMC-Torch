import warnings
from pathlib import Path

import numpy as np
import torch

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.particle_system import interatomic_distance, remove_mean
from utils.plot_utils import viz_energy_hist, viz_interatomic_dist_hist

DATA_PATH = Path(__file__).parent / "data"


def lennard_jones_energy(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJones(BaseEnergy):
    is_particle_system = True

    def __init__(
        self,
        spatial_dim: int,
        n_particles: int,
        device: str | torch.device,
        data_path: Path,
        epsilon: float = 1.0,
        min_radius: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 1.0,
        energy_factor: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(device=device, ndim=spatial_dim * n_particles, seed=seed, plot_bound=1.0)

        self.spatial_dim = spatial_dim
        self.n_particles = n_particles

        self.epsilon = epsilon
        self.min_radius = min_radius
        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale
        self.energy_factor = energy_factor

        self.h_initial = torch.ones(n_particles, device=device).unsqueeze(1)

        data = torch.tensor(np.load(data_path))
        self.approx_sample = remove_mean(data, self.n_particles, self.spatial_dim)

    def energy(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        # dists is a tensor of shape [..., n_particles * (n_particles - 1) // 2]
        dists = interatomic_distance(x, self.n_particles, self.spatial_dim)

        lj_energies = lennard_jones_energy(dists, self.epsilon, self.min_radius)

        # Each interaction is counted twice
        lj_energies = lj_energies.sum(dim=-1) * self.energy_factor * 2.0

        if self.oscillator:
            x = remove_mean(x, self.n_particles, self.spatial_dim)
            osc_energies = 0.5 * x.pow(2).sum(dim=-1)
            lj_energies = lj_energies + osc_energies * self.oscillator_scale

        return lj_energies

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        assert self.approx_sample is not None
        with temp_seed(seed or self.seed):
            perm_idx = torch.randperm(self.approx_sample.shape[0])[:batch_size]
        return self.approx_sample[perm_idx].to(self.device)

    def visualize(
        self, samples: torch.Tensor, weights: torch.Tensor | None = None, **kwargs
    ) -> dict:
        if weights is not None:
            warnings.warn(
                "Can't visualize weighted samples for Lennard-Jones energy. Ignoring them..."
            )
            return {}
        out_dict = {}
        out_dict.update(viz_interatomic_dist_hist(self, samples))
        out_dict.update(viz_energy_hist(self, samples))
        return out_dict


class LJ13(LennardJones):
    def __init__(self, device: str | torch.device, seed: int = 0):
        super().__init__(
            spatial_dim=3,
            n_particles=13,
            device=device,
            data_path=DATA_PATH / "LJ13.npy",
            seed=seed,
        )


class LJ55(LennardJones):
    def __init__(self, device: str | torch.device, seed: int = 0):
        super().__init__(
            spatial_dim=3,
            n_particles=55,
            device=device,
            data_path=DATA_PATH / "LJ55.npy",
            seed=seed,
        )
