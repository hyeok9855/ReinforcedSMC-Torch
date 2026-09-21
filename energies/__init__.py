import argparse

import torch

from .aldp import ALDP
from .base import BaseEnergy
from .funnel import Funnel
from .gmm40 import GMM40
from .intermediate_energy import IntermediateEnergy
from .lennard_jones import LJ13, LJ55, LennardJones
from .lgcp import LGCP
from .manywell import ManyWell
from .nice import NICE, NICEMNIST, NICEFashionMNIST
from .student_t_mixture import StudentTMixture
from .twenty_five_gmm import TwentyFiveGaussianMixture

__all__ = [
    "BaseEnergy",
    "ALDP",
    "Funnel",
    "GMM40",
    "LJ13",
    "LJ55",
    "LennardJones",
    "LGCP",
    "ManyWell",
    "NICE",
    "NICEFashionMNIST",
    "NICEMNIST",
    "StudentTMixture",
    "TwentyFiveGaussianMixture",
    "IntermediateEnergy",
    "get_energy",
]


def get_energy(
    args: argparse.Namespace,
    device: torch.device,
    seed: int = 0,
    n_threads: int = 16,  # only for ALDP
) -> BaseEnergy:
    energy_name: str = args.energy_name
    ndim: int = args.ndim

    if energy_name == "25gmm":
        if ndim != 2:
            raise ValueError("25GMM is only supported for 2D")
        energy = TwentyFiveGaussianMixture(device=device, seed=seed)
    elif energy_name == "gmm40":
        energy = GMM40(device=device, ndim=ndim, seed=seed)
    elif energy_name == "student_t_mixture":
        energy = StudentTMixture(device=device, ndim=ndim, seed=seed)
    elif energy_name == "funnel":
        energy = Funnel(device=device, ndim=ndim, seed=seed)
    elif energy_name == "manywell":
        energy = ManyWell(device=device, ndim=ndim, seed=seed)
    elif energy_name == "nice_mnist":
        energy = NICEMNIST(device=device, seed=seed)
    elif energy_name == "nice_fashion_mnist":
        energy = NICEFashionMNIST(device=device, seed=seed)
    elif energy_name == "lgcp":
        energy = LGCP(device=device, seed=seed)
    elif energy_name == "lj13":
        energy = LJ13(device=device, seed=seed)
    elif energy_name == "lj55":
        energy = LJ55(device=device, seed=seed)
    elif energy_name == "aldp":
        energy = ALDP(device=device, seed=seed, n_threads=n_threads)
    else:
        raise ValueError(f"Unknown energy: {energy_name}")
    return energy
