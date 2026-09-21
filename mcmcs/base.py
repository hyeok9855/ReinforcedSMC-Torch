import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from energies import BaseEnergy


class BaseMCMC(abc.ABC):
    def __init__(self, energy: "BaseEnergy"):
        self.energy = energy

    @abc.abstractmethod
    def sample(self, initial_position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
