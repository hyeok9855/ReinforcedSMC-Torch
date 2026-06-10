from typing import Callable, Literal

import torch

from buffers.datasets import CustomDataset
from utils.train_utils import binary_search_smoothing


class TerminalStateBuffer:
    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        prioritization: Literal["none", "reward", "loss", "iw", "normalized_iw"],
        sampling_func: Callable[[torch.Tensor, int, bool], torch.Tensor],
        logr_lb: float | None = None,
        target_ess: float = 0.05,  # for normalized_iw and iw
    ) -> None:
        assert prioritization in ["none", "reward", "loss", "iw", "normalized_iw"]
        if prioritization == "normalized_iw":
            assert sampling_func is not None

        self.buffer_size = buffer_size
        self.device = device
        self.prioritization = prioritization
        self.sampling_func = sampling_func
        self.logr_lb = logr_lb
        self.target_ess = target_ess

        self.x_dataset = CustomDataset(buffer_size, device)
        self.logr_dataset = CustomDataset(buffer_size, device)
        self.priority_dataset = CustomDataset(buffer_size, device)

    def __len__(self) -> int:
        return len(self.x_dataset)

    def get_logr_mask(self, log_rs: torch.Tensor) -> torch.Tensor:
        if self.logr_lb is None:
            mask = torch.ones_like(log_rs, dtype=torch.bool)
        else:
            mask = log_rs > self.logr_lb
        return mask

    def add(
        self,
        xs: torch.Tensor,
        log_rs: torch.Tensor,
        log_iws: torch.Tensor | None = None,
        losses: torch.Tensor | None = None,
    ) -> None:
        mask = self.get_logr_mask(log_rs)

        self.x_dataset.add(xs[mask])
        self.logr_dataset.add(log_rs[mask])
        match self.prioritization:
            case "reward":
                self.priority_dataset.add(log_rs[mask])
            case "loss":
                assert losses is not None
                self.priority_dataset.add(losses[mask].log())
            case "normalized_iw":
                assert log_iws is not None
                log_iws_smoothed, _ = binary_search_smoothing(log_iws.unsqueeze(1), self.target_ess)
                log_iws = log_iws_smoothed.squeeze(1).log_softmax(dim=0)
                self.priority_dataset.add(log_iws[mask])
            case "iw":
                assert log_iws is not None
                self.priority_dataset.add(log_iws[mask])
            case _:
                self.priority_dataset.add(torch.ones_like(log_rs[mask]))

    def update(
        self,
        indices: torch.Tensor,
        xs: torch.Tensor | None = None,
        log_rs: torch.Tensor | None = None,
        log_iws: torch.Tensor | None = None,
        losses: torch.Tensor | None = None,
    ) -> None:
        if xs is not None:
            self.x_dataset.update(indices, xs)
        if log_rs is not None:
            self.logr_dataset.update(indices, log_rs)

        match self.prioritization:
            case "reward":
                assert log_rs is not None
                self.priority_dataset.update(indices, log_rs)
            case "loss":
                assert losses is not None
                self.priority_dataset.update(indices, losses.log())
            case "normalized_iw":
                assert log_iws is not None
                log_iws_smoothed, _ = binary_search_smoothing(log_iws.unsqueeze(1), self.target_ess)
                log_iws = log_iws_smoothed.squeeze(1)
                self.priority_dataset.update(indices, log_iws)
            case "iw":
                assert log_iws is not None
                self.priority_dataset.update(indices, log_iws)
            case _:
                self.priority_dataset.update(indices, torch.ones_like(log_rs))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self) > 0, "Buffer is empty"

        weights = self.priority_dataset.data
        if self.prioritization == "iw":
            # apply ESS-based smoothing
            log_iws_smoothed, _ = binary_search_smoothing(
                log_weights=weights.unsqueeze(1),
                target_ess=self.target_ess,
            )
            weights = log_iws_smoothed.squeeze(1)
        weights = weights.softmax(dim=0)

        replacement = True if self.prioritization in ["iw", "normalized_iw"] else False
        indices = self.sampling_func(weights, batch_size, replacement)
        xs, log_rs = self.x_dataset[indices], self.logr_dataset[indices]

        return xs, log_rs, indices
