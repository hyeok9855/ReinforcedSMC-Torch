import abc
import warnings

import torch


class BaseEnergy(abc.ABC):
    is_particle_system = False
    # Flipped off for energies whose `energy` cannot be differentiated by functorch (e.g. ALDP,
    # whose boltzgen/OpenMM bridge is a legacy autograd.Function).
    _func_grad_ok: bool = True

    def __init__(
        self,
        device: str | torch.device,
        ndim: int,
        seed: int = 0,
        plot_bound: float = 1.0,
    ) -> None:
        self.device = device
        self.ndim = ndim
        self.seed = seed
        self.plot_bound = plot_bound
        self.gt_xs: torch.Tensor | None = None
        self.gt_xs_log_rewards: torch.Tensor | None = None

    @abc.abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        log_r = -self.energy(x)
        return log_r

    def grad_log_reward(self, x: torch.Tensor) -> torch.Tensor:
        if self._func_grad_ok:
            try:
                lgv = torch.func.grad(lambda state: self.log_reward(state).sum())(x.detach())
                return lgv.detach()
            except RuntimeError as e:
                if "setup_context" not in str(e):
                    raise
                self._func_grad_ok = False
                warnings.warn(
                    f"{type(self).__name__}.energy uses an autograd.Function that functorch "
                    "cannot transform; falling back to autograd for grad_log_reward."
                )

        with torch.enable_grad():  # `get_lp` runs under no_grad during eval
            copy_x = x.detach().clone().requires_grad_(True)
            (lgv,) = torch.autograd.grad(self.log_reward(copy_x).sum(), copy_x)
        return lgv.detach()

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        raise NotImplementedError

    def gt_logz(self) -> float:
        raise NotImplementedError

    def cached_sample(
        self, batch_size: int, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gt_xs is None or batch_size != self.gt_xs.size(0):
            self.gt_xs = self.sample(batch_size, seed)
            self.gt_xs_log_rewards = self.log_reward(self.gt_xs)
        assert self.gt_xs_log_rewards is not None
        return self.gt_xs, self.gt_xs_log_rewards

    def visualize(self, samples: torch.Tensor, **kwargs) -> dict:
        raise NotImplementedError
