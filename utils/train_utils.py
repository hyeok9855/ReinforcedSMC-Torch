from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from models import GFN


class CompositeOptimizer:
    """Wraps several optimizers so the trainer can treat them as one.

    Network parameters are optimized with Adam; ``logZ`` gets its own optimizer
    so its step size can be tuned separately.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        logZ_optimizer: torch.optim.Optimizer,
        clip_params: list[torch.nn.Parameter],
        logZ_clip_params: list[torch.nn.Parameter] | None = None,
    ):
        self.optimizer = optimizer
        self.logZ_optimizer = logZ_optimizer
        self._clip_params = list(clip_params)
        self._logZ_clip_params = list(logZ_clip_params or [])

    @property
    def param_groups(self):
        groups = [g for g in self.optimizer.param_groups]
        groups.extend(self.logZ_optimizer.param_groups)
        return groups

    def clip_grad_norm_(self, max_norm: float, logZ_max_norm: float = 0.0):
        if max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self._clip_params, max_norm)
        if logZ_max_norm > 0.0 and self._logZ_clip_params:
            torch.nn.utils.clip_grad_norm_(self._logZ_clip_params, logZ_max_norm)

    def step(self):
        self.optimizer.step()
        self.logZ_optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self.logZ_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "logZ_optimizer": self.logZ_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.logZ_optimizer.load_state_dict(state_dict["logZ_optimizer"])


class CompositeScheduler:
    """Wraps several LR schedulers so the trainer can step them as one."""

    def __init__(self, schedulers: list[torch.optim.lr_scheduler.LRScheduler]):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return {"schedulers": [s.state_dict() for s in self.schedulers]}

    def load_state_dict(self, state_dict):
        for scheduler, sd in zip(self.schedulers, state_dict["schedulers"]):
            scheduler.load_state_dict(sd)


def get_gfn_optimizer(
    gfn_model: "GFN",
    lr_fwd: float,
    lr_bwd: float,
    lr_logZ: float,
    lr_flow: float | None = None,
    lr_beta: float | None = None,
    lr_lgv: float | None = None,
    momentum_logZ: float = 0.0,
    logZ_optimizer_type: str = "adam",
    use_weight_decay=False,
    weight_decay=1e-7,
    use_scheduler=False,
    milestones: list[int] = [100000],
    gamma: float = 1.0,
):

    module_param_groups = gfn_model.pred_module.get_param_groups()

    # Network parameters -> Adam.
    network_param_groups = [
        {"params": module_param_groups.forward_params, "lr": lr_fwd},
        {"params": module_param_groups.backward_params, "lr": lr_bwd},
    ]
    if len(module_param_groups.lgv_params) > 0:
        assert lr_lgv is not None
        network_param_groups.append({"params": module_param_groups.lgv_params, "lr": lr_lgv})

    if len(module_param_groups.flow_params) > 0:
        assert lr_flow is not None
        network_param_groups.append({"params": module_param_groups.flow_params, "lr": lr_flow})

    if gfn_model.beta_model is not None:
        assert lr_beta is not None
        network_param_groups.append({"params": [gfn_model.beta_model], "lr": lr_beta})

    optimizer = torch.optim.Adam(
        network_param_groups,
        lr=0.0,
        weight_decay=weight_decay if use_weight_decay else 0.0,
    )

    if logZ_optimizer_type == "adam":
        scalar_param_groups = [{"params": module_param_groups.logZ_params, "lr": lr_logZ}]
        logZ_optimizer = torch.optim.Adam(scalar_param_groups, lr=0.0)
    elif logZ_optimizer_type == "sgd":
        scalar_param_groups = [
            {"params": module_param_groups.logZ_params, "lr": lr_logZ, "momentum": momentum_logZ},
        ]
        logZ_optimizer = torch.optim.SGD(scalar_param_groups, lr=0.0)
    else:
        raise ValueError(
            f"logZ_optimizer_type must be 'adam' or 'sgd' (got {logZ_optimizer_type!r})"
        )

    clip_params = [p for group in network_param_groups for p in group["params"]]
    gfn_optimizer = CompositeOptimizer(
        optimizer=optimizer,
        logZ_optimizer=logZ_optimizer,
        clip_params=clip_params,
        logZ_clip_params=module_param_groups.logZ_params,
    )

    gfn_scheduler = None
    if use_scheduler:
        gfn_scheduler = CompositeScheduler(
            [
                torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
                for opt in (optimizer, logZ_optimizer)
            ]
        )
    return gfn_optimizer, gfn_scheduler


#######################################
# Importance weight related functions #
#######################################


def solve_mixing_ratio(normalized_weights: torch.Tensor, target_ess: float) -> float:
    """
    Find the mixing ratio to achieve the target effective sample size (ESS)

    normalized_weights_mix = (1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size

    ESS_mix = 1 / (normalized_weights_mix^2).sum()
            = 1 / (((1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size)^2).sum()

    Solve the following equation for mixing_ratio:
    1 / ESS_mix = (((1 - mixing_ratio) * normalized_weights + mixing_ratio / batch_size)^2).sum()
                = 1 / target_ess

    This is equivalent to the following quadratic equation:
        A * (mixing_ratio^2) - 2 * B * mixing_ratio + C = 0
    where
        A = normalized_weights^2.sum() - 2 * normalized_weights.sum() / N + 1 / N
        B = normalized_weights^2.sum() - normalized_weights.sum() / N
        C = normalized_weights^2.sum() - 1 / target_ess
    """
    N = len(normalized_weights)
    nw_sum = 1.0
    nw_squared_sum = (normalized_weights**2).sum().item()

    ess_before = 1 / nw_squared_sum
    if ess_before >= target_ess:
        return 0.0

    A = nw_squared_sum - 2 * nw_sum / N + 1 / N
    B = nw_squared_sum - nw_sum / N
    C = nw_squared_sum - 1 / target_ess

    min_lhs = C - B**2 / A
    if min_lhs >= 0:
        raise ValueError(f"Cannot achieve target ESS: {target_ess}")
        return 1.0

    mixing_ratio_1 = (B + (B**2 - A * C) ** 0.5) / A
    mixing_ratio_2 = (B - (B**2 - A * C) ** 0.5) / A

    valid_1 = mixing_ratio_1 >= 0.0 and mixing_ratio_1 <= 1.0
    valid_2 = mixing_ratio_2 >= 0.0 and mixing_ratio_2 <= 1.0

    if valid_1 and valid_2:
        raise ValueError(f"Multiple solutions: {mixing_ratio_1}, {mixing_ratio_2}")
    elif valid_1:
        return mixing_ratio_1
    elif valid_2:
        return mixing_ratio_2
    else:
        raise ValueError(f"No valid solution: {mixing_ratio_1}, {mixing_ratio_2}")


def ess(
    log_weights: torch.Tensor | None = None,  # (bs, T)
    normalized_weights: torch.Tensor | None = None,  # (bs, T)
) -> torch.Tensor:
    if normalized_weights is None:
        assert log_weights is not None
        normalized_weights = log_weights.softmax(dim=0)  # (bs, T)
    return 1 / (normalized_weights**2).sum(dim=0)  # (T,)


def binary_search_smoothing(
    log_weights: torch.Tensor,  # (bs, T)
    target_ess: float,
    tol=1e-3,
    max_steps=1000,
    check_every=20,
) -> tuple[torch.Tensor, torch.Tensor]:  # (bs, T), (1, T)
    bs = log_weights.shape[0]

    search_min, search_max = get_min_max(log_weights)
    search_min = torch.tensor(search_min, device=log_weights.device).repeat(1, log_weights.shape[1])
    search_max = torch.tensor(search_max, device=log_weights.device).repeat(1, log_weights.shape[1])
    original_order = ess(log_weights / search_min) < ess(log_weights / search_max)

    dones = ess(log_weights=log_weights) / bs >= target_ess  # (T,)
    mid = torch.where(
        dones.unsqueeze(0), torch.ones_like(search_min), (search_min + search_max) / 2
    )  # (1, T)
    log_weights_smoothed = log_weights.clone()  # (bs, T)

    steps = 0
    while not bool(dones.all()):
        for _ in range(min(check_every, max_steps + 1 - steps)):
            steps += 1
            mid = torch.where(dones.unsqueeze(0), mid, (search_min + search_max) / 2)  # (1, T)

            new_log_weights = log_weights / mid  # (bs, T)
            new_ess = ess(log_weights=new_log_weights) / bs  # (T,)
            new_dones = (~dones) & (abs(new_ess - target_ess) / target_ess < tol)  # (T,)
            log_weights_smoothed = torch.where(
                new_dones.unsqueeze(0), new_log_weights, log_weights_smoothed
            )
            dones = dones | new_dones

            search_max = torch.where((new_ess > target_ess) == original_order, mid, search_max)
            search_min = torch.where((new_ess < target_ess) == original_order, mid, search_min)

        if steps > max_steps:
            print(f"Warning: Binary search failed in {max_steps} steps")
            log_weights_smoothed = torch.where(
                dones.unsqueeze(0), log_weights_smoothed, new_log_weights
            )
            break
    return log_weights_smoothed, mid


def get_min_max(log_weights: torch.Tensor) -> tuple[float, float]:
    _min = torch.nan_to_num(log_weights, nan=float("inf"), neginf=float("inf")).min().item()
    _max = torch.nan_to_num(log_weights, nan=float("-inf"), posinf=float("-inf")).max().item()
    return 1.0, (_max - _min) / 2
