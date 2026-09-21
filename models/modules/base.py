from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class ParamGroups:
    forward_params: list[nn.Parameter]
    backward_params: list[nn.Parameter]
    flow_params: list[nn.Parameter]
    logZ_params: list[nn.Parameter]
    lgv_params: list[nn.Parameter]


class BaseModule(nn.Module, ABC):
    def __init__(
        self,
        ndim: int,
        conditional_flow_model: bool,
        lp: bool = False,
        lp_scaling_per_dimension: bool = True,
        clipping: bool = False,
        out_clip: float = 1e4,
        lgv_clip: float = 1e2,
        learn_pb: bool = False,
        pb_scale_range: float = 1.0,
        learn_variance: bool = True,
        log_var_range: float = 4.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.ndim = ndim

        self.conditional_flow_model = conditional_flow_model
        self.flow_model: nn.Module | nn.Parameter | None = None

        self.lp = lp
        self.lp_scaling_per_dimension = lp_scaling_per_dimension

        self.clipping = clipping
        self.out_clip = out_clip
        self.lgv_clip = lgv_clip

        self.learn_pb = learn_pb
        self.pb_scale_range = pb_scale_range

        self.learn_variance = learn_variance
        self.log_var_range = log_var_range

        self.out_dim = 2 * ndim if learn_variance else ndim
        self.lgv_out_dim = ndim if lp_scaling_per_dimension else 1

        self.use_checkpoint = use_checkpoint

        self.flow_model: nn.Module | None = None
        self.log_Z: nn.Parameter = nn.Parameter(torch.tensor(0.0))

    @abstractmethod
    def get_param_groups(self) -> ParamGroups:
        raise NotImplementedError

    def set_log_Z(self, value: float) -> None:
        # Set the parameter value to a given value
        self.log_Z.data.copy_(torch.tensor(value))

    @abstractmethod
    def predict_forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        grad_logr_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # mean, logvar, flow
        # Basic structure:
        # 1. Predict `out` with a prediction model (to be implemented by user)
        # 2. Get `mean`, `logvar` from `out` using `get_gaussian_params`
        # 3. Get `flow` with `predict_flow`
        # 4. Return `mean`, `logvar`, `flow`
        raise NotImplementedError

    def forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        grad_logr_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # mean, logvar, flow
        if self.use_checkpoint and self.training:
            return checkpoint(  # type: ignore
                self.predict_forward,
                s,
                t,
                grad_logr_fn,
                use_reentrant=False,
            )
        else:
            return self.predict_forward(s, t, grad_logr_fn)

    def predict_flow(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def predict_backward(
        self, s_next: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:  # bwd_mean_correction, bwd_var_correction
        if self.learn_pb:
            return self.predict_backward_correction(s_next, t_next, **kwargs)
        else:
            return torch.ones_like(s_next), torch.ones_like(s_next)

    def backward(
        self, s_next: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:  # bwd_mean_correction, bwd_var_correction
        if self.use_checkpoint and self.training:
            return checkpoint(  # type: ignore
                self.predict_backward,
                s_next,
                t_next,
                **kwargs,
                use_reentrant=False,
            )
        else:
            return self.predict_backward(s_next, t_next, **kwargs)

    def predict_backward_correction(
        self, s_next: torch.Tensor, t_next: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_gaussian_params(
        self,
        out: torch.Tensor,
        s: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        grad_logr_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs,  # get_lp_scaling kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out.shape[-1] == self.out_dim

        out = torch.nan_to_num(out)
        if self.clipping:
            out = torch.clip(out, -self.out_clip, self.out_clip)

        if not self.learn_variance:
            mean = out
            logvar = torch.zeros_like(mean)
        else:
            mean, logvar = torch.chunk(out, 2, dim=-1)
            logvar = torch.tanh(logvar) * self.log_var_range

        if self.lp:
            assert s is not None and t is not None and grad_logr_fn is not None
            mean = mean + self.get_lp(s=s, t=t, grad_logr_fn=grad_logr_fn, **kwargs)

        return mean, logvar

    def get_lp(
        self,
        s: torch.Tensor,  # (bsz, ndim)
        t: torch.Tensor,  # (bsz,)
        grad_logr_fn: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        assert self.lp
        grad_logr = torch.nan_to_num(grad_logr_fn(s))
        if self.clipping:
            grad_logr = torch.clip(grad_logr, -self.lgv_clip, self.lgv_clip)
        return self.get_lp_scaling(t=t, **kwargs) * grad_logr

    def get_lp_scaling(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
