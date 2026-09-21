import torch
from torch import nn

from models.modules.mlp_modules import MLPModule


class PISMLPModule(MLPModule):
    def initialize(self):
        assert self.s_emb_dim == self.t_emb_dim, (
            "Dimensionality of state embedding and time embedding should be the same!"
        )

        self.t_model = TimeEncodingPIS(self.harmonics_dim, self.t_emb_dim, self.hidden_dim)
        self.s_model = StateEncodingPIS(self.ndim, self.s_emb_dim)
        self.joint_model = JointPolicyPIS(
            self.s_emb_dim, self.hidden_dim, self.out_dim, self.joint_layers, self.zero_init
        )

        self.bwd_t_model = self.bwd_s_model = self.bwd_joint_model = None
        if self.learn_pb:
            self.bwd_t_model = TimeEncodingPIS(self.harmonics_dim, self.t_emb_dim, self.hidden_dim)
            self.bwd_s_model = StateEncodingPIS(self.ndim, self.s_emb_dim)
            self.bwd_joint_model = JointPolicyPIS(
                self.s_emb_dim, self.hidden_dim, 2 * self.ndim, self.joint_layers, self.zero_init
            )

        if self.conditional_flow_model:
            assert self.flow_t_emb_dim == self.flow_s_emb_dim
            self.t_model_flow = (
                TimeEncodingPIS(self.flow_harmonics_dim, self.flow_t_emb_dim, self.flow_hidden_dim)
                if not self.share_embeddings
                else nn.Identity()
            )
            self.s_model_flow = (
                StateEncodingPIS(self.ndim, self.flow_s_emb_dim)
                if not self.share_embeddings
                else nn.Identity()
            )
            self.flow_model = FlowModelPIS(
                self.flow_s_emb_dim if not self.share_embeddings else self.s_emb_dim,
                self.flow_hidden_dim,
                1,
                self.flow_layers,
            )

        self.lp_scaling_model = None
        if self.lp:
            self.lp_scaling_model = LangevinScalingModelPIS(
                self.harmonics_dim,
                self.hidden_dim,
                self.lgv_out_dim,
                self.lgv_layers,
                self.zero_init,
            )

    def get_lp_scaling(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        assert self.lp_scaling_model is not None
        return self.lp_scaling_model(t)


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
        )
        self.register_buffer("pe", torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            t: torch.Tensor
        """
        t_sin = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        t_cos = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, ndim: int, s_emb_dim: int) -> None:
        super().__init__()

        self.s_model = nn.Linear(ndim, s_emb_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.s_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Linear(s_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(1e-8)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(s_emb + t_emb)


class FlowModelPIS(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Linear(s_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.model(s_emb + t_emb)


class LangevinScalingModelPIS(nn.Module):
    def __init__(
        self,
        harmonics_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])
        self.lgv_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.register_buffer("pe", torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None])

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(1e-8)
            self.lgv_model[-1].bias.data.fill_(0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_sin = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        t_cos = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)
