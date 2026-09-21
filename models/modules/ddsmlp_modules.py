from typing import Callable

import torch
from torch import nn

from models.modules.mlp_modules import MLPModule, ParamGroups


class DDSMLPModule(MLPModule):
    def initialize(self):
        self.timestep_phase = nn.Parameter(torch.zeros(self.harmonics_dim)[None])
        self.register_buffer(
            "pe", torch.linspace(start=0.1, end=100, steps=self.harmonics_dim)[None]
        )

        self.time_coder_state = nn.Sequential(
            nn.Linear(2 * self.harmonics_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.t_emb_dim),
        )

        self.time_coder_grad = None
        if self.lp:
            self.time_coder_grad = nn.Sequential(
                nn.Linear(2 * self.harmonics_dim, self.hidden_dim),
                *[
                    nn.Sequential(
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    for _ in range(self.lgv_layers - 1)
                ],
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.lgv_out_dim),
            )
            if self.zero_init:
                self.time_coder_grad[-1].weight.data.fill_(1e-8)
                self.time_coder_grad[-1].bias.data.fill_(0.1)

        self.state_time_net = nn.Sequential(
            nn.Linear(self.ndim + self.t_emb_dim, self.hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU())
                for _ in range(self.joint_layers - 1)
            ],
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        if self.zero_init:
            self.state_time_net[-1].weight.data.fill_(1e-8)
            self.state_time_net[-1].bias.data.fill_(0.0)

        self.bwd_t_model = self.bwd_s_model = self.bwd_joint_model = None
        if self.learn_pb:  # TODO: implement backward correction
            raise NotImplementedError("Backward correction is not implemented for DDSMLPModule!")

        self.flow_timestep_phase = None
        self.flow_time_coder_state = None
        self.flow_state_time_net = None
        if self.conditional_flow_model:
            if not self.share_embeddings:
                self.flow_timestep_phase = nn.Parameter(torch.zeros(self.flow_harmonics_dim)[None])
                self.register_buffer(
                    "flow_pe",
                    torch.linspace(start=0.1, end=100, steps=self.flow_harmonics_dim)[None],
                )
                self.flow_time_coder_state = nn.Sequential(
                    nn.Linear(2 * self.flow_harmonics_dim, self.flow_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.flow_hidden_dim, self.flow_t_emb_dim),
                )

            self.flow_state_time_net = nn.Sequential(
                nn.Linear(self.ndim + self.flow_t_emb_dim, self.flow_hidden_dim),
                nn.GELU(),
                *[
                    nn.Sequential(nn.Linear(self.flow_hidden_dim, self.flow_hidden_dim), nn.GELU())
                    for _ in range(self.flow_layers - 1)
                ],
                nn.Linear(self.flow_hidden_dim, 1),
            )
            if self.zero_init:
                self.flow_state_time_net[-1].weight.data.fill_(1e-8)
                self.flow_state_time_net[-1].bias.data.fill_(0.0)

    def get_param_groups(self) -> ParamGroups:
        forward_params = []
        forward_params += [self.timestep_phase]
        forward_params += list(self.time_coder_state.parameters())
        forward_params += list(self.state_time_net.parameters())

        backward_params = []

        flow_params = []
        if self.flow_state_time_net is not None:
            if not self.share_embeddings:
                flow_params += [self.flow_timestep_phase]
                flow_params += list(self.flow_time_coder_state.parameters())
            flow_params += list(self.flow_state_time_net.parameters())

        logZ_params = [self.log_Z]

        lgv_params = []
        if self.time_coder_grad is not None:
            lgv_params += list(self.time_coder_grad.parameters())

        return ParamGroups(
            forward_params=forward_params,
            backward_params=backward_params,
            flow_params=flow_params,
            logZ_params=logZ_params,
            lgv_params=lgv_params,
        )

    def predict_forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        grad_logr_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sin_embed_cond = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).sin()  # type: ignore
        cos_embed_cond = ((t.unsqueeze(1) * self.pe) + self.timestep_phase).cos()  # type: ignore
        time_array_emb = torch.cat([sin_embed_cond, cos_embed_cond], dim=-1)

        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = torch.cat([s, t_net1], dim=-1)
        out_state = self.state_time_net(extended_input)

        mean, logvar = self.get_gaussian_params(
            out_state, s, t, grad_logr_fn, time_array_emb=time_array_emb
        )
        flow = self.predict_flow(s=s, t=t, extended_input=extended_input)
        return mean, logvar, flow

    def predict_flow(
        self, s: torch.Tensor, t: torch.Tensor, extended_input: torch.Tensor
    ) -> torch.Tensor:
        if self.conditional_flow_model:
            assert self.flow_state_time_net is not None
            if not self.share_embeddings:
                sin_embed_cond = (t.unsqueeze(1) * self.flow_pe + self.flow_timestep_phase).sin()  # type: ignore
                cos_embed_cond = (t.unsqueeze(1) * self.flow_pe + self.flow_timestep_phase).cos()  # type: ignore
                flow_time_array_emb = torch.cat([sin_embed_cond, cos_embed_cond], dim=-1)
                flow_t_net1 = self.flow_time_coder_state(flow_time_array_emb)
                if flow_t_net1.shape[0] == 1:
                    flow_t_net1 = flow_t_net1.repeat(s.shape[0], 1)
                flow_extended_input = torch.cat([s, flow_t_net1], dim=-1)
            else:
                flow_extended_input = extended_input
            flow = self.flow_state_time_net(flow_extended_input).squeeze(-1)
        else:
            flow = torch.zeros(s.shape[0], device=s.device)
        return flow

    def get_lp_scaling(
        self, t: torch.Tensor, time_array_emb: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        assert self.time_coder_grad is not None
        return self.time_coder_grad(time_array_emb)
