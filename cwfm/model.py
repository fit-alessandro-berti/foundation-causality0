from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CWFMConfig:
    max_variables: int = 12
    hidden_dim: int = 96
    heads: int = 4
    axial_blocks: int = 3
    mechanism_experts: int = 4
    top_k_experts: int = 2
    world_particles: int = 8
    dropout: float = 0.05

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SparseMechanismMoE(nn.Module):
    def __init__(self, dim: int, experts: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                )
                for _ in range(experts)
            ]
        )
        self.top_k = min(top_k, experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.router(x).softmax(-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
        sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = (outputs * sparse.unsqueeze(-1)).sum(-2)
        return mixed, probabilities


class AxialBlock(nn.Module):
    def __init__(self, config: CWFMConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.variable_attention = nn.MultiheadAttention(
            d, config.heads, config.dropout, batch_first=True
        )
        self.sample_attention = nn.MultiheadAttention(
            d, config.heads, config.dropout, batch_first=True
        )
        self.relation_projection = nn.Linear(d, d)
        self.variable_norm = nn.LayerNorm(d)
        self.sample_norm = nn.LayerNorm(d)
        self.relation_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.moe = SparseMechanismMoE(
            d, config.mechanism_experts, config.top_k_experts, config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        variable_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rows, variables, dim = x.shape
        variable_input = x.reshape(batch * rows, variables, dim)
        key_padding = (~variable_mask).repeat_interleave(rows, dim=0)
        update, _ = self.variable_attention(
            variable_input,
            variable_input,
            variable_input,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        x = self.variable_norm(x + update.reshape_as(x))

        row_summary = (x * variable_mask[:, None, :, None]).sum(2)
        row_summary = row_summary / variable_mask.sum(1)[:, None, None].clamp_min(1)
        degree = adjacency.sum(-1, keepdim=True)
        relation_message = torch.bmm(adjacency, row_summary) / degree.clamp_min(1.0)
        x = self.relation_norm(
            x + self.relation_projection(relation_message).unsqueeze(2)
        )

        sample_input = x.permute(0, 2, 1, 3).reshape(batch * variables, rows, dim)
        update, _ = self.sample_attention(
            sample_input, sample_input, sample_input, need_weights=False
        )
        x = self.sample_norm(
            x + update.reshape(batch, variables, rows, dim).permute(0, 2, 1, 3)
        )
        mixed, routing = self.moe(x)
        return self.ff_norm(x + mixed), routing


class CWFM(nn.Module):
    """Compact query-conditioned causal-world foundation model.

    It is intentionally a runnable reference implementation, not the full-scale
    384-dimensional configuration proposed in the design document.
    """

    def __init__(self, config: CWFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or CWFMConfig()
        c = self.config
        d = c.hidden_dim
        self.value_encoder = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.role_embedding = nn.Embedding(6, d)
        self.covariance_encoder = nn.Sequential(
            nn.Linear(1, d),
            nn.Tanh(),
            nn.Linear(d, d),
        )
        self.task_embedding = nn.Embedding(3, d)
        self.design_embedding = nn.Embedding(4, d)
        self.blocks = nn.ModuleList([AxialBlock(c) for _ in range(c.axial_blocks)])
        self.world_queries = nn.Parameter(torch.randn(c.world_particles, d) * 0.05)
        self.world_attention = nn.MultiheadAttention(
            d, c.heads, c.dropout, batch_first=True
        )
        self.world_weight = nn.Linear(d, 1)
        self.world_effect = nn.Linear(d, 2)
        self.identification_head = nn.Linear(d, 1)
        self.support_head = nn.Linear(d, 1)
        self.mechanism_head = nn.Linear(d, 3)
        self.flexible_route_head = nn.Linear(d, 1)
        self.structure_variable_head = nn.Linear(d, 1)
        self.structure_none_head = nn.Linear(d, 1)
        self.instability_scale = nn.Parameter(torch.tensor(2.0))
        self.graph_left = nn.Linear(d, d, bias=False)
        self.graph_right = nn.Linear(d, d, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = batch["values"]
        missing = batch["missing"]
        roles = batch["roles"]
        variable_mask = batch["variable_mask"]
        task = batch["task"]
        design = batch["design"]
        adjacency = batch["adjacency"]
        cell_input = torch.stack([values, missing], dim=-1)
        query = self.task_embedding(task) + self.design_embedding(design)
        x = self.value_encoder(cell_input) + self.role_embedding(roles)[:, None]
        x = x + query[:, None, None]
        routing = []
        for block in self.blocks:
            x, route = block(x, variable_mask, adjacency)
            routing.append(route.mean(dim=(1, 2)))
        variable_tokens = x.mean(1)
        covariance = torch.matmul(values.transpose(1, 2), values) / values.shape[1]
        pair_tokens = self.covariance_encoder(covariance.unsqueeze(-1))
        source_roles = self.role_embedding(roles)[:, None, :, :]
        pair_mask = variable_mask[:, None, :, None]
        covariance_tokens = ((pair_tokens + source_roles) * pair_mask).sum(2)
        covariance_tokens = covariance_tokens / variable_mask.sum(1)[:, None, None].clamp_min(1)
        variable_tokens = variable_tokens + covariance_tokens
        variable_tokens = variable_tokens * variable_mask.unsqueeze(-1)
        worlds = self.world_queries.unsqueeze(0).expand(len(values), -1, -1)
        worlds = worlds + query.unsqueeze(1)
        worlds, _ = self.world_attention(
            worlds,
            variable_tokens,
            variable_tokens,
            key_padding_mask=~variable_mask,
            need_weights=False,
        )
        world_weights = self.world_weight(worlds).squeeze(-1).softmax(-1)
        particle_params = self.world_effect(worlds)
        particle_means = particle_params[..., 0]
        particle_scales = F.softplus(particle_params[..., 1]) + 0.03
        # A differentiable typed linear anchor gives the direct decoder a
        # statistically efficient starting point; particles learn nonlinear
        # and misspecification corrections around it.
        anchors = []
        instability = []
        for index in range(len(values)):
            valid = variable_mask[index]
            role = roles[index]
            outcome_index = torch.nonzero((role == 2) & valid, as_tuple=False)[0, 0]
            treatment = torch.nonzero((role == 1) & valid, as_tuple=False)
            exposure = torch.nonzero((role == 3) & valid, as_tuple=False)
            predictors = torch.nonzero(
                ((role == 0) | (role == 4)) & valid, as_tuple=False
            ).flatten()
            if len(exposure):
                focal = exposure[0, 0]
            elif len(treatment):
                focal = treatment[0, 0]
            else:
                focal = None
            if focal is None:
                anchors.append(values.new_zeros(()))
            else:
                columns = torch.cat([predictors, focal.reshape(1)])
                design_matrix = values[index, :, columns]
                response = values[index, :, outcome_index]
                gram = design_matrix.T @ design_matrix
                ridge = torch.eye(len(columns), device=values.device) * 0.1
                coefficient = torch.linalg.solve(
                    gram + ridge, design_matrix.T @ response
                )[-1]
                anchors.append(
                    coefficient * (0.5 if len(exposure) else 1.0)
                )
            ordinary = torch.nonzero((role == 4) & valid, as_tuple=False).flatten()
            if len(ordinary):
                product = values[index, :, ordinary[0]] * values[index, :, outcome_index]
                scores = []
                for variable in range(values.shape[-1]):
                    if not valid[variable] or role[variable] != 0:
                        scores.append(values.new_zeros(()))
                        continue
                    side = values[index, :, variable] > 0
                    if side.sum() < 8 or (~side).sum() < 8:
                        scores.append(values.new_zeros(()))
                    else:
                        scores.append(
                            (product[side].mean() - product[~side].mean()).abs()
                        )
                instability.append(torch.stack(scores))
            else:
                instability.append(values.new_zeros(values.shape[-1]))
        anchor = torch.stack(anchors).clamp(-3.0, 3.0)
        instability_score = torch.stack(instability)
        residual_mean = (world_weights * particle_means).sum(-1)
        effect_mean = anchor + residual_mean
        second_moment = (
            world_weights * (particle_scales.square() + particle_means.square())
        ).sum(-1)
        effect_scale = (second_moment - effect_mean.square()).clamp_min(1e-5).sqrt()
        global_token = (worlds * world_weights.unsqueeze(-1)).sum(1)
        structure_logits = self.structure_variable_head(variable_tokens).squeeze(-1)
        structure_logits = structure_logits + F.softplus(self.instability_scale) * instability_score
        structure_logits = structure_logits.masked_fill(~variable_mask, -1e4)
        none_logit = self.structure_none_head(global_token)
        structure_logits = torch.cat([structure_logits, none_logit], dim=-1)
        graph_logits = torch.matmul(
            self.graph_left(variable_tokens),
            self.graph_right(variable_tokens).transpose(1, 2),
        ) / self.config.hidden_dim**0.5
        return {
            "effect_mean": effect_mean,
            "effect_scale": effect_scale,
            "particle_means": particle_means,
            "particle_scales": particle_scales,
            "world_weights": world_weights,
            "identification_logit": self.identification_head(global_token).squeeze(-1),
            "support_logit": self.support_head(global_token).squeeze(-1),
            "mechanism_logits": self.mechanism_head(global_token),
            "flexible_route_logit": self.flexible_route_head(global_token).squeeze(-1),
            "structure_logits": structure_logits,
            "graph_logits": graph_logits,
            "expert_routing": torch.stack(routing, dim=1),
        }
