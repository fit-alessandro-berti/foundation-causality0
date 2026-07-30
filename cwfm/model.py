from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import (
    DESIGN_NONIDENTIFIED,
    DESIGN_POOR_SUPPORT,
    ROLE_COVARIATE,
    ROLE_PADDING,
)


@dataclass
class CWFMConfig:
    max_variables: int = 12
    hidden_dim: int = 96
    heads: int = 4
    axial_blocks: int = 3
    mechanism_experts: int = 4
    top_k_experts: int = 2
    world_particles: int = 6
    estimator_experts: int = 6
    dropout: float = 0.08

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
        logits = self.router(x)
        probabilities = logits.softmax(-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
        sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (outputs * sparse.unsqueeze(-1)).sum(-2), probabilities


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
        self.graph_projection = nn.Linear(d, d)
        self.cluster_projection = nn.Linear(d, d)
        self.role_relation_gate = nn.Linear(d, d)
        self.variable_norm = nn.LayerNorm(d)
        self.sample_norm = nn.LayerNorm(d)
        self.relation_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.moe = SparseMechanismMoE(
            d, config.mechanism_experts, config.top_k_experts, config.dropout
        )

    @staticmethod
    def _cluster_message(
        unit: torch.Tensor,
        cluster_ids: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(unit)
        for batch_index in range(len(unit)):
            valid_ids = torch.unique(cluster_ids[batch_index, row_mask[batch_index]])
            for cluster in valid_ids:
                members = (cluster_ids[batch_index] == cluster) & row_mask[batch_index]
                result[batch_index, members] = unit[batch_index, members].mean(0)
        return result

    def forward(
        self,
        x: torch.Tensor,
        variable_mask: torch.Tensor,
        row_mask: torch.Tensor,
        adjacency: torch.Tensor,
        cluster_ids: torch.Tensor,
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

        unit = (x * variable_mask[:, None, :, None]).sum(2)
        unit = unit / variable_mask.sum(1)[:, None, None].clamp_min(1)
        degree = adjacency.sum(-1, keepdim=True)
        graph_message = torch.bmm(adjacency, unit) / degree.clamp_min(1.0)
        cluster_message = self._cluster_message(unit, cluster_ids, row_mask)
        typed_message = (
            self.graph_projection(graph_message)
            + self.cluster_projection(cluster_message)
        ).unsqueeze(2)
        x = self.relation_norm(x + typed_message * self.role_relation_gate(x).sigmoid())

        sample_input = x.permute(0, 2, 1, 3).reshape(batch * variables, rows, dim)
        sample_padding = (~row_mask).repeat_interleave(variables, dim=0)
        update, _ = self.sample_attention(
            sample_input,
            sample_input,
            sample_input,
            key_padding_mask=sample_padding,
            need_weights=False,
        )
        x = self.sample_norm(
            x + update.reshape(batch, variables, rows, dim).permute(0, 2, 1, 3)
        )
        mixed, routing = self.moe(x)
        x = self.ff_norm(x + mixed)
        return x * row_mask[:, :, None, None], routing


class CWFM(nn.Module):
    """Query-correct CWFM with a risk-routed anchor and do-no-harm residual."""

    def __init__(self, config: CWFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or CWFMConfig()
        c = self.config
        d = c.hidden_dim
        self.value_encoder = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.role_embedding = nn.Embedding(6, d)
        self.task_embedding = nn.Embedding(3, d)
        self.query_encoder = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.summary_encoder = nn.Sequential(
            nn.Linear(9, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d)
        )
        self.summary_attention = nn.MultiheadAttention(
            d, c.heads, c.dropout, batch_first=True
        )
        self.blocks = nn.ModuleList([AxialBlock(c) for _ in range(c.axial_blocks)])
        self.task_adapters = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
                for _ in range(3)
            ]
        )

        self.estimator_feature_encoder = nn.Sequential(
            nn.Linear(c.estimator_experts * 2 + 3, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.estimator_router = nn.Linear(d * 2, c.estimator_experts)
        self.residual_heads = nn.ModuleList([nn.Linear(d * 2, 1) for _ in range(3)])
        self.residual_gate = nn.Linear(d * 2 + 4, 1)

        self.world_queries = nn.Parameter(torch.randn(c.world_particles, d) * 0.05)
        self.world_attention = nn.MultiheadAttention(
            d, c.heads, c.dropout, batch_first=True
        )
        self.world_weight = nn.Linear(d, 1)
        self.world_effect = nn.Linear(d, 2)
        self.world_mechanism = nn.Linear(d, 3)
        self.world_graph_shift = nn.Linear(d, 2)

        self.support_head = nn.Linear(d * 2, 1)
        self.prior_mismatch_head = nn.Linear(d * 2, 1)
        self.mechanism_head = nn.Linear(d * 2, 3)
        self.graph_left = nn.Linear(d, d, bias=False)
        self.graph_right = nn.Linear(d, d, bias=False)

        self.regime_candidate = nn.Sequential(
            nn.Linear(6, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.split_head = nn.Linear(d * 2 + 2, 1)
        self.change_type_head = nn.Linear(d * 2, 3)
        self.regime_delta_head = nn.Linear(d * 2, 2)

        # The initialized model is exactly the compiled estimator.
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, -5.0)
        for head in self.residual_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.world_effect.weight)
        nn.init.zeros_(self.world_effect.bias)

    @staticmethod
    def _summary_tokens(
        values: torch.Tensor,
        missing: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = row_mask[:, :, None].to(values.dtype)
        count = valid.sum(1).clamp_min(1.0)
        mean = (values * valid).sum(1) / count
        centered = (values - mean[:, None]) * valid
        variance = centered.square().sum(1) / count
        std = variance.clamp_min(1e-7).sqrt()
        # Padded rows are excluded by extreme sentinels for min/max; the
        # normalized simulator makes these stable low-cost quantile proxies.
        low = values.masked_fill(~row_mask[:, :, None], torch.inf).amin(1)
        high = values.masked_fill(~row_mask[:, :, None], -torch.inf).amax(1)
        missing_rate = (missing * valid).sum(1) / count
        abs_mean = (values.abs() * valid).sum(1) / count
        positive = ((values > 0).to(values.dtype) * valid).sum(1) / count
        return torch.stack(
            [mean, std, low, high, missing_rate, abs_mean, positive, count.expand_as(mean), variance],
            dim=-1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = batch["values"]
        missing = batch["missing"]
        roles = batch["roles"]
        variable_mask = batch["variable_mask"]
        row_mask = batch["row_mask"]
        task = batch["task"]
        adjacency = batch["adjacency"]
        cluster_ids = batch["cluster_ids"]

        query = self.task_embedding(task) + self.query_encoder(batch["query"])
        cell_input = torch.stack([values, missing], dim=-1)
        x = self.value_encoder(cell_input) + self.role_embedding(roles)[:, None]
        x = x + query[:, None, None]
        routing = []
        for block in self.blocks:
            x, route = block(
                x, variable_mask, row_mask, adjacency, cluster_ids
            )
            routing.append(
                (
                    route
                    * row_mask[:, :, None, None]
                    * variable_mask[:, None, :, None]
                ).sum(dim=(1, 2))
                / (
                    row_mask.sum(1)[:, None] * variable_mask.sum(1)[:, None]
                ).clamp_min(1)
            )

        variable_tokens = x.sum(1) / row_mask.sum(1)[:, None, None].clamp_min(1)
        summaries = self._summary_tokens(values, missing, row_mask)
        summaries = summaries.masked_fill(~variable_mask[:, :, None], 0.0)
        summary_tokens = self.summary_encoder(summaries)
        summary_update, _ = self.summary_attention(
            variable_tokens,
            summary_tokens,
            summary_tokens,
            key_padding_mask=~variable_mask,
            need_weights=False,
        )
        variable_tokens = (variable_tokens + summary_update) * variable_mask.unsqueeze(-1)
        pooled = variable_tokens.sum(1) / variable_mask.sum(1)[:, None].clamp_min(1)
        adapted = torch.stack(
            [adapter(pooled) for adapter in self.task_adapters], dim=1
        )
        task_token = adapted[torch.arange(len(task), device=task.device), task]

        estimates = batch["expert_estimates"]
        standard_errors = batch["expert_standard_errors"].clamp_min(1e-4)
        expert_mask = batch["expert_mask"]
        disagreement = estimates.std(-1, keepdim=True)
        overlap = torch.stack(
            [
                standard_errors.mean(-1),
                standard_errors.max(-1).values,
                expert_mask.to(values.dtype).mean(-1),
            ],
            dim=-1,
        )
        estimator_features = self.estimator_feature_encoder(
            torch.cat([estimates, standard_errors, overlap], dim=-1)
        )
        causal_token = torch.cat([task_token, estimator_features], dim=-1)
        router_logits = self.estimator_router(causal_token).masked_fill(
            ~expert_mask, -1e4
        )
        router_weights = router_logits.softmax(-1)
        no_effect_expert = ~expert_mask.any(-1)
        router_weights = torch.where(
            no_effect_expert[:, None],
            torch.zeros_like(router_weights),
            router_weights,
        )
        compiled = (router_weights * estimates).sum(-1)
        compiled_se = (
            (router_weights * standard_errors).square().sum(-1).clamp_min(1e-6).sqrt()
        )

        task_residuals = torch.stack(
            [head(causal_token).squeeze(-1) for head in self.residual_heads], dim=-1
        )
        residual = task_residuals.gather(1, task[:, None]).squeeze(1)
        support_features = torch.cat(
            [
                disagreement,
                compiled_se[:, None],
                router_weights.max(-1).values[:, None],
                standard_errors.max(-1).values[:, None],
            ],
            dim=-1,
        )
        gate_logit = self.residual_gate(torch.cat([causal_token, support_features], -1)).squeeze(-1)
        gate = gate_logit.sigmoid()
        correction = gate * residual
        effect_mean = compiled.detach() + correction

        worlds = self.world_queries.unsqueeze(0).expand(len(values), -1, -1)
        worlds = worlds + task_token.unsqueeze(1)
        worlds, _ = self.world_attention(
            worlds,
            variable_tokens,
            variable_tokens,
            key_padding_mask=~variable_mask,
            need_weights=False,
        )
        world_weights = self.world_weight(worlds).squeeze(-1).softmax(-1)
        particle_params = self.world_effect(worlds)
        particle_residuals = particle_params[..., 0]
        particle_scales = F.softplus(particle_params[..., 1]) + compiled_se[:, None] + 0.02
        particle_means = effect_mean[:, None] + gate[:, None] * particle_residuals
        mixture_mean = (world_weights * particle_means).sum(-1)
        second_moment = (
            world_weights * (particle_scales.square() + particle_means.square())
        ).sum(-1)
        effect_scale = (second_moment - mixture_mean.square()).clamp_min(1e-6).sqrt()

        regime_evidence = batch["regime_evidence"]
        candidate_logits = self.regime_candidate(regime_evidence).squeeze(-1)
        eligible = (roles == ROLE_COVARIATE) & variable_mask
        candidate_logits = candidate_logits.masked_fill(
            ~eligible[:, :, None], -1e4
        )
        flat_candidates = candidate_logits.flatten(1)
        best_evidence = regime_evidence[..., 0].amax(dim=(1, 2))
        mean_evidence = regime_evidence[..., 0].mean(dim=(1, 2))
        split_logit = self.split_head(
            torch.cat(
                [causal_token, best_evidence[:, None], mean_evidence[:, None]], -1
            )
        ).squeeze(-1)
        variable_logits = torch.logsumexp(candidate_logits, dim=-1)
        threshold_logits = torch.logsumexp(candidate_logits, dim=1)
        structure_logits = torch.cat(
            [variable_logits + F.logsigmoid(split_logit)[:, None], F.logsigmoid(-split_logit)[:, None]],
            dim=-1,
        )

        graph_logits = torch.matmul(
            self.graph_left(variable_tokens),
            self.graph_right(variable_tokens).transpose(1, 2),
        ) / self.config.hidden_dim**0.5
        graph_shift = self.world_graph_shift(worlds)
        world_graph_logits = (
            graph_logits[:, None]
            * (1.0 + 0.25 * torch.tanh(graph_shift[..., 0]))[:, :, None, None]
            + graph_shift[..., 1, None, None]
        )
        formal_identified = batch["design"] != DESIGN_NONIDENTIFIED
        formal_supported = batch["design"] != DESIGN_POOR_SUPPORT
        identification_logit = torch.where(
            formal_identified, values.new_tensor(20.0), values.new_tensor(-20.0)
        )
        support_empirical = self.support_head(causal_token).squeeze(-1)
        support_logit = torch.where(
            formal_supported, support_empirical + 6.0, values.new_tensor(-20.0)
        )
        flexible_probability = router_weights[:, 2:5].sum(-1).clamp(1e-6, 1 - 1e-6)
        return {
            "effect_mean": mixture_mean,
            "effect_scale": effect_scale,
            "compiled_mean": compiled,
            "compiled_scale": compiled_se,
            "correction": mixture_mean - compiled.detach(),
            "residual": residual,
            "residual_gate": gate,
            "residual_gate_logit": gate_logit,
            "particle_means": particle_means,
            "particle_scales": particle_scales,
            "world_weights": world_weights,
            "world_mechanism_logits": self.world_mechanism(worlds),
            "world_graph_logits": world_graph_logits,
            "identification_logit": identification_logit,
            "support_logit": support_logit,
            "prior_mismatch_logit": self.prior_mismatch_head(causal_token).squeeze(-1),
            "mechanism_logits": self.mechanism_head(causal_token),
            "estimator_router_logits": router_logits,
            "estimator_weights": router_weights,
            "flexible_route_logit": torch.logit(flexible_probability),
            "split_logit": split_logit,
            "structure_logits": structure_logits,
            "threshold_logits": threshold_logits,
            "change_type_logits": self.change_type_head(causal_token),
            "regime_delta": self.regime_delta_head(causal_token),
            "graph_logits": graph_logits,
            "expert_routing": torch.stack(routing, dim=1),
        }
