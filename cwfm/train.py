from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import (
    EVALUATION_SCENARIOS,
    OOD_DEVELOPMENT_SCENARIOS,
    ROLE_COVARIATE,
    TASK_ATE,
    TASK_INTERFERENCE,
    TASK_REGIME,
    collate_episodes,
    generate_episode,
)
from .estimators import EXPERT_NAMES
from .model import CWFM, CWFMConfig


LOSS_WEIGHTS = {
    "effect_point": 1.0,
    "world_mixture_nll": 0.35,
    "no_harm": 1.0,
    "correction_scale": 0.03,
    "router_risk": 0.45,
    "router_mixture": 0.5,
    "router_balance": 0.02,
    "mechanism": 0.12,
    "world_structure": 0.08,
    "support": 0.15,
    "regime_detection": 1.2,
    "regime_localization": 1.2,
    "regime_threshold": 0.35,
    "regime_change_type": 0.2,
    "regime_delta": 0.25,
    "graph": 0.08,
}


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum()
    if int(count) == 0:
        return values.new_zeros(())
    return values[mask].sum() / count


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach())


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    residual_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"]
    effect_mask = (
        batch["identified"].bool()
        & batch["supported"].bool()
        & (batch["task"] != TASK_REGIME)
    )
    scale = output["effect_scale"].clamp(0.02, 5.0)
    point_per = (output["effect_mean"] - target).square()
    point_loss = _masked_mean(point_per, effect_mask)

    component_scale = output["particle_scales"].clamp(0.02, 5.0)
    component_log_prob = (
        -0.5 * ((target[:, None] - output["particle_means"]) / component_scale).square()
        - component_scale.log()
        - 0.5 * math.log(2 * math.pi)
    )
    mixture_nll_per = -torch.logsumexp(
        output["world_weights"].clamp_min(1e-9).log() + component_log_prob,
        dim=-1,
    )
    mixture_nll = _masked_mean(mixture_nll_per, effect_mask)

    compiled_error = (output["compiled_mean"] - target).square()
    final_error = point_per
    no_harm_per = F.relu(final_error - compiled_error - 0.0025)
    no_harm = _masked_mean(no_harm_per, effect_mask)
    correction_scale = _masked_mean(
        (output["correction"] / output["compiled_scale"].clamp_min(0.03)).square(),
        effect_mask,
    )

    expert_error = (batch["expert_estimates"] - target[:, None]).abs()
    oracle_logits = (-expert_error / 0.12).masked_fill(~batch["expert_mask"], -1e4)
    oracle_weights = oracle_logits.softmax(-1)
    oracle_weights = torch.where(
        batch["expert_mask"].any(-1, keepdim=True),
        oracle_weights,
        torch.zeros_like(oracle_weights),
    )
    router_ce_per = -(
        oracle_weights
        * output["estimator_router_logits"].log_softmax(-1)
    ).sum(-1)
    router_risk = _masked_mean(router_ce_per, effect_mask)
    router_mixture = _masked_mean(compiled_error, effect_mask)
    expert_mass = output["estimator_weights"][effect_mask].mean(0) if effect_mask.any() else target.new_zeros(len(EXPERT_NAMES))
    router_balance = ((expert_mass - 1.0 / len(EXPERT_NAMES)) ** 2).mean()

    support_loss = F.binary_cross_entropy_with_logits(
        output["support_logit"], batch["supported"]
    )
    mechanism_loss = F.cross_entropy(
        output["mechanism_logits"], batch["mechanism"]
    )
    world_mechanism_log_prob = output["world_mechanism_logits"].log_softmax(-1)
    mechanism_index = batch["mechanism"][:, None, None].expand(
        -1, output["world_mechanism_logits"].shape[1], 1
    )
    selected_mechanism_log_prob = world_mechanism_log_prob.gather(
        -1, mechanism_index
    ).squeeze(-1)
    world_structure = -torch.logsumexp(
        output["world_weights"].clamp_min(1e-9).log()
        + selected_mechanism_log_prob,
        dim=-1,
    ).mean()

    regime = batch["task"] == TASK_REGIME
    split = regime & batch["split_target"].bool()
    if regime.any():
        positives = batch["split_target"][regime].sum()
        negatives = regime.sum() - positives
        pos_weight = (negatives / positives.clamp_min(1)).clamp(0.5, 4.0)
        regime_detection = F.binary_cross_entropy_with_logits(
            output["split_logit"][regime],
            batch["split_target"][regime],
            pos_weight=pos_weight,
        )
    else:
        regime_detection = target.new_zeros(())
    regime_localization = (
        F.cross_entropy(
            output["structure_logits"][split, :-1],
            batch["structure_target"][split],
        )
        if split.any()
        else target.new_zeros(())
    )
    if split.any():
        split_indices = torch.nonzero(split, as_tuple=False).flatten()
        variables = batch["structure_target"][split]
        candidates = batch["threshold_values"][split_indices, variables]
        threshold_target = (
            candidates - batch["regime_threshold"][split, None]
        ).abs().argmin(-1)
        regime_threshold = F.cross_entropy(
            output["threshold_logits"][split], threshold_target
        )
        regime_change_type = F.cross_entropy(
            output["change_type_logits"][split], batch["change_type"][split]
        )
        delta_mean = output["regime_delta"][split, 0]
        delta_scale = F.softplus(output["regime_delta"][split, 1]) + 0.03
        regime_delta = (
            0.5
            * (
                ((target[split] - delta_mean) / delta_scale).square()
                + 2 * delta_scale.log()
            )
        ).mean()
    else:
        regime_threshold = target.new_zeros(())
        regime_change_type = target.new_zeros(())
        regime_delta = target.new_zeros(())

    variable_mask = batch["variable_mask"]
    graph_mask = variable_mask[:, :, None] & variable_mask[:, None, :]
    eye = torch.eye(graph_mask.shape[-1], device=graph_mask.device, dtype=torch.bool)
    graph_mask &= ~eye.unsqueeze(0)
    graph_targets = batch["graph"][:, None].expand_as(
        output["world_graph_logits"]
    )
    particle_graph_loss = F.binary_cross_entropy_with_logits(
        output["world_graph_logits"],
        graph_targets,
        pos_weight=target.new_tensor(2.0),
        reduction="none",
    )
    particle_graph_loss = (
        particle_graph_loss * graph_mask[:, None]
    ).sum(dim=(-1, -2)) / graph_mask.sum(dim=(-1, -2))[:, None].clamp_min(1)
    graph_loss = (particle_graph_loss * output["world_weights"]).sum(-1).mean()

    raw = {
        "effect_point": point_loss,
        "world_mixture_nll": mixture_nll,
        "no_harm": no_harm,
        "correction_scale": correction_scale,
        "router_risk": router_risk,
        "router_mixture": router_mixture,
        "router_balance": router_balance,
        "mechanism": mechanism_loss,
        "world_structure": world_structure,
        "support": support_loss,
        "regime_detection": regime_detection,
        "regime_localization": regime_localization,
        "regime_threshold": regime_threshold,
        "regime_change_type": regime_change_type,
        "regime_delta": regime_delta,
        "graph": graph_loss,
    }
    residual_terms = {"effect_point", "world_mixture_nll", "no_harm", "correction_scale"}
    weighted = {
        name: value
        * LOSS_WEIGHTS[name]
        * (residual_weight if name in residual_terms else 1.0)
        for name, value in raw.items()
    }
    loss = sum(weighted.values())
    parts = {"loss/total": _as_float(loss), "loss": _as_float(loss)}
    for name, value in raw.items():
        parts[f"loss/{name}_raw"] = _as_float(value)
        parts[f"loss/{name}_weighted"] = _as_float(weighted[name])
    # Backward-compatible concise keys remain useful to downstream notebooks.
    parts.update(
        {
            "effect": _as_float(point_loss),
            "identification": 0.0,
            "support": _as_float(support_loss),
            "mechanism": _as_float(mechanism_loss),
            "structure": _as_float(regime_detection + regime_localization),
            "graph": _as_float(graph_loss),
            "count/estimable": int(effect_mask.sum()),
            "count/identified": int(batch["identified"].sum()),
            "count/supported": int(batch["supported"].sum()),
            "count/regime": int(regime.sum()),
            "count/split": int(split.sum()),
            "count/non_split": int((regime & ~batch["split_target"].bool()).sum()),
            "count/task_ate": int((batch["task"] == TASK_ATE).sum()),
            "count/task_interference": int((batch["task"] == TASK_INTERFERENCE).sum()),
            "count/task_regime": int(regime.sum()),
        }
    )
    return loss, parts


def _raw_predictions(
    model: CWFM, episodes: list, device: torch.device
) -> tuple[dict[str, np.ndarray], dict]:
    batch = _to_device(collate_episodes(episodes, model.config.max_variables), device)
    with torch.no_grad():
        output = model(batch)
    raw_scale = batch["raw_scale"]
    arrays = {
        "target": batch["raw_target"].cpu().numpy(),
        "final": (output["effect_mean"] * raw_scale).cpu().numpy(),
        "compiled": (output["compiled_mean"] * raw_scale).cpu().numpy(),
        "compiled_se": (output["compiled_scale"] * raw_scale).cpu().numpy(),
        "scale": (output["effect_scale"] * raw_scale).cpu().numpy(),
        "correction": (output["correction"] * raw_scale).cpu().numpy(),
        "gate": output["residual_gate"].cpu().numpy(),
        "weights": output["estimator_weights"].cpu().numpy(),
        "expert_estimates": (
            batch["expert_estimates"] * raw_scale[:, None]
        ).cpu().numpy(),
        "world_weights": output["world_weights"].cpu().numpy(),
        "split_probability": output["split_logit"].sigmoid().cpu().numpy(),
        "structure": output["structure_logits"].argmax(-1).cpu().numpy(),
        "threshold_index": output["threshold_logits"].argmax(-1).cpu().numpy(),
        "identified_probability": output["identification_logit"].sigmoid().cpu().numpy(),
        "support_probability": output["support_logit"].sigmoid().cpu().numpy(),
    }
    return arrays, batch


def _robustness_metrics(
    model: CWFM, episode, device: torch.device
) -> dict[str, float]:
    base, _ = _raw_predictions(model, [episode], device)
    base_effect = float(base["final"][0])

    row_permuted = copy.deepcopy(episode)
    order = np.random.default_rng(823).permutation(len(row_permuted.values))
    row_permuted.values = row_permuted.values[order]
    row_permuted.adjacency = row_permuted.adjacency[np.ix_(order, order)]
    row_permuted.cluster_ids = row_permuted.cluster_ids[order]
    row, _ = _raw_predictions(model, [row_permuted], device)

    variable_permuted = copy.deepcopy(episode)
    ordinary = np.flatnonzero(variable_permuted.roles == ROLE_COVARIATE)
    permutation = np.arange(variable_permuted.values.shape[1])
    permutation[ordinary] = ordinary[::-1]
    inverse = np.argsort(permutation)
    variable_permuted.values = variable_permuted.values[:, permutation]
    variable_permuted.roles = variable_permuted.roles[permutation]
    variable_permuted.graph = variable_permuted.graph[np.ix_(permutation, permutation)]
    if variable_permuted.structure_target < len(permutation):
        variable_permuted.structure_target = int(inverse[variable_permuted.structure_target])
    variable, _ = _raw_predictions(model, [variable_permuted], device)

    scaled = copy.deepcopy(episode)
    outcome = np.flatnonzero(scaled.roles == 2)[0]
    scaled.values[:, outcome] *= 1.7
    scaled.target *= 1.7
    affine, _ = _raw_predictions(model, [scaled], device)

    metrics = {
        "robustness/row_permutation_delta": abs(float(row["final"][0]) - base_effect),
        "robustness/variable_equivariance_error": abs(float(variable["final"][0]) - base_effect),
        "robustness/affine_transform_delta": abs(float(affine["final"][0]) / 1.7 - base_effect),
    }
    if episode.values.shape[1] < model.config.max_variables:
        irrelevant = copy.deepcopy(episode)
        noise = np.random.default_rng(717).normal(size=len(irrelevant.values))
        irrelevant.values = np.column_stack([irrelevant.values, noise]).astype(np.float32)
        irrelevant.roles = np.r_[irrelevant.roles, ROLE_COVARIATE]
        graph = np.zeros((len(irrelevant.roles), len(irrelevant.roles)), dtype=np.float32)
        graph[:-1, :-1] = irrelevant.graph
        irrelevant.graph = graph
        inserted, _ = _raw_predictions(model, [irrelevant], device)
        metrics["robustness/irrelevant_variable_delta"] = abs(
            float(inserted["final"][0]) - base_effect
        )
    return metrics


@torch.no_grad()
def validate(
    model: CWFM,
    device: torch.device,
    seed: int,
    episodes_per_scenario: int = 4,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    all_records: list[dict[str, Any]] = []
    banks = {
        "validation_id": EVALUATION_SCENARIOS,
        "validation_ood": OOD_DEVELOPMENT_SCENARIOS,
    }
    bank_arrays: dict[str, dict[str, np.ndarray]] = {}
    bank_episodes: dict[str, list] = {}
    for bank_index, (split_name, scenarios) in enumerate(banks.items()):
        episodes = [
            generate_episode(
                seed + bank_index * 1_000_000 + scenario_index * 10_000 + offset,
                task,
                scenario,
            )
            for scenario_index, (task, scenario) in enumerate(scenarios)
            for offset in range(episodes_per_scenario)
        ]
        arrays, _ = _raw_predictions(model, episodes, device)
        bank_arrays[split_name] = arrays
        bank_episodes[split_name] = episodes
        for index, episode in enumerate(episodes):
            if episode.task == TASK_REGIME:
                all_records.append(
                    {
                        "split": split_name,
                        "task": "observed_regime",
                        "scenario": episode.scenario,
                        "target": episode.structure_target,
                        "split_truth": float(
                            episode.structure_target < len(episode.roles)
                        ),
                        "split_probability": float(arrays["split_probability"][index]),
                        "structure": int(arrays["structure"][index]),
                    }
                )
                continue
            all_records.append(
                {
                    "split": split_name,
                    "task": "static_ate"
                    if episode.task == TASK_ATE
                    else "network_interference",
                    "scenario": episode.scenario,
                    "target": float(arrays["target"][index]),
                    "estimable": bool(
                        episode.identified and episode.supported
                    ),
                    "identified_probability": float(
                        arrays["identified_probability"][index]
                    ),
                    "support_probability": float(
                        arrays["support_probability"][index]
                    ),
                    "final": float(arrays["final"][index]),
                    "compiled": float(arrays["compiled"][index]),
                    "scale": float(arrays["scale"][index]),
                    "correction": float(arrays["correction"][index]),
                    "gate": float(arrays["gate"][index]),
                    "weights": arrays["weights"][index].tolist(),
                    "expert_estimates": arrays["expert_estimates"][index].tolist(),
                    "world_weights": arrays["world_weights"][index].tolist(),
                }
            )

    metrics: dict[str, float] = {}
    for split_name in banks:
        stress_records = [
            record
            for record in all_records
            if record["split"] == split_name and "final" in record
        ]
        records = [
            record for record in stress_records if record["estimable"]
        ]
        final_error = np.asarray([abs(r["final"] - r["target"]) for r in records])
        compiled_error = np.asarray([abs(r["compiled"] - r["target"]) for r in records])
        signed = np.asarray([r["final"] - r["target"] for r in records])
        correction = np.asarray([r["correction"] for r in records])
        gate = np.asarray([r["gate"] for r in records])
        weights = np.asarray([r["weights"] for r in records])
        world_weights = np.asarray([r["world_weights"] for r in records])
        expert_error = np.abs(
            np.asarray([r["expert_estimates"] for r in records])
            - np.asarray([r["target"] for r in records])[:, None]
        )
        routed_regret = compiled_error - expert_error.min(1)
        prefix = f"{split_name}/"
        metrics.update(
            {
                prefix + "effect/mae": float(final_error.mean()),
                prefix + "effect/rmse": float(np.sqrt(np.mean(final_error**2))),
                prefix + "effect/signed_bias": float(signed.mean()),
                prefix + "effect/median_absolute_error": float(np.median(final_error)),
                prefix + "effect/q90_absolute_error": float(np.quantile(final_error, 0.9)),
                prefix + "residual/mean_gain": float((compiled_error - final_error).mean()),
                prefix + "residual/win_rate": float(np.mean(final_error < compiled_error)),
                prefix + "residual/harm_rate": float(np.mean(final_error > compiled_error + 1e-6)),
                prefix + "residual/catastrophic_harm_rate": float(
                    np.mean(final_error - compiled_error > 0.2)
                ),
                prefix + "residual/correction_abs_mean": float(np.abs(correction).mean()),
                prefix + "residual/correction_q90": float(np.quantile(np.abs(correction), 0.9)),
                prefix + "gate/alpha_mean": float(gate.mean()),
                prefix + "gate/alpha_std": float(gate.std()),
                prefix + "gate/alpha_p10": float(np.quantile(gate, 0.1)),
                prefix + "gate/alpha_p50": float(np.quantile(gate, 0.5)),
                prefix + "gate/alpha_p90": float(np.quantile(gate, 0.9)),
                prefix + "router/mean_regret": float(routed_regret.mean()),
                prefix + "router/q90_regret": float(np.quantile(routed_regret, 0.9)),
                prefix + "router/oracle_error": float(expert_error.min(1).mean()),
                prefix + "router/best_single_expert_error": float(
                    expert_error.mean(0).min()
                ),
                prefix + "router/entropy": float(
                    (-(weights * np.log(np.clip(weights, 1e-9, 1))).sum(1)).mean()
                ),
                prefix + "router/effective_experts": float(
                    (1.0 / np.square(weights).sum(1).clip(1e-9)).mean()
                ),
                prefix + "router/dead_expert_count": float(
                    np.sum(weights.mean(0) < 0.01)
                ),
                prefix + "world/entropy_mean": float(
                    (-(world_weights * np.log(np.clip(world_weights, 1e-9, 1))).sum(1)).mean()
                ),
                prefix + "world/entropy_std_across_episodes": float(
                    (-(world_weights * np.log(np.clip(world_weights, 1e-9, 1))).sum(1)).std()
                ),
                prefix + "world/effective_particle_count": float(
                    (1.0 / np.square(world_weights).sum(1)).mean()
                ),
                prefix + "uncertainty/coverage_90": float(
                    np.mean(
                        [
                            abs(r["final"] - r["target"]) <= 1.645 * r["scale"]
                            for r in records
                        ]
                    )
                ),
                prefix + "uncertainty/mean_width_90": float(
                    np.mean([2 * 1.645 * r["scale"] for r in records])
                ),
            }
        )
        rejected = [
            record
            for record in stress_records
            if not record["estimable"]
        ]
        metrics[prefix + "safety/false_answer_rate"] = float(
            np.mean(
                [
                    record["identified_probability"] >= 0.5
                    and record["support_probability"] >= 0.5
                    for record in rejected
                ]
            )
            if rejected
            else 0.0
        )
        for task_name in sorted({record["task"] for record in records}):
            task_records = [
                record for record in records if record["task"] == task_name
            ]
            task_error = np.asarray(
                [abs(r["final"] - r["target"]) for r in task_records]
            )
            task_signed = np.asarray(
                [r["final"] - r["target"] for r in task_records]
            )
            metrics[prefix + f"effect/{task_name}_mae"] = float(
                task_error.mean()
            )
            metrics[prefix + f"effect/{task_name}_bias"] = float(
                task_signed.mean()
            )
            metrics[prefix + f"effect/{task_name}_q90"] = float(
                np.quantile(task_error, 0.9)
            )
        for scenario_name in sorted(
            {record["scenario"] for record in records}
        ):
            scenario_error = [
                abs(record["final"] - record["target"])
                for record in records
                if record["scenario"] == scenario_name
            ]
            metrics[prefix + f"scenario/{scenario_name}_mae"] = float(
                np.mean(scenario_error)
            )
        scenario_mae = [
            np.mean(
                [
                    abs(r["final"] - r["target"])
                    for r in records
                    if r["scenario"] == scenario
                ]
            )
            for scenario in sorted({r["scenario"] for r in records})
        ]
        metrics[prefix + "effect/worst_group_mae"] = float(max(scenario_mae))
        for expert_index, expert_name in enumerate(EXPERT_NAMES):
            metrics[prefix + f"router/{expert_name}_weight"] = float(
                weights[:, expert_index].mean()
            )

        regime_records = [
            record
            for record in all_records
            if record["split"] == split_name and "split_truth" in record
        ]
        if regime_records:
            truth = np.asarray([r["split_truth"] for r in regime_records], dtype=bool)
            predicted = np.asarray(
                [r["split_probability"] >= 0.5 for r in regime_records]
            )
            strong = np.asarray(
                ["linear_split" in r["scenario"] or "intercept" in r["scenario"] for r in regime_records]
            )
            weak = np.asarray(["weak_split" in r["scenario"] for r in regime_records])
            localized = np.asarray(
                [
                    r["structure"] == r["target"]
                    for r in regime_records
                ]
            )
            metrics[prefix + "regime/null_false_positive_rate"] = float(
                predicted[~truth].mean() if (~truth).any() else 0.0
            )
            metrics[prefix + "regime/strong_split_recall"] = float(
                predicted[strong].mean() if strong.any() else 0.0
            )
            metrics[prefix + "regime/weak_split_recall"] = float(
                predicted[weak].mean() if weak.any() else 0.0
            )
            metrics[prefix + "regime/variable_accuracy_given_split"] = float(
                localized[truth].mean() if truth.any() else 0.0
            )
            metrics[prefix + "regime/predicted_no_split_rate"] = float(
                (~predicted).mean()
            )

    # Metadata-only shortcut baseline: task/design/n/p should have little
    # predictive value for signed effects on a genuinely randomized prior.
    id_effect = [
        (episode, record)
        for episode, record in zip(
            bank_episodes["validation_id"],
            [r for r in all_records if r["split"] == "validation_id"],
        )
        if "final" in record
    ]
    if id_effect:
        features = np.asarray(
            [
                [
                    1,
                    episode.task,
                    episode.design,
                    len(episode.values) / 128,
                    episode.values.shape[1] / 12,
                ]
                for episode, _ in id_effect
            ]
        )
        targets = np.asarray([record["target"] for _, record in id_effect])
        coefficient = np.linalg.lstsq(features, targets, rcond=None)[0]
        metrics["validation_id/shortcut/metadata_only_mae"] = float(
            np.mean(np.abs(features @ coefficient - targets))
        )

    probe = generate_episode(seed + 9_999_999, TASK_ATE, "linear")
    metrics.update(_robustness_metrics(model, probe, device))
    return metrics, all_records


def _parameter_norm(model: CWFM) -> float:
    return float(
        torch.sqrt(
            sum(parameter.detach().float().square().sum() for parameter in model.parameters())
        )
    )


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    output: Path,
    steps: int = 1400,
    batch_size: int = 16,
    seed: int = 1729,
    device_name: str = "auto",
    validation_interval: int = 100,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    config = CWFMConfig()
    model = CWFM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    train_metrics = output.parent / "train_metrics.jsonl"
    validation_metrics = output.parent / "validation_metrics.jsonl"
    validation_diagnostics = (
        output.parent / "validation_episode_diagnostics.jsonl"
    )
    train_metrics.unlink(missing_ok=True)
    validation_metrics.unlink(missing_ok=True)
    validation_diagnostics.unlink(missing_ok=True)

    started = time.time()
    clipped_steps = 0
    best_score = float("inf")
    best_step = 0
    best_validation: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    previous_parameters = _parameter_norm(model)
    history: list[dict[str, Any]] = []
    no_improvement_rounds = 0

    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        stage = (
            "compiled_and_router"
            if fraction < 0.15
            else "residual"
            if fraction < 0.65
            else "joint_finetune"
        )
        residual_weight = 0.0 if stage == "compiled_and_router" else 1.0
        for parameter in list(model.residual_heads.parameters()) + list(
            model.residual_gate.parameters()
        ):
            parameter.requires_grad_(residual_weight > 0)
        model.train()
        episodes = [
            generate_episode(seed + step * batch_size + offset)
            for offset in range(batch_size)
        ]
        batch = _to_device(
            collate_episodes(episodes, config.max_variables), device
        )
        optimizer.zero_grad(set_to_none=True)
        output_values = model(batch)
        loss, parts = compute_loss(output_values, batch, residual_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped = float(grad_norm > 1.0)
        clipped_steps += int(clipped)
        nonfinite_gradients = sum(
            int(not torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == steps - 1:
            parameter_norm = _parameter_norm(model)
            record: dict[str, Any] = {
                "step": step + 1,
                "stage": stage,
                "optim/learning_rate": optimizer.param_groups[0]["lr"],
                "optim/grad_norm_preclip": float(grad_norm),
                "optim/grad_norm_after_clip": float(min(float(grad_norm), 1.0)),
                "optim/grad_clipped": clipped,
                "optim/clipping_fraction": clipped_steps / (step + 1),
                "optim/parameter_norm": parameter_norm,
                "optim/update_to_parameter_norm": abs(parameter_norm - previous_parameters)
                / max(previous_parameters, 1e-9),
                "optim/nonfinite_gradient_count": nonfinite_gradients,
                "optim/episodes_per_second": (step + 1)
                * batch_size
                / max(time.time() - started, 1e-9),
                "target/mean": float(batch["raw_target"].mean()),
                "target/std": float(batch["raw_target"].std()),
                "target/q10": float(batch["raw_target"].quantile(0.1)),
                "target/q90": float(batch["raw_target"].quantile(0.9)),
                **parts,
            }
            previous_parameters = parameter_norm
            history.append(record)
            _append_jsonl(train_metrics, [record])
            print(json.dumps(record), flush=True)

        should_validate = (
            (step + 1) % validation_interval == 0 or step == steps - 1
        )
        if should_validate:
            validation, records = validate(
                model, device, seed + 10_000_000, episodes_per_scenario=4
            )
            validation_rows = [
                {
                    "step": step + 1,
                    "split": key.split("/", 1)[0]
                    if "/" in key
                    else "robustness",
                    "metric": key,
                    "value": value,
                }
                for key, value in validation.items()
            ]
            _append_jsonl(validation_metrics, validation_rows)
            eligible = (
                all(np.isfinite(list(validation.values())))
                and validation["validation_id/residual/mean_gain"] >= -0.03
                and validation["validation_ood/residual/mean_gain"] >= -0.05
                and validation["validation_id/regime/null_false_positive_rate"] <= 0.30
                and validation["validation_id/router/dead_expert_count"] <= 3
            )
            score = (
                validation["validation_ood/effect/worst_group_mae"]
                + 0.4 * validation["validation_id/effect/mae"]
                + 0.2 * validation["validation_ood/router/mean_regret"]
            )
            if eligible and score < best_score - 1e-5:
                best_score = score
                best_step = step + 1
                best_validation = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "validation_score": score,
                        "eligible": eligible,
                        "best_step": best_step,
                        "id_mae": validation["validation_id/effect/mae"],
                        "ood_mae": validation["validation_ood/effect/mae"],
                        "residual_gain": validation[
                            "validation_id/residual/mean_gain"
                        ],
                    }
                ),
                flush=True,
            )
            # Keep training for the requested fixed episode budget; the
            # patience diagnostic is recorded but does not silently shorten a
            # confirmatory run.
            _append_jsonl(
                validation_diagnostics,
                [{"step": step + 1, **record} for record in records],
            )

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_step = steps
        best_validation, _ = validate(
            model, device, seed + 10_000_000, episodes_per_scenario=4
        )
    model.load_state_dict(best_state)
    runtime = time.time() - started
    payload = {
        "state_dict": best_state,
        "config": config.to_dict(),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "best_step": best_step,
        "validation": best_validation,
        "runtime_seconds": runtime,
        "torch_version": torch.__version__,
        "stream_seeds": {
            "training_start": seed,
            "id_development_start": seed + 10_000_000,
            "ood_development_start": seed + 11_000_000,
            "calibration_start": 19_000_000,
            "final_test_start": 20_000_000,
        },
    }
    torch.save(payload, output)
    manifest = {
        "status": "completed",
        "configuration": {
            **config.to_dict(),
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "device": str(device),
            "validation_interval": validation_interval,
        },
        "loss_weights": LOSS_WEIGHTS,
        "expert_definitions": list(EXPERT_NAMES),
        "streams": payload["stream_seeds"],
        "checkpoint_selection": {
            "rule": "eligibility constraints then OOD-worst/ID/router lexicographic composite",
            "best_step": best_step,
            "best_score": best_score,
            "rounds_without_improvement": no_improvement_rounds,
        },
        "source_sha256": {
            str(path): _source_hash(path)
            for path in [
                Path("cwfm/model.py"),
                Path("cwfm/data.py"),
                Path("cwfm/estimators.py"),
                Path("cwfm/train.py"),
            ]
        },
        "episodes_seen": steps * batch_size,
        "runtime_seconds": runtime,
        "torch_version": torch.__version__,
    }
    (output.parent / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    output.with_suffix(".history.json").write_text(
        json.dumps(
            {
                "configuration": manifest["configuration"],
                "history": history,
                "validation": best_validation,
                "runtime_seconds": runtime,
            },
            indent=2,
        )
        + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the revised CWFM")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cwfm/checkpoint.pt")
    )
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-interval", type=int, default=100)
    args = parser.parse_args()
    train(
        args.output,
        args.steps,
        args.batch_size,
        args.seed,
        args.device,
        args.validation_interval,
    )


if __name__ == "__main__":
    main()
