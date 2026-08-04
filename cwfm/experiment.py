from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines import all_baselines
from .data import (
    EVALUATION_SCENARIOS,
    TASK_ATE,
    TASK_INTERFERENCE,
    TASK_REGIME,
    collate_episodes,
    generate_episode,
)
from .estimators import EXPERT_NAMES
from .model import CWFM, CWFMConfig


TASK_NAMES = {
    TASK_ATE: "static_ate",
    TASK_REGIME: "observed_regime",
    TASK_INTERFERENCE: "network_interference",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _predict_model(
    model: CWFM, episodes: list, device: torch.device
) -> list[dict[str, Any]]:
    batch = _to_device(
        collate_episodes(episodes, model.config.max_variables), device
    )
    model.eval()
    with torch.no_grad():
        output = model(batch)
    raw_scale = batch["raw_scale"]
    final = output["effect_mean"] * raw_scale
    compiled = output["compiled_mean"] * raw_scale
    scales = output["effect_scale"] * raw_scale
    compiled_scales = output["compiled_scale"] * raw_scale
    id_probability = output["identification_logit"].sigmoid()
    support_probability = output["support_logit"].sigmoid()
    structure = output["structure_logits"].argmax(-1)
    threshold_index = output["threshold_logits"].argmax(-1)
    result = []
    for index, episode in enumerate(episodes):
        selected_structure = int(structure[index].cpu())
        selected_threshold = np.nan
        if (
            episode.task == TASK_REGIME
            and selected_structure < model.config.max_variables
        ):
            candidate = int(threshold_index[index].cpu())
            selected_threshold = float(
                np.nanquantile(
                    episode.values[:, selected_structure],
                    [0.2, 0.35, 0.5, 0.65, 0.8],
                )[candidate]
            )
        evidence = batch["regime_evidence"][index, ..., 0]
        evidence = evidence.masked_fill(
            batch["roles"][index, :, None] != 0, -torch.inf
        )
        evidence_flat_index = int(evidence.flatten().argmax().cpu())
        evidence_structure = evidence_flat_index // evidence.shape[-1]
        evidence_threshold_index = evidence_flat_index % evidence.shape[-1]
        evidence_threshold = (
            float(
                np.nanquantile(
                    episode.values[:, evidence_structure],
                    [0.2, 0.35, 0.5, 0.65, 0.8],
                )[evidence_threshold_index]
            )
            if episode.task == TASK_REGIME
            and evidence_structure < len(episode.roles)
            else np.nan
        )
        weights = output["estimator_weights"][index].cpu().numpy()
        result.append(
            {
                "final": float(final[index].cpu()),
                "compiled": float(compiled[index].cpu()),
                "outcome_scale": float(raw_scale[index].cpu()),
                "raw_scale": float(scales[index].cpu()),
                "compiled_scale": float(compiled_scales[index].cpu()),
                "correction": float(
                    (output["correction"][index] * raw_scale[index]).cpu()
                ),
                "residual_gate": float(output["residual_gate"][index].cpu()),
                "id_probability": float(id_probability[index].cpu()),
                "support_probability": float(support_probability[index].cpu()),
                "prior_mismatch_probability": float(
                    output["prior_mismatch_logit"][index].sigmoid().cpu()
                ),
                "flexible_probability": float(
                    output["flexible_route_logit"][index].sigmoid().cpu()
                ),
                "estimator_weights": weights.tolist(),
                "estimator_router_logits": output["estimator_router_logits"][
                    index
                ].cpu().numpy().tolist(),
                "selected_expert": EXPERT_NAMES[int(weights.argmax())],
                "expert_estimates": (
                    batch["expert_estimates"][index] * raw_scale[index]
                ).cpu().numpy().tolist(),
                "expert_standard_errors": (
                    batch["expert_standard_errors"][index] * raw_scale[index]
                ).cpu().numpy().tolist(),
                "expert_mask": batch["expert_mask"][index].cpu().numpy().tolist(),
                "structure": selected_structure,
                "split_probability": float(
                    output["split_logit"][index].sigmoid().cpu()
                ),
                "threshold": selected_threshold,
                "regime_evidence_score": float(
                    evidence.max().clamp_min(0).cpu()
                ),
                "regime_evidence_structure": int(evidence_structure),
                "regime_evidence_threshold": evidence_threshold,
                "change_type": int(
                    output["change_type_logits"][index].argmax().cpu()
                ),
                "world_weights": output["world_weights"][index].cpu().numpy().tolist(),
                "world_entropy": float(
                    -(
                        output["world_weights"][index]
                        * output["world_weights"][index].clamp_min(1e-9).log()
                    )
                    .sum()
                    .cpu()
                ),
                "world_effective_count": float(
                    (
                        1.0
                        / output["world_weights"][index].square().sum().clamp_min(1e-9)
                    ).cpu()
                ),
            }
        )
    return result


def _point_predictions(
    model: CWFM, episodes: list, device: torch.device
) -> list[dict[str, Any]]:
    neural = _predict_model(model, episodes, device)
    return [
        {
            "neural": prediction,
            "baselines": all_baselines(episode),
        }
        for episode, prediction in zip(episodes, neural)
    ]


def _calibration_key(task: str, prediction: dict[str, Any]) -> str:
    disagreement = float(np.std(prediction["expert_estimates"]))
    bucket = "high_disagreement" if disagreement > 0.25 else "low_disagreement"
    return f"{task}|{prediction['selected_expert']}|{bucket}"


def _conformal_quantile(
    values: list[float], coverage: float = 0.90
) -> float:
    """Finite-sample split-conformal order statistic."""
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(ordered) + 1) * coverage)) - 1
    return float(ordered[min(max(rank, 0), len(ordered) - 1)])


def _calibration(
    model: CWFM, device: torch.device, count: int, seed_start: int
) -> dict[str, Any]:
    normalized: dict[str, list[float]] = {}
    absolute: dict[str, dict[str, list[float]]] = {}
    regime_scores: list[float] = []
    regime_truth: list[bool] = []
    chunk = 24
    for start in range(0, count, chunk):
        episodes = [
            generate_episode(seed_start + index, index % 3)
            for index in range(start, min(start + chunk, count))
        ]
        predictions = _point_predictions(model, episodes, device)
        for episode, values in zip(episodes, predictions):
            if episode.task == TASK_REGIME:
                regime_scores.append(
                    float(values["neural"]["regime_evidence_score"])
                )
                regime_truth.append(
                    episode.structure_target < len(episode.roles)
                )
                continue
            if not (
                episode.identified and episode.supported
            ):
                continue
            task = TASK_NAMES[episode.task]
            prediction = values["neural"]
            score = abs(prediction["final"] - episode.target) / max(
                prediction["raw_scale"], 1e-4
            )
            normalized.setdefault(task, []).append(score)
            normalized.setdefault(
                _calibration_key(task, prediction), []
            ).append(score)
            methods = {
                "CWFM-Compiled": prediction["compiled"],
                **{
                    str(name): float(value)
                    for name, value in values["baselines"].items()
                },
            }
            for method, estimate in methods.items():
                absolute.setdefault(task, {}).setdefault(method, []).append(
                    abs(estimate - episode.target)
                )
    normalized_radius = {
        key: _conformal_quantile(values)
        for key, values in normalized.items()
        if len(values) >= 12 or "|" not in key
    }
    absolute_radius = {
        task: {
            method: _conformal_quantile(errors)
            for method, errors in methods.items()
        }
        for task, methods in absolute.items()
    }
    scores = np.asarray(regime_scores)
    truth = np.asarray(regime_truth, dtype=bool)
    threshold_candidates = np.unique(
        np.r_[scores, np.nextafter(scores.max(initial=0.0), np.inf)]
    )
    feasible: list[tuple[float, float, float]] = []
    for threshold in threshold_candidates:
        predicted = scores >= threshold
        false_positive_rate = (
            float(predicted[~truth].mean()) if (~truth).any() else 0.0
        )
        recall = float(predicted[truth].mean()) if truth.any() else 0.0
        if false_positive_rate <= 0.10:
            feasible.append((recall, -false_positive_rate, float(threshold)))
    selected = max(feasible) if feasible else (0.0, 0.0, float(scores.max(initial=0.0)))
    return {
        "normalized_radius": normalized_radius,
        "absolute_radius": absolute_radius,
        "counts": {key: len(values) for key, values in normalized.items()},
        "regime": {
            "evidence_threshold": selected[2],
            "calibration_recall": selected[0],
            "calibration_false_positive_rate": -selected[1],
            "episodes": len(scores),
        },
    }


def _bootstrap_difference(
    frame: pd.DataFrame, first: str, second: str, seed: int = 921
) -> tuple[float, float, float]:
    pivot = frame.pivot_table(
        index=["scenario", "seed"], columns="method", values="absolute_error"
    ).dropna(subset=[first, second])
    differences = pivot[first].to_numpy() - pivot[second].to_numpy()
    if not len(differences):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.mean(rng.choice(differences, size=len(differences), replace=True))
            for _ in range(4000)
        ]
    )
    return (
        float(np.mean(differences)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def run(
    checkpoint: Path,
    output_dir: Path,
    seeds: int = 20,
    calibration_episodes: int = 240,
    seed_start: int = 20_000_000,
    device_name: str = "auto",
) -> dict[str, Any]:
    started = time.time()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CWFM(CWFMConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["state_dict"])
    calibration = _calibration(
        model, device, calibration_episodes, seed_start - 1_000_000
    )
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for task, scenario in EVALUATION_SCENARIOS:
        episodes = [
            generate_episode(
                seed_start + task * 100_000 + scenario_index,
                task,
                scenario,
            )
            for scenario_index in range(seeds)
        ]
        predictions = _point_predictions(model, episodes, device)
        for episode, values in zip(episodes, predictions):
            neural = values["neural"]
            gate = (
                neural["id_probability"] >= 0.5
                and neural["support_probability"] >= 0.5
            )
            diagnostics.append(
                {
                    "scenario": episode.scenario,
                    "seed": episode.seed,
                    "task": TASK_NAMES[episode.task],
                    "identified_truth": episode.identified,
                    "supported_truth": episode.supported,
                    "id_probability": neural["id_probability"],
                    "support_probability": neural["support_probability"],
                    "prior_mismatch_probability": neural[
                        "prior_mismatch_probability"
                    ],
                    "residual_gate": neural["residual_gate"],
                    "correction": neural["correction"],
                    "compiled": neural["compiled"],
                    "final": neural["final"],
                    "selected_expert": neural["selected_expert"],
                    "estimator_weights": json.dumps(neural["estimator_weights"]),
                    "expert_estimates": json.dumps(neural["expert_estimates"]),
                    "world_entropy": neural["world_entropy"],
                    "world_effective_count": neural["world_effective_count"],
                    "world_weights": json.dumps(neural["world_weights"]),
                    "split_probability": neural["split_probability"],
                    "predicted_threshold": neural["threshold"],
                    "regime_evidence_score": neural["regime_evidence_score"],
                }
            )
            if episode.task == TASK_REGIME:
                truth = (
                    episode.structure_target
                    if episode.structure_target < len(episode.roles)
                    else model.config.max_variables
                )
                calibrated_split = (
                    neural["regime_evidence_score"]
                    >= calibration["regime"]["evidence_threshold"]
                )
                revised_structure = (
                    int(neural["regime_evidence_structure"])
                    if calibrated_split
                    else model.config.max_variables
                )
                method_predictions = {
                    "CWFM-Revised": revised_structure,
                    **{
                        name: int(value)
                        for name, value in values["baselines"].items()
                    },
                }
                for method, prediction in method_predictions.items():
                    predicted_split = prediction != model.config.max_variables
                    records.append(
                        {
                            "scenario": episode.scenario,
                            "seed": episode.seed,
                            "task": TASK_NAMES[episode.task],
                            "method": method,
                            "truth": truth,
                            "estimate": prediction,
                            "answered": True,
                            "correct": float(prediction == truth),
                            "split_truth": float(
                                truth != model.config.max_variables
                            ),
                            "split_predicted": float(predicted_split),
                            "split_probability": neural["regime_evidence_score"]
                            if method == "CWFM-Revised"
                            else float(predicted_split),
                            "threshold_truth": episode.threshold
                            if truth != model.config.max_variables
                            else np.nan,
                            "threshold_estimate": neural[
                                "regime_evidence_threshold"
                            ]
                            if method == "CWFM-Revised" and predicted_split
                            else np.nan,
                            "absolute_error": np.nan,
                            "lower": np.nan,
                            "upper": np.nan,
                            "covered": np.nan,
                            "interval_width": np.nan,
                        }
                    )
                continue

            method_predictions = {
                "CWFM-Revised": float(neural["final"]),
                "CWFM-Compiled": float(neural["compiled"]),
                **{
                    str(name): float(value)
                    for name, value in values["baselines"].items()
                },
            }
            task_name = TASK_NAMES[episode.task]
            normalized_key = _calibration_key(task_name, neural)
            normalized_radius = calibration["normalized_radius"].get(
                normalized_key,
                calibration["normalized_radius"][task_name],
            )
            for method, estimate in method_predictions.items():
                if method == "CWFM-Revised":
                    half_width = normalized_radius * neural["raw_scale"]
                    answered = bool(gate)
                else:
                    half_width = calibration["absolute_radius"][task_name][method]
                    answered = True
                records.append(
                    {
                        "scenario": episode.scenario,
                        "seed": episode.seed,
                        "task": task_name,
                        "method": method,
                        "truth": episode.target,
                        "estimate": estimate if answered else np.nan,
                        "latent_estimate": estimate,
                        "answered": answered,
                        "correct": np.nan,
                        "split_truth": np.nan,
                        "split_predicted": np.nan,
                        "split_probability": np.nan,
                        "threshold_truth": np.nan,
                        "threshold_estimate": np.nan,
                        "absolute_error": abs(estimate - episode.target)
                        if answered
                        else np.nan,
                        "lower": estimate - half_width if answered else np.nan,
                        "upper": estimate + half_width if answered else np.nan,
                        "covered": float(
                            estimate - half_width
                            <= episode.target
                            <= estimate + half_width
                        )
                        if answered
                        else np.nan,
                        "interval_width": 2 * half_width if answered else np.nan,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(records)
    per_seed.to_csv(output_dir / "per_seed.csv", index=False)
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(output_dir / "diagnostics.csv", index=False)
    numeric = [
        "absolute_error",
        "covered",
        "interval_width",
        "correct",
        "answered",
        "split_predicted",
        "split_probability",
    ]
    summary = (
        per_seed.groupby(["task", "scenario", "method"], dropna=False)[numeric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "summary.csv", index=False)

    effect_valid = per_seed[
        per_seed["task"].isin(["static_ate", "network_interference"])
        & ~per_seed["scenario"].str.contains("poor|hidden")
    ]
    comparisons = {}
    comparison_methods = sorted(
        set(effect_valid["method"])
        - {"CWFM-Revised", "CWFM-Compiled"}
    )
    for baseline in ["CWFM-Compiled", *comparison_methods]:
        difference = _bootstrap_difference(
            effect_valid, "CWFM-Revised", baseline
        )
        comparisons[baseline] = {
            "mean_paired_mae_difference": difference[0],
            "ci95": [difference[1], difference[2]],
        }

    router_rows = diagnostic_frame[
        diagnostic_frame["task"].isin(
            ["static_ate", "network_interference"]
        )
    ]
    router_summary: dict[str, Any] = {}
    for scenario, group in router_rows.groupby("scenario"):
        weights = np.asarray(
            [json.loads(value) for value in group["estimator_weights"]]
        )
        estimates = np.asarray(
            [json.loads(value) for value in group["expert_estimates"]]
        )
        truth = np.asarray(
            [
                generate_episode(int(row.seed), TASK_ATE if row.task == "static_ate" else TASK_INTERFERENCE, row.scenario.split("_", 1)[1]).target
                for row in group.itertuples()
            ]
        )
        router_summary[scenario] = {
            "mean_weights": {
                name: float(weights[:, index].mean())
                for index, name in enumerate(EXPERT_NAMES)
            },
            "expert_mae": {
                name: float(np.abs(estimates[:, index] - truth).mean())
                for index, name in enumerate(EXPERT_NAMES)
            },
        }

    manifest = {
        "status": "completed",
        "architecture": "risk-routed compiled anchor plus gated residual",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _hash(checkpoint),
        "source_sha256": {
            str(path): _hash(path)
            for path in [
                Path("cwfm/model.py"),
                Path("cwfm/data.py"),
                Path("cwfm/estimators.py"),
                Path("cwfm/baselines.py"),
                Path("cwfm/train.py"),
                Path("cwfm/experiment.py"),
                Path("requirements.txt"),
            ]
        },
        "checkpoint_training": {
            key: saved[key]
            for key in [
                "seed",
                "steps",
                "batch_size",
                "best_step",
                "runtime_seconds",
                "torch_version",
            ]
        },
        "evaluation_seed_start": seed_start,
        "seeds_per_scenario": seeds,
        "scenarios": [
            f"{TASK_NAMES[task]}:{scenario}"
            for task, scenario in EVALUATION_SCENARIOS
        ],
        "calibration_seed_start": seed_start - 1_000_000,
        "calibration_episodes": calibration_episodes,
        "calibration": calibration,
        "paired_comparisons": comparisons,
        "router_by_scenario": router_summary,
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the revised CWFM against specialists"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/cwfm/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/cwfm/results")
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--calibration-episodes", type=int, default=240)
    parser.add_argument("--seed-start", type=int, default=20_000_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run(
        args.checkpoint,
        args.output_dir,
        args.seeds,
        args.calibration_episodes,
        args.seed_start,
        args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
