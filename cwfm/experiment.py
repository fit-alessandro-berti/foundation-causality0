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

from .baselines import all_baselines, forest_effect, linear_effect
from .data import (
    EVALUATION_SCENARIOS,
    TASK_ATE,
    TASK_INTERFERENCE,
    TASK_REGIME,
    collate_episodes,
    generate_episode,
)
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


def _predict_model(model: CWFM, episodes: list, device: torch.device) -> list[dict[str, float | int]]:
    batch = collate_episodes(episodes, model.config.max_variables)
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model.eval()
    with torch.no_grad():
        output = model(tensor_batch)
    raw_scale = tensor_batch["raw_scale"]
    means = output["effect_mean"] * raw_scale
    scales = output["effect_scale"] * raw_scale
    id_probability = output["identification_logit"].sigmoid()
    support_probability = output["support_logit"].sigmoid()
    flexible = output["flexible_route_logit"].sigmoid()
    structure = output["structure_logits"].argmax(-1)
    result = []
    for i in range(len(episodes)):
        result.append(
            {
                "direct": float(means[i].cpu()),
                "raw_scale": float(scales[i].cpu()),
                "id_probability": float(id_probability[i].cpu()),
                "support_probability": float(support_probability[i].cpu()),
                "flexible_probability": float(flexible[i].cpu()),
                "structure": int(structure[i].cpu()),
                "world_entropy": float(
                    -(output["world_weights"][i] * output["world_weights"][i].clamp_min(1e-9).log()).sum().cpu()
                ),
            }
        )
    return result


def _point_predictions(model: CWFM, episodes: list, device: torch.device) -> list[dict[str, Any]]:
    neural = _predict_model(model, episodes, device)
    results = []
    for ep, pred in zip(episodes, neural):
        baselines = all_baselines(ep)
        if ep.task == TASK_REGIME:
            results.append({"neural": pred, "baselines": baselines})
            continue
        linear = float(baselines["Linear g-computation"])
        flexible = float(
            baselines.get("AIPW", baselines["Random forest g-computation"])
            if ep.task == TASK_ATE
            else baselines["Random forest g-computation"]
        )
        compiled = (1.0 - pred["flexible_probability"]) * linear + pred[
            "flexible_probability"
        ] * flexible
        hybrid = 0.5 * pred["direct"] + 0.5 * compiled
        results.append(
            {
                "neural": pred,
                "baselines": baselines,
                "compiled": compiled,
                "hybrid": hybrid,
            }
        )
    return results


def _calibration(
    model: CWFM, device: torch.device, count: int, seed_start: int
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = {}
    chunk = 24
    for start in range(0, count, chunk):
        episodes = [
            generate_episode(seed_start + index, index % 3)
            for index in range(start, min(start + chunk, count))
        ]
        predictions = _point_predictions(model, episodes, device)
        for ep, values in zip(episodes, predictions):
            if ep.task == TASK_REGIME or not (ep.identified and ep.supported):
                continue
            task = TASK_NAMES[ep.task]
            bucket = buckets.setdefault(task, {})
            estimates = {
                "CWFM-Calibrated": float(values["hybrid"]),
                **{
                    str(name): float(value)
                    for name, value in values["baselines"].items()
                },
            }
            for method, estimate in estimates.items():
                bucket.setdefault(method, []).append(abs(estimate - ep.target))
    return {
        task: {
            method: float(np.quantile(errors, 0.90))
            for method, errors in methods.items()
        }
        for task, methods in buckets.items()
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
            for _ in range(2000)
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
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        ("cpu" if device_name == "auto" else device_name)
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
            generate_episode(seed_start + task * 100_000 + index, task, scenario)
            for index in range(seeds)
        ]
        predictions = _point_predictions(model, episodes, device)
        for ep, values in zip(episodes, predictions):
            neural = values["neural"]
            gate = neural["id_probability"] >= 0.5 and neural["support_probability"] >= 0.5
            diagnostics.append(
                {
                    "scenario": ep.scenario,
                    "seed": ep.seed,
                    "task": TASK_NAMES[ep.task],
                    "identified_truth": ep.identified,
                    "supported_truth": ep.supported,
                    "id_probability": neural["id_probability"],
                    "support_probability": neural["support_probability"],
                    "flexible_probability": neural["flexible_probability"],
                    "world_entropy": neural["world_entropy"],
                }
            )
            if ep.task == TASK_REGIME:
                truth = ep.structure_target if ep.structure_target < len(ep.roles) else model.config.max_variables
                method_predictions = {
                    "CWFM-Direct": int(neural["structure"]),
                    "CWFM-Calibrated": int(neural["structure"]),
                    **{name: int(value) for name, value in values["baselines"].items()},
                }
                for method, prediction in method_predictions.items():
                    records.append(
                        {
                            "scenario": ep.scenario,
                            "seed": ep.seed,
                            "task": TASK_NAMES[ep.task],
                            "method": method,
                            "truth": truth,
                            "estimate": prediction,
                            "answered": True,
                            "correct": float(prediction == truth),
                            "split_truth": float(truth != model.config.max_variables),
                            "split_predicted": float(prediction != model.config.max_variables),
                            "absolute_error": np.nan,
                            "lower": np.nan,
                            "upper": np.nan,
                            "covered": np.nan,
                        }
                    )
                continue
            method_predictions = {
                "CWFM-Direct": float(neural["direct"]),
                "CWFM-Calibrated": float(values["hybrid"]),
                **{
                    str(name): float(value)
                    for name, value in values["baselines"].items()
                },
            }
            task_calibration = calibration[TASK_NAMES[ep.task]]
            for method, estimate in method_predictions.items():
                if method == "CWFM-Direct":
                    half_width = 1.645 * neural["raw_scale"]
                    answered = True
                else:
                    calibration_name = "CWFM-Calibrated" if method == "CWFM-Calibrated" else method
                    half_width = task_calibration[calibration_name]
                    answered = bool(gate) if method == "CWFM-Calibrated" else True
                records.append(
                    {
                        "scenario": ep.scenario,
                        "seed": ep.seed,
                        "task": TASK_NAMES[ep.task],
                        "method": method,
                        "truth": ep.target,
                        "estimate": estimate if answered else np.nan,
                        "latent_estimate": estimate,
                        "answered": answered,
                        "correct": np.nan,
                        "split_truth": np.nan,
                        "split_predicted": np.nan,
                        "absolute_error": abs(estimate - ep.target) if answered else np.nan,
                        "lower": estimate - half_width if answered else np.nan,
                        "upper": estimate + half_width if answered else np.nan,
                        "covered": (
                            float(estimate - half_width <= ep.target <= estimate + half_width)
                            if answered
                            else np.nan
                        ),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(records)
    per_seed.to_csv(output_dir / "per_seed.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)
    numeric = [
        "absolute_error",
        "covered",
        "correct",
        "answered",
        "split_predicted",
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
    for baseline in [
        "Linear g-computation",
        "Random forest g-computation",
        "AIPW",
    ]:
        difference = _bootstrap_difference(effect_valid, "CWFM-Calibrated", baseline)
        comparisons[baseline] = {
            "mean_paired_mae_difference": difference[0],
            "ci95": [difference[1], difference[2]],
        }
    manifest = {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _hash(checkpoint),
        "source_sha256": {
            str(path): _hash(path)
            for path in [
                Path("cwfm/model.py"),
                Path("cwfm/data.py"),
                Path("cwfm/baselines.py"),
                Path("cwfm/train.py"),
                Path("cwfm/experiment.py"),
                Path("requirements.txt"),
            ]
        },
        "checkpoint_training": {
            key: saved[key]
            for key in ["seed", "steps", "batch_size", "runtime_seconds", "torch_version"]
        },
        "evaluation_seed_start": seed_start,
        "seeds_per_scenario": seeds,
        "scenarios": [f"{TASK_NAMES[task]}:{scenario}" for task, scenario in EVALUATION_SCENARIOS],
        "calibration_episodes": calibration_episodes,
        "calibration": calibration,
        "paired_comparisons": comparisons,
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CWFM against classical baselines")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cwfm/checkpoint.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cwfm/results"))
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
