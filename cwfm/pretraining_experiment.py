"""Small matched ablation of the released router, without retraining.

Run ``python -m cwfm.pretraining_experiment`` from the repository root.
The protocol reuses the previous confirmatory bank explicitly as a post-review
analysis. Original artifacts and the public inference implementation are read
only. Targets are used for scoring after inference, never as model inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from .data import (
    OOD_DEVELOPMENT_SCENARIOS,
    TASK_REGIME,
    collate_observed_inputs,
    generate_episode,
)
from .estimators import EXPERT_NAMES
from .model import CWFM, CWFMConfig


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _group(task: int, mask: np.ndarray) -> str:
    return f"{task}:" + "".join(str(int(v)) for v in mask)


def _export_development(checkpoint: dict, output: Path) -> pd.DataFrame:
    """Preserve historical estimates from before the estimator-layer repair."""
    source = Path("artifacts/cwfm/validation_episode_diagnostics.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [r for r in rows if r["step"] == checkpoint["best_step"]
            and r["split"] == "validation_ood"]
    counts: Counter = Counter()
    exported = []
    for row in rows:
        index = next(i for i, (_, name) in enumerate(OOD_DEVELOPMENT_SCENARIOS)
                     if row["scenario"].endswith(name))
        ordinal = counts[index]
        counts[index] += 1
        result = dict(row)
        result["episode_number"] = ordinal + 1
        result["seed"] = int(checkpoint["seed"] + 11_000_000 + index * 10_000 + ordinal)
        result["seed_provenance"] = "reconstructed from saved stream and validate generation order"
        if "final" in row:
            result["absolute_error"] = abs(row["final"] - row["target"])
        else:
            result["structure_correct"] = row["structure"] == row["target"]
        for key, value in list(result.items()):
            if isinstance(value, list):
                result[key] = json.dumps(value)
        exported.append(result)
    if len(counts) != len(OOD_DEVELOPMENT_SCENARIOS) or set(counts.values()) != {4}:
        raise RuntimeError(f"Unexpected development bank: {counts}")
    frame = pd.DataFrame(exported)
    frame.to_csv(output / "development_episodes.csv", index=False)
    effects = frame[frame["absolute_error"].notna()]
    effects.pivot(index="scenario", columns="episode_number", values="absolute_error").to_csv(
        output / "development_effect_errors.csv"
    )
    return frame


def run(protocol_path: Path, output: Path, device_name: str, batch_size: int) -> None:
    started = time.perf_counter()
    protocol = json.loads(protocol_path.read_text())
    checkpoint_path = Path(protocol["checkpoint"])
    if _sha(checkpoint_path) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint does not match the prespecified protocol")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = CWFMConfig(**checkpoint["config"])
    torch.set_num_threads(4)
    device = torch.device(device_name)
    models = {}
    trained = CWFM(config).to(device).eval()
    trained.load_state_dict(checkpoint["state_dict"])
    models["Pretrained"] = trained
    for seed in protocol["initialization_seeds"]:
        torch.manual_seed(seed)
        models[f"Untrained seed {seed}"] = CWFM(config).to(device).eval()
    initial_states = {name: {key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()}
                      for name, model in models.items()}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings = {"generation_and_fitting_seconds": 0.0,
               "transfer_seconds": 0.0,
               "forward_seconds": {name: 0.0 for name in models}}
    tuning_weights: dict[str, list[np.ndarray]] = {}
    records = []
    weight_records = []
    max_residual = {name: 0.0 for name in models}
    for stream in ("tuning", "test"):
        count = protocol[f"{stream}_episodes_per_scenario"]
        specs = [(task, scenario, protocol[f"{stream}_seed_start"] + scenario_index
                  * protocol["scenario_seed_stride"] + offset)
                 for scenario_index, (task, scenario) in enumerate(protocol["scenarios"])
                 for offset in range(count)]
        active_models = {"Pretrained": trained} if stream == "tuning" else models
        for start in range(0, len(specs), batch_size):
            tick = time.perf_counter()
            selected = specs[start:start + batch_size]
            episodes = [generate_episode(seed, task, scenario) for task, scenario, seed in selected]
            if any(ep.task == TASK_REGIME or not ep.identified or not ep.supported for ep in episodes):
                raise RuntimeError("The matched effect bank must satisfy the declared gate")
            batch = collate_observed_inputs(episodes, config.max_variables)
            timings["generation_and_fitting_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            masks = batch["expert_mask"].numpy()
            raw_scale = batch["raw_scale"].numpy()
            candidate_effects = batch["expert_estimates"].numpy() * raw_scale[:, None]
            device_batch = {key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in batch.items()}
            synchronize()
            timings["transfer_seconds"] += time.perf_counter() - tick
            for name, model in active_models.items():
                tick = time.perf_counter()
                with torch.inference_mode():
                    prediction = model(device_batch)
                synchronize()
                timings["forward_seconds"][name] += time.perf_counter() - tick
                weights = prediction["estimator_weights"].cpu().numpy()
                estimates = prediction["effect_mean"].cpu().numpy() * raw_scale
                residual = prediction["correction"].cpu().numpy() * raw_scale
                max_residual[name] = max(max_residual[name], float(np.max(np.abs(residual))))
                if not np.allclose(weights.sum(1), 1, atol=1e-6) or np.any(weights[~masks]):
                    raise RuntimeError("Invalid compatibility weights")
                compiled = prediction["compiled_mean"].cpu().numpy() * raw_scale
                if not np.allclose(estimates, compiled, atol=1e-6, rtol=1e-6):
                    raise RuntimeError("A residual pathway confounds the routing ablation")
                for i, (ep, (_, _, seed)) in enumerate(zip(episodes, selected)):
                    group = _group(ep.task, masks[i])
                    if stream == "tuning":
                        tuning_weights.setdefault(group, []).append(weights[i].copy())
                        continue
                    records.append({"scenario": ep.scenario, "task": ep.task,
                                    "seed": seed, "method": name, "truth": ep.target,
                                    "estimate": float(estimates[i]), "answered": True,
                                    "absolute_error": abs(float(estimates[i]) - ep.target)})
                    w = weights[i]
                    errors = np.abs(candidate_effects[i] - ep.target)
                    oracle = np.flatnonzero(masks[i])[np.argmin(errors[masks[i]])]
                    row = {"scenario": ep.scenario, "task": ep.task, "seed": seed,
                           "method": name, "group": group,
                           "normalized_entropy": float(-np.sum(w[masks[i]] * np.log(w[masks[i]].clip(1e-12))) / np.log(masks[i].sum())),
                           "oracle_top_choice": int(np.argmax(w) == oracle)}
                    for j, candidate in enumerate(EXPERT_NAMES):
                        row[f"weight_{candidate}"] = float(w[j])
                        row[f"candidate_{candidate}"] = float(candidate_effects[i, j])
                        row[f"eligible_{candidate}"] = bool(masks[i, j])
                    weight_records.append(row)
            if stream == "test":
                for i, (ep, (_, _, seed)) in enumerate(zip(episodes, selected)):
                    group = _group(ep.task, masks[i])
                    fixed = np.mean(tuning_weights[group], axis=0)
                    for name, weights in (("Uniform", masks[i] / masks[i].sum()),
                                          ("Mean pretrained weights", fixed)):
                        estimate = float(np.dot(weights, candidate_effects[i]))
                        records.append({"scenario": ep.scenario, "task": ep.task,
                                        "seed": seed, "method": name, "truth": ep.target,
                                        "estimate": estimate, "answered": True,
                                        "absolute_error": abs(estimate - ep.target)})
            if start == 0 or start + batch_size >= len(specs) or (start // batch_size) % 10 == 0:
                print(f"{stream}: {min(start + batch_size, len(specs))}/{len(specs)} episodes", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_seed.csv", index=False)
    weights = pd.DataFrame(weight_records)
    weights.to_csv(output / "router_weights.csv", index=False)
    weight_columns = [f"weight_{name}" for name in EXPERT_NAMES]
    weights.groupby(["method", "scenario"])[weight_columns + ["normalized_entropy", "oracle_top_choice"]].mean().to_csv(output / "router_summary.csv")
    weights[weights.method == "Pretrained"].groupby("group")[weight_columns].std().to_csv(output / "router_weight_variation.csv")
    errors = frame.pivot(index=["scenario", "seed"], columns="method", values="absolute_error")
    random_methods = [name for name in errors.columns if name.startswith("Untrained seed")]
    errors["Untrained mean"] = errors[random_methods].mean(axis=1)
    errors.to_csv(output / "paired_errors.csv")
    summary = errors.groupby(level="scenario").mean()
    summary.loc["Pooled"] = errors.mean()
    summary.to_csv(output / "summary.csv")

    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    comparisons = {}
    for comparator in ("Untrained mean", "Uniform", "Mean pretrained weights"):
        difference = errors[comparator] - errors["Pretrained"]
        replicates = np.zeros(protocol["bootstrap"]["replicates"])
        for _, cell in difference.groupby(level="scenario"):
            values = cell.to_numpy()
            indices = rng.integers(0, len(values), size=(len(replicates), len(values)))
            replicates += values[indices].mean(axis=1) / len(protocol["scenarios"])
        comparisons[comparator] = {"comparator_minus_pretrained_mae": float(difference.mean()),
                                   "ci95": np.quantile(replicates, [0.025, 0.975]).tolist()}
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    previous = pd.read_csv("artifacts/cwfm/revision/confirmatory_per_seed.csv")
    previous = previous[previous.method == "Pretrained soft router"]
    matched = frame[frame.method == "Pretrained"].merge(previous, on=["scenario", "seed"], suffixes=("_new", "_old"))
    max_difference = float((matched.estimate_new - matched.estimate_old).abs().max())
    if len(matched) != 1800 or max_difference > 1e-5:
        raise RuntimeError(f"Pretrained baseline does not reproduce existing results: {len(matched)}, {max_difference}")
    for name, model in models.items():
        if any(not torch.equal(value.detach().cpu(), initial_states[name][key])
               for key, value in model.state_dict().items()):
            raise RuntimeError(f"Model state changed during inference: {name}")
    development = _export_development(checkpoint, output)
    acceptance = previous.groupby("scenario").answered.agg(["sum", "count"])
    acceptance.to_csv(output / "acceptance.csv")
    sources = [protocol_path, checkpoint_path, Path(protocol["source_protocol"]),
               Path(__file__), Path("cwfm/model.py"), Path("cwfm/data.py"),
               Path("cwfm/estimators.py"), Path("cwfm/train.py"),
               Path("artifacts/cwfm/validation_episode_diagnostics.jsonl"),
               Path("artifacts/cwfm/revision/confirmatory_per_seed.csv")]
    manifest = {"protocol": protocol, "source_sha256": {str(p): _sha(p) for p in sources},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "device": str(device), "torch_threads": torch.get_num_threads(),
                                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                "model": {"parameters": sum(p.numel() for p in trained.parameters()),
                          "float32_parameter_bytes": sum(p.numel() * p.element_size() for p in trained.parameters()),
                          "checkpoint_bytes": checkpoint_path.stat().st_size,
                          "original_training_seconds": checkpoint["runtime_seconds"],
                          "original_training_steps": checkpoint["steps"],
                          "original_batch_size": checkpoint["batch_size"],
                          "selected_step": checkpoint["best_step"],
                          "original_gpu_model": "not recorded in the original manifest"},
                "checks": {"original_prediction_max_difference": max_difference,
                           "max_absolute_residual_by_model": max_residual,
                           "model_states_unchanged": True, "development_records": len(development),
                           "effect_development_records": int(development.absolute_error.notna().sum()),
                           "all_test_methods_accept_all_episodes": bool(frame.answered.all())},
                "tuning_mean_weights": {key: np.mean(value, axis=0).tolist() for key, value in tuning_weights.items()},
                "timings": timings, "total_runtime_seconds": time.perf_counter() - started,
                "artifact_sha256": {p.name: _sha(p) for p in sorted(output.glob("*.csv"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.round(6).to_string(), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"Completed in {manifest['total_runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/pretraining_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cwfm/pretraining_revision"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.protocol, args.output, args.device, args.batch_size)
