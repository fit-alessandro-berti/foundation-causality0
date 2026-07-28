from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
)


EPS = 1e-12


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def save_dataset(
    root: Path,
    method: str,
    scenario: str,
    seed: int,
    discovery: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    """Persist observed arrays separately from evaluator-only truth."""
    out = root / method / scenario / f"seed_{seed:04d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "discovery.npz", **discovery)
    np.savez_compressed(out / "evaluation.npz", **evaluation)
    np.savez_compressed(out / "truth.npz", **truth)
    write_json(out / "metadata.json", metadata)
    return out


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale = np.where(scale < EPS, 1.0, scale)
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - mean) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    design = add_intercept(x)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def ridge_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ coef


def sse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum((y - pred) ** 2))


def normalized_joint_loss(y: np.ndarray, pred: np.ndarray, reference_mean: np.ndarray) -> float:
    numerator = np.sum((y - pred) ** 2, axis=0)
    denominator = np.sum((y - reference_mean) ** 2, axis=0)
    return float(np.mean(numerator / np.maximum(denominator, EPS)))


def safe_div(num: float, den: float, missing: float = np.nan) -> float:
    return float(num / den) if abs(den) > EPS else float(missing)


def binary_scores(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    precision = safe_div(tp, tp + fp, 1.0 if tp + fp == 0 and tp + fn == 0 else 0.0)
    recall = safe_div(tp, tp + fn, 1.0 if tp + fn == 0 else 0.0)
    f1 = safe_div(2 * precision * recall, precision + recall, 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": safe_div(fp, tp + fp, 0.0),
        "fnr": safe_div(fn, tp + fn, 0.0),
        "specificity": safe_div(tn, tn + fp, 1.0),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def set_scores(true_items: Iterable[Any], estimated_items: Iterable[Any]) -> dict[str, float]:
    true_set, est_set = set(true_items), set(estimated_items)
    universe = sorted(true_set | est_set, key=str)
    yt = [int(item in true_set) for item in universe]
    yp = [int(item in est_set) for item in universe]
    if not universe:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "fdr": 0.0,
            "fnr": 0.0,
            "specificity": 1.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
    return binary_scores(yt, yp)


def optimal_label_accuracy(true_labels: np.ndarray, estimated_labels: np.ndarray) -> float:
    true_values = np.unique(true_labels)
    est_values = np.unique(estimated_labels)
    overlap = np.zeros((len(true_values), len(est_values)), dtype=int)
    for i, tv in enumerate(true_values):
        for j, ev in enumerate(est_values):
            overlap[i, j] = int(np.sum((true_labels == tv) & (estimated_labels == ev)))
    rows, cols = linear_sum_assignment(-overlap)
    return float(overlap[rows, cols].sum() / max(len(true_labels), 1))


def partition_scores(true_labels: np.ndarray, estimated_labels: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(true_labels, estimated_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, estimated_labels)),
        "row_accuracy": optimal_label_accuracy(true_labels, estimated_labels),
    }


def match_components(weight: np.ndarray) -> list[tuple[int, int]]:
    if weight.size == 0:
        return []
    rows, cols = linear_sum_assignment(-weight)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.empty((a.shape[1], b.shape[1]))
    az = standardize_apply(a, *standardize_fit(a))
    bz = standardize_apply(b, *standardize_fit(b))
    return az.T @ bz / max(len(a) - 1, 1)


def edge_vectors(matrix: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    idx = [(i, j) for i in range(matrix.shape[0]) for j in range(i + 1, matrix.shape[0])]
    return np.asarray([matrix[i, j] for i, j in idx]), idx


def graph_scores(true_partial: np.ndarray, estimated_partial: np.ndarray, threshold: float) -> dict[str, float]:
    true_values, _ = edge_vectors(true_partial)
    est_values, _ = edge_vectors(estimated_partial)
    yt = (np.abs(true_values) > EPS).astype(int)
    yp = (np.abs(est_values) >= threshold).astype(int)
    scores = binary_scores(yt, yp)
    scores.update(
        {
            "mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) > 1 else float(np.mean(yt == yp)),
            "shd": float(np.sum(yt != yp)),
            "normalized_shd": float(np.mean(yt != yp)) if len(yt) else 0.0,
            "auprc": float(average_precision_score(yt, np.abs(est_values))) if np.any(yt) else float(np.mean(yp == 0)),
            "partial_corr_rmse": float(np.sqrt(np.mean((est_values - true_values) ** 2))) if len(yt) else 0.0,
            "exact_graph": float(np.array_equal(yt, yp)),
            "any_false_edge": float(np.any((yt == 0) & (yp == 1))),
            "n_false_edges": float(np.sum((yt == 0) & (yp == 1))),
        }
    )
    true_edge = yt == 1
    detected_true = true_edge & (yp == 1)
    scores["edge_rmse"] = (
        float(np.sqrt(np.mean((est_values[true_edge] - true_values[true_edge]) ** 2)))
        if np.any(true_edge)
        else np.nan
    )
    scores["nonedge_rmse"] = (
        float(np.sqrt(np.mean(est_values[~true_edge] ** 2))) if np.any(~true_edge) else np.nan
    )
    scores["detected_edge_sign_accuracy"] = (
        float(np.mean(np.sign(est_values[detected_true]) == np.sign(true_values[detected_true])))
        if np.any(detected_true)
        else np.nan
    )
    return scores


@dataclass
class BreakpointMatch:
    pairs: list[tuple[int, int]]
    unmatched_true: list[int]
    unmatched_estimated: list[int]


def match_breakpoints(
    true_breaks: Iterable[int], estimated_breaks: Iterable[int], tolerance: int
) -> BreakpointMatch:
    true_values = sorted(int(v) for v in true_breaks)
    est_values = sorted(int(v) for v in estimated_breaks)
    if not true_values or not est_values:
        return BreakpointMatch([], true_values, est_values)
    large = 10**9
    cost = np.full((len(true_values), len(est_values)), large, dtype=float)
    for i, true_value in enumerate(true_values):
        for j, est_value in enumerate(est_values):
            distance = abs(true_value - est_value)
            if distance <= tolerance:
                cost[i, j] = distance
    rows, cols = linear_sum_assignment(cost)
    pairs = [
        (true_values[i], est_values[j])
        for i, j in zip(rows, cols)
        if cost[i, j] < large
    ]
    matched_true = {v for v, _ in pairs}
    matched_est = {v for _, v in pairs}
    return BreakpointMatch(
        pairs,
        [v for v in true_values if v not in matched_true],
        [v for v in est_values if v not in matched_est],
    )


def breakpoint_scores(
    true_breaks: Iterable[int],
    estimated_breaks: Iterable[int],
    tolerance: int,
    length: int,
    prefix: str = "",
) -> dict[str, float]:
    true_values = list(true_breaks)
    est_values = list(estimated_breaks)
    matching = match_breakpoints(true_values, est_values, tolerance)
    tp = len(matching.pairs)
    fp = len(matching.unmatched_estimated)
    fn = len(matching.unmatched_true)
    precision = safe_div(tp, tp + fp, 1.0 if not true_values else 0.0)
    recall = safe_div(tp, tp + fn, 1.0 if not true_values else 0.0)
    f1 = safe_div(2 * precision * recall, precision + recall, 0.0)
    distances = np.asarray([abs(a - b) for a, b in matching.pairs], dtype=float)
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}precision": precision,
        f"{key}recall": recall,
        f"{key}f1": f1,
        f"{key}fdr": safe_div(fp, tp + fp, 0.0),
        f"{key}count_error": float(abs(len(true_values) - len(est_values))),
        f"{key}exact_set": float(set(true_values) == set(est_values)),
        f"{key}localization_mae": float(np.mean(distances)) if len(distances) else np.nan,
        f"{key}localization_median": float(np.median(distances)) if len(distances) else np.nan,
        f"{key}localization_rmse": float(np.sqrt(np.mean(distances**2))) if len(distances) else np.nan,
        f"{key}localization_normalized": float(np.mean(distances) / length) if len(distances) else np.nan,
        f"{key}any_detection": float(bool(est_values)),
    }


def segment_labels(length: int, breaks: Iterable[int]) -> np.ndarray:
    boundaries = [0] + sorted(int(v) for v in breaks) + [length]
    labels = np.empty(length, dtype=int)
    for label, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[start:end] = label
    return labels


def summarize_records(records: list[dict[str, Any]], metrics_dir: Path, method: str) -> pd.DataFrame:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(metrics_dir / "per_seed.csv", index=False)
    (metrics_dir / "per_seed.jsonl").write_text(
        "".join(json.dumps(jsonable(record), sort_keys=True) + "\n" for record in records)
    )
    id_columns = {"method", "scenario", "seed", "status", "error"}
    numeric = [
        column
        for column in frame.columns
        if column not in id_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    good = frame[frame.get("status", "ok") == "ok"] if "status" in frame else frame
    rows: list[dict[str, Any]] = []
    for scenario, group in good.groupby("scenario", sort=True):
        row: dict[str, Any] = {
            "method": method,
            "scenario": scenario,
            "n_completed": int(len(group)),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}__mean"] = float(values.mean())
            row[f"{column}__std"] = float(values.std(ddof=1)) if values.count() > 1 else 0.0
            row[f"{column}__median"] = float(values.median())
            row[f"{column}__q05"] = float(values.quantile(0.05))
            row[f"{column}__q95"] = float(values.quantile(0.95))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(metrics_dir / "summary.csv", index=False)
    write_json(metrics_dir / "summary.json", rows)
    write_json(
        metrics_dir / "manifest.json",
        {
            "method": method,
            "n_records": len(records),
            "n_success": int(sum(record.get("status") == "ok" for record in records)),
            "n_failed": int(sum(record.get("status") != "ok" for record in records)),
            "scenarios": sorted(frame["scenario"].unique().tolist()) if len(frame) else [],
            "metric_fields": numeric,
        },
    )
    return summary


class Timer:
    def __init__(self) -> None:
        self.start = time.monotonic()

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.start


def process_memory_mb() -> float:
    try:
        import resource

        scale = 1024.0 if os.uname().sysname != "Darwin" else 1024.0 * 1024.0
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale)
    except Exception:
        return np.nan
