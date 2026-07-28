from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from .common import (
    EPS,
    Timer,
    breakpoint_scores,
    normalized_joint_loss,
    partition_scores,
    ridge_fit,
    ridge_predict,
    safe_div,
    save_dataset,
    segment_labels,
    standardize_apply,
    standardize_fit,
    summarize_records,
)


METHOD = "temporal_split_detection"
SCENARIOS = [
    "complete_null",
    "one_coefficient_break",
    "multiple_coefficient_breaks",
    "weak_coefficient_break",
    "sparse_coefficient_break",
    "intercept_only_break",
    "variance_only_break",
    "feature_shift_only",
    "global_nonlinear_correct",
    "misspecified_local_model",
    "strong_temporal_dependence",
    "short_segment",
    "gradual_drift",
]


@dataclass
class TemporalPair:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]


def generate_pair(scenario: str, seed: int, length: int = 600) -> TemporalPair:
    rng = np.random.default_rng(seed + 300_007)
    n_features, n_outcomes = 8, 3
    midpoint = length // 2
    first_third, second_third = length // 3, 2 * length // 3
    short_break = max(30, length // 8)
    transition_start, transition_end = midpoint - length // 12, midpoint + length // 12
    if scenario == "multiple_coefficient_breaks":
        breaks_all = breaks_mean = breaks_slope = [first_third, second_third]
    elif scenario in {
        "one_coefficient_break",
        "weak_coefficient_break",
        "sparse_coefficient_break",
        "strong_temporal_dependence",
    }:
        breaks_all = breaks_mean = breaks_slope = [midpoint]
    elif scenario == "short_segment":
        breaks_all = breaks_mean = breaks_slope = [short_break]
    elif scenario == "intercept_only_break":
        breaks_all = breaks_mean = [midpoint]
        breaks_slope = []
    elif scenario == "variance_only_break":
        breaks_all, breaks_mean, breaks_slope = [midpoint], [], []
    elif scenario == "gradual_drift":
        breaks_all = breaks_mean = breaks_slope = [midpoint]
    else:
        breaks_all = breaks_mean = breaks_slope = []
    parameter_breaks = sorted(set(breaks_all))
    segment_count = len(parameter_breaks) + 1
    base_b = rng.normal(scale=0.35, size=(n_features, n_outcomes))
    base_b[0, 0] += 0.8
    coefficients = np.repeat(base_b[None, :, :], segment_count, axis=0)
    direction = rng.normal(size=base_b.shape)
    direction /= np.linalg.norm(direction)
    delta = 0.32 if scenario == "weak_coefficient_break" else 1.15
    if scenario == "sparse_coefficient_break":
        direction[:] = 0
        direction[0, 0], direction[2, 1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    if breaks_slope and scenario != "gradual_drift":
        for segment in range(1, segment_count):
            coefficients[segment] = coefficients[segment - 1] + delta * direction * (-1) ** (segment + 1)
    intercepts = np.zeros((segment_count, n_outcomes))
    if scenario == "intercept_only_break":
        intercepts[1] = np.asarray([1.2, -0.8, 0.5])
    base_cov = np.fromfunction(
        lambda i, j: 0.35 ** np.abs(i - j), (n_outcomes, n_outcomes)
    ) * 0.65**2
    noise_covariances = np.repeat(base_cov[None, :, :], segment_count, axis=0)
    if scenario == "variance_only_break":
        noise_covariances[1] *= 4.0
    phi = 0.8 if scenario == "strong_temporal_dependence" else 0.35
    residual_phi = 0.45 if scenario == "strong_temporal_dependence" else 0.0
    nonlinear = scenario in {"global_nonlinear_correct", "misspecified_local_model"}

    def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.zeros((length + 100, n_features))
        feature_break = midpoint if scenario == "feature_shift_only" else None
        for t in range(1, len(x)):
            report_t = t - 100
            shift = 0.75 if feature_break is not None and report_t >= feature_break else 0.0
            x[t] = phi * x[t - 1] + shift * (1 - phi) + rng.normal(
                scale=np.sqrt(1 - phi**2), size=n_features
            )
        x = x[100:]
        y = np.empty((length, n_outcomes))
        residual_previous = np.zeros(n_outcomes)
        true_segment = segment_labels(length, parameter_breaks)
        for t in range(length):
            segment = true_segment[t]
            if nonlinear:
                mean = np.asarray(
                    [
                        x[t, 0] ** 2,
                        0.7 * x[t, 0] ** 2 - 0.4 * x[t, 1],
                        -0.5 * x[t, 0] ** 2 + 0.3 * x[t, 2],
                    ]
                )
            elif scenario == "gradual_drift":
                progress = np.clip(
                    (t - transition_start) / max(transition_end - transition_start, 1),
                    0,
                    1,
                )
                mean = x[t] @ (base_b + progress * delta * direction)
            else:
                mean = intercepts[segment] + x[t] @ coefficients[segment]
            innovation = rng.multivariate_normal(np.zeros(n_outcomes), noise_covariances[segment])
            residual = residual_phi * residual_previous + innovation
            y[t] = mean + residual
            residual_previous = residual
        return x, y, true_segment

    xd, yd, sd = sample()
    xe, ye, se = sample()
    metadata = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "feature_names": [f"X{j + 1}" for j in range(n_features)],
        "outcome_names": [f"Y{j + 1}" for j in range(n_outcomes)],
        "breakpoint_convention": "break after one-indexed row tau; arrays split at Python index tau",
        "correct_nonlinear_basis": scenario == "global_nonlinear_correct",
        "transition_window": [transition_start, transition_end] if scenario == "gradual_drift" else None,
        "x_process": {
            "phi": phi,
            "feature_distribution_break": midpoint if scenario == "feature_shift_only" else None,
        },
    }
    truth = {
        "true_breakpoints_all": np.asarray(breaks_all, dtype=int),
        "true_breakpoints_mean": np.asarray(breaks_mean, dtype=int),
        "true_breakpoints_slope": np.asarray(breaks_slope, dtype=int),
        "segment_discovery": sd,
        "segment_evaluation": se,
        "intercepts": intercepts,
        "coefficients": coefficients,
        "noise_covariances": noise_covariances,
    }
    observed_d = {"X": xd, "Y": yd, "timestamps": np.arange(1, length + 1)}
    observed_e = {"X": xe, "Y": ye, "timestamps": np.arange(1, length + 1)}
    return TemporalPair(observed_d, observed_e, truth, metadata)


def _design(x: np.ndarray, nonlinear_basis: bool) -> np.ndarray:
    return np.column_stack([x, x[:, 0] ** 2]) if nonlinear_basis else x


def _segment_fit(
    x: np.ndarray, y: np.ndarray, start: int, end: int, alpha: float = 0.1
) -> tuple[np.ndarray, float]:
    coef = ridge_fit(x[start:end], y[start:end], alpha=alpha)
    residual = y[start:end] - ridge_predict(x[start:end], coef)
    return coef, float(np.sum(residual**2))


def _best_break(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    min_segment: int,
    scan_step: int,
) -> dict[str, float | int | None]:
    _, pooled_sse = _segment_fit(x, y, start, end)
    n = end - start
    extra_df = (x.shape[1] + 1) * y.shape[1]
    best: dict[str, float | int | None] = {
        "breakpoint": None,
        "score": -np.inf,
        "statistic": -np.inf,
    }
    candidates = list(range(start + min_segment, end - min_segment + 1, scan_step))
    if candidates and candidates[-1] != end - min_segment:
        candidates.append(end - min_segment)
    for breakpoint in candidates:
        _, left_sse = _segment_fit(x, y, start, breakpoint)
        _, right_sse = _segment_fit(x, y, breakpoint, end)
        split_sse = left_sse + right_sse
        statistic = n * y.shape[1] * np.log(max(pooled_sse, EPS) / max(split_sse, EPS))
        score = statistic - extra_df * np.log(n)
        if score > best["score"]:
            best = {
                "breakpoint": int(breakpoint),
                "score": float(score),
                "statistic": float(statistic),
            }
    return best


def _binary_segmentation(
    x: np.ndarray,
    y: np.ndarray,
    critical_score: float,
    min_segment: int,
    scan_step: int,
    max_breaks: int = 4,
) -> list[int]:
    breaks: list[int] = []

    def recurse(start: int, end: int) -> None:
        if len(breaks) >= max_breaks or end - start < 2 * min_segment:
            return
        best = _best_break(x, y, start, end, min_segment, scan_step)
        if best["breakpoint"] is None or float(best["score"]) <= critical_score:
            return
        breakpoint = int(best["breakpoint"])
        breaks.append(breakpoint)
        recurse(start, breakpoint)
        recurse(breakpoint, end)

    recurse(0, len(x))
    return sorted(breaks)


def _resample_residuals(
    residuals: np.ndarray, rng: np.random.Generator, block_size: int
) -> np.ndarray:
    if block_size <= 1:
        return residuals[rng.integers(0, len(residuals), size=len(residuals))]
    starts = rng.integers(0, len(residuals) - block_size + 1, size=int(np.ceil(len(residuals) / block_size)))
    return np.concatenate([residuals[start : start + block_size] for start in starts], axis=0)[
        : len(residuals)
    ]


def _calibrate_full_search(
    x: np.ndarray,
    y: np.ndarray,
    min_segment: int,
    scan_step: int,
    rng: np.random.Generator,
    n_bootstrap: int,
    block_size: int,
) -> tuple[float, float, list[float]]:
    pooled, _ = _segment_fit(x, y, 0, len(x))
    fitted = ridge_predict(x, pooled)
    residuals = y - fitted
    observed = float(_best_break(x, y, 0, len(x), min_segment, scan_step)["score"])
    statistics = []
    for _ in range(n_bootstrap):
        boot_y = fitted + _resample_residuals(residuals, rng, block_size)
        statistics.append(
            float(_best_break(x, boot_y, 0, len(x), min_segment, scan_step)["score"])
        )
    p_value = (1 + sum(value >= observed for value in statistics)) / (n_bootstrap + 1)
    critical = max(0.0, float(np.max(statistics))) if statistics else 0.0
    return critical, float(p_value), statistics


def _classify_slope_breaks(
    x: np.ndarray, y: np.ndarray, raw_breaks: list[int]
) -> tuple[list[int], dict[int, float]]:
    boundaries = [0] + raw_breaks + [len(x)]
    retained, p_values = [], {}
    n_features = x.shape[1]
    df = n_features * y.shape[1]
    for position, breakpoint in enumerate(raw_breaks, start=1):
        left_start, right_end = boundaries[position - 1], boundaries[position + 1]
        left_end = right_start = breakpoint
        _, left_sse = _segment_fit(x, y, left_start, left_end)
        _, right_sse = _segment_fit(x, y, right_start, right_end)
        full_sse = left_sse + right_sse
        x_adjacent = x[left_start:right_end]
        y_adjacent = y[left_start:right_end]
        side = np.zeros((len(x_adjacent), 1))
        side[left_end - left_start :] = 1.0
        restricted_design = np.column_stack([side, x_adjacent])
        _, restricted_sse = _segment_fit(
            restricted_design, y_adjacent, 0, len(y_adjacent)
        )
        statistic = len(y_adjacent) * y.shape[1] * np.log(
            max(restricted_sse, EPS) / max(full_sse, EPS)
        )
        p_value = float(chi2.sf(max(statistic, 0), df))
        p_values[breakpoint] = p_value
        if p_value <= 0.05 / max(len(raw_breaks), 1):
            retained.append(breakpoint)
    return retained, p_values


def _fit_piecewise(
    x: np.ndarray, y: np.ndarray, breaks: list[int]
) -> tuple[list[np.ndarray], np.ndarray]:
    boundaries = [0] + sorted(breaks) + [len(x)]
    coefficients, predictions = [], np.empty_like(y)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        coef, _ = _segment_fit(x, y, start, end)
        coefficients.append(coef)
        predictions[start:end] = ridge_predict(x[start:end], coef)
    return coefficients, predictions


def _predict_piecewise(
    x: np.ndarray, coefficients: list[np.ndarray], breaks: list[int]
) -> np.ndarray:
    boundaries = [0] + sorted(breaks) + [len(x)]
    prediction = np.empty((len(x), coefficients[0].shape[1]))
    for coef, (start, end) in zip(coefficients, zip(boundaries[:-1], boundaries[1:])):
        prediction[start:end] = ridge_predict(x[start:end], coef)
    return prediction


def _parameter_error(
    estimated_coefficients: list[np.ndarray],
    estimated_breaks: list[int],
    true_coefficients: np.ndarray,
    true_intercepts: np.ndarray,
    true_breaks: list[int],
    length: int,
) -> tuple[float, float]:
    estimated_boundaries = [0] + estimated_breaks + [length]
    true_boundaries = [0] + true_breaks + [length]
    errors, denominator, intercept_errors = [], 0.0, []
    for coef, (start, end) in zip(
        estimated_coefficients, zip(estimated_boundaries[:-1], estimated_boundaries[1:])
    ):
        midpoint = (start + end - 1) // 2
        true_segment = int(np.searchsorted(true_boundaries[1:], midpoint, side="right"))
        truth = true_coefficients[min(true_segment, len(true_coefficients) - 1)]
        estimated_slope = coef[1 : 1 + truth.shape[0]]
        errors.append(np.linalg.norm(estimated_slope - truth))
        denominator += np.linalg.norm(truth)
        intercept_errors.append(
            np.linalg.norm(
                coef[0] - true_intercepts[min(true_segment, len(true_intercepts) - 1)]
            )
        )
    return safe_div(sum(errors), denominator + EPS), float(np.mean(intercept_errors))


def _breakpoint_set_similarity(
    first: list[int], second: list[int], tolerance: int
) -> float:
    from .common import match_breakpoints

    matched = len(match_breakpoints(first, second, tolerance).pairs)
    union = len(first) + len(second) - matched
    return matched / union if union else 1.0


def _bootstrap_break_stability(
    x: np.ndarray,
    y: np.ndarray,
    fitted: np.ndarray,
    raw_breaks: list[int],
    true_breaks: list[int],
    critical: float,
    min_segment: int,
    scan_step: int,
    block_size: int,
    rng: np.random.Generator,
    n_replicates: int,
) -> dict[str, float]:
    residuals = y - fitted
    break_sets: list[list[int]] = []
    tolerance = max(5, int(np.ceil(0.01 * len(x))))
    for _ in range(n_replicates):
        boot_y = fitted + _resample_residuals(residuals, rng, block_size)
        break_sets.append(
            _binary_segmentation(x, boot_y, critical, min_segment, scan_step)
        )
    similarities = [
        _breakpoint_set_similarity(break_sets[i], break_sets[j], tolerance)
        for i in range(len(break_sets))
        for j in range(i + 1, len(break_sets))
    ]
    detected_frequencies = [
        np.mean([any(abs(value - target) <= tolerance for value in values) for values in break_sets])
        for target in raw_breaks
    ]
    interval_widths, coverages = [], []
    for truth in true_breaks:
        locations = [
            min(values, key=lambda value: abs(value - truth))
            for values in break_sets
            if values and min(abs(value - truth) for value in values) <= tolerance
        ]
        if len(locations) >= 2:
            lower, upper = np.quantile(locations, [0.05, 0.95])
            interval_widths.append(upper - lower)
            coverages.append(lower <= truth <= upper)
    return {
        "bootstrap_break_jaccard": float(np.mean(similarities)) if similarities else np.nan,
        "bootstrap_any_break_frequency": float(np.mean([bool(values) for values in break_sets])),
        "bootstrap_mean_break_count": float(np.mean([len(values) for values in break_sets])),
        "bootstrap_detected_break_frequency": float(np.mean(detected_frequencies))
        if detected_frequencies
        else np.nan,
        "bootstrap_location_interval_width": float(np.mean(interval_widths))
        if interval_widths
        else np.nan,
        "bootstrap_location_ci_coverage": float(np.mean(coverages)) if coverages else np.nan,
    }


def evaluate_pair(pair: TemporalPair, seed: int, n_bootstrap: int = 19) -> dict[str, Any]:
    timer = Timer()
    xd_raw, yd_raw = pair.discovery["X"], pair.discovery["Y"]
    xe_raw, ye_raw = pair.evaluation["X"], pair.evaluation["Y"]
    xm, xs = standardize_fit(xd_raw)
    ym, ys = standardize_fit(yd_raw)
    xd = standardize_apply(xd_raw, xm, xs)
    xe = standardize_apply(xe_raw, xm, xs)
    yd = standardize_apply(yd_raw, ym, ys)
    ye = standardize_apply(ye_raw, ym, ys)
    nonlinear_basis = bool(pair.metadata["correct_nonlinear_basis"])
    xd_design, xe_design = _design(xd, nonlinear_basis), _design(xe, nonlinear_basis)
    min_segment = 80
    scan_step = 5
    block_size = 12 if pair.metadata["x_process"]["phi"] >= 0.8 else 1
    rng = np.random.default_rng(seed + 39019)
    critical, p_value, _ = _calibrate_full_search(
        xd_design,
        yd,
        min_segment,
        scan_step,
        rng,
        n_bootstrap,
        block_size,
    )
    if p_value <= 0.05:
        raw_breaks = _binary_segmentation(
            xd_design, yd, critical, min_segment, scan_step
        )
    else:
        raw_breaks = []
    slope_breaks, slope_p_values = _classify_slope_breaks(
        xd_design, yd, raw_breaks
    )
    estimated_coefficients, discovery_fitted = _fit_piecewise(
        xd_design, yd, raw_breaks
    )
    segmented_pred = _predict_piecewise(xe_design, estimated_coefficients, raw_breaks)
    pooled_coef, _ = _segment_fit(xd_design, yd, 0, len(xd_design))
    pooled_pred = ridge_predict(xe_design, pooled_coef)
    true_breaks_all = pair.truth["true_breakpoints_all"].tolist()
    true_breaks_slope = pair.truth["true_breakpoints_slope"].tolist()
    oracle_coefficients, _ = _fit_piecewise(xd_design, yd, true_breaks_all)
    oracle_pred = _predict_piecewise(xe_design, oracle_coefficients, true_breaks_all)
    pooled_loss = normalized_joint_loss(ye, pooled_pred, np.mean(yd, axis=0))
    segmented_loss = normalized_joint_loss(ye, segmented_pred, np.mean(yd, axis=0))
    oracle_loss = normalized_joint_loss(ye, oracle_pred, np.mean(yd, axis=0))
    tolerance = max(5, int(np.ceil(0.01 * len(xd))))
    raw_metrics = breakpoint_scores(
        true_breaks_all, raw_breaks, tolerance, len(xd), "raw_break"
    )
    slope_metrics = breakpoint_scores(
        true_breaks_slope, slope_breaks, tolerance, len(xd), "slope_break"
    )
    true_labels = segment_labels(len(xd), true_breaks_all)
    estimated_labels = segment_labels(len(xd), raw_breaks)
    partition = partition_scores(true_labels, estimated_labels)
    standardized_coefficients = (
        pair.truth["coefficients"] * xs[None, :, None] / ys[None, None, :]
    )
    standardized_intercepts = np.empty_like(pair.truth["intercepts"])
    for segment in range(len(standardized_intercepts)):
        standardized_intercepts[segment] = (
            pair.truth["intercepts"][segment]
            + xm @ pair.truth["coefficients"][segment]
            - ym
        ) / ys
    coefficient_error, intercept_error = _parameter_error(
        estimated_coefficients,
        raw_breaks,
        standardized_coefficients,
        standardized_intercepts,
        true_breaks_all,
        len(xd),
    )
    scenario = pair.metadata["scenario"]
    if nonlinear_basis or scenario in {"misspecified_local_model", "gradual_drift"}:
        coefficient_error = np.nan
        intercept_error = np.nan
    result: dict[str, Any] = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "status": "ok",
        "global_search_p": p_value,
        "critical_score": critical,
        "n_true_breaks_all": float(len(true_breaks_all)),
        "n_true_breaks_slope": float(len(true_breaks_slope)),
        "n_raw_breaks": float(len(raw_breaks)),
        "n_slope_breaks": float(len(slope_breaks)),
        "segment_ari": partition["ari"],
        "segment_nmi": partition["nmi"],
        "segment_row_accuracy": partition["row_accuracy"],
        "boundary_displacement": 1.0 - partition["row_accuracy"],
        "relative_coefficient_error": coefficient_error,
        "intercept_error": intercept_error,
        "joint_test_loss_pooled": pooled_loss,
        "joint_test_loss_segmented": segmented_loss,
        "joint_test_loss_oracle": oracle_loss,
        "oracle_improvement_recovered": safe_div(
            pooled_loss - segmented_loss, pooled_loss - oracle_loss
        ),
        "false_slope_attribution": float(
            not true_breaks_slope and bool(slope_breaks)
        ),
        "runtime_seconds": timer.seconds,
    }
    result.update(raw_metrics)
    result.update(slope_metrics)
    result.update(
        _bootstrap_break_stability(
            xd_design,
            yd,
            discovery_fitted,
            raw_breaks,
            true_breaks_all,
            critical,
            min_segment,
            scan_step,
            block_size,
            rng,
            min(5, n_bootstrap),
        )
    )
    for fraction in [0.0, 0.02, 0.05]:
        metric = breakpoint_scores(
            true_breaks_slope,
            slope_breaks,
            int(np.ceil(fraction * len(xd))),
            len(xd),
            f"slope_{int(fraction * 100)}pct",
        )
        result.update(metric)
    return result


def run(
    data_root: Path,
    metrics_root: Path,
    seeds: list[int],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in seeds:
            try:
                pair = generate_pair(scenario, seed)
                save_dataset(
                    data_root,
                    METHOD,
                    scenario,
                    seed,
                    pair.discovery,
                    pair.evaluation,
                    pair.truth,
                    pair.metadata,
                )
                records.append(evaluate_pair(pair, seed, n_bootstrap))
            except Exception as exc:
                records.append(
                    {
                        "method": METHOD,
                        "scenario": scenario,
                        "seed": seed,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            summarize_records(records, metrics_root / METHOD, METHOD)
    return records
