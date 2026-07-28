from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from .common import (
    EPS,
    Timer,
    binary_scores,
    breakpoint_scores,
    normalized_joint_loss,
    partition_scores,
    ridge_fit,
    ridge_predict,
    safe_div,
    save_dataset,
    segment_labels,
    set_scores,
    standardize_apply,
    standardize_fit,
    summarize_records,
)
from .temporal_split import (
    _best_break,
    _binary_segmentation,
    _calibrate_full_search,
    _fit_piecewise,
    _predict_piecewise,
    _segment_fit,
    _breakpoint_set_similarity,
    _resample_residuals,
)


METHOD = "temporal_split_detection_method"
SCENARIOS = [
    "complete_null",
    "one_B_break",
    "several_B_breaks",
    "weak_B_break",
    "A_only_break",
    "intercept_only_break",
    "noise_only_break",
    "X_process_break",
    "mixed_A_B_break",
    "B_zero_to_nonzero",
    "B_nonzero_to_zero",
    "high_persistence",
    "high_dimensional",
    "wrong_lag_order",
    "contemporaneous_only_relation",
    "close_breakpoints",
]


@dataclass
class VarxPair:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _stable_matrix(rng: np.random.Generator, size: int, target_radius: float) -> np.ndarray:
    matrix = rng.normal(scale=0.2, size=(size, size))
    matrix[rng.random((size, size)) < 0.55] = 0.0
    radius = max(np.max(np.abs(np.linalg.eigvals(matrix))), EPS)
    return matrix * (target_radius / radius)


def generate_pair(scenario: str, seed: int, length: int = 700, burn_in: int = 180) -> VarxPair:
    rng = np.random.default_rng(seed + 400_009)
    n_features, n_outcomes = (22, 5) if scenario == "high_dimensional" else (7, 3)
    midpoint = length // 2
    first_third, second_third = length // 3, 2 * length // 3
    close_gap = max(30, length // 9)
    if scenario == "several_B_breaks":
        breaks_all = breaks_b = [first_third, second_third]
    elif scenario == "close_breakpoints":
        breaks_all = breaks_b = [midpoint - close_gap // 2, midpoint + close_gap // 2]
    elif scenario in {
        "one_B_break",
        "weak_B_break",
        "high_persistence",
        "high_dimensional",
    }:
        breaks_all = breaks_b = [midpoint]
    elif scenario == "mixed_A_B_break":
        breaks_all = breaks_b = [midpoint]
    elif scenario in {"B_zero_to_nonzero", "B_nonzero_to_zero"}:
        breaks_all = breaks_b = [midpoint]
    elif scenario == "A_only_break":
        breaks_all, breaks_b = [midpoint], []
    elif scenario == "intercept_only_break":
        breaks_all, breaks_b = [midpoint], []
    elif scenario == "noise_only_break":
        breaks_all, breaks_b = [midpoint], []
    else:
        breaks_all, breaks_b = [], []
    breaks_a = [midpoint] if scenario in {"A_only_break", "mixed_A_B_break"} else []
    breaks_c = [midpoint] if scenario == "intercept_only_break" else []
    breaks_noise = [midpoint] if scenario == "noise_only_break" else []
    parameter_breaks = sorted(set(breaks_all))
    segment_count = len(parameter_breaks) + 1
    target_radius = 0.82 if scenario == "high_persistence" else 0.42
    a_base = _stable_matrix(rng, n_outcomes, target_radius)
    a_matrices = np.repeat(a_base[None, :, :], segment_count, axis=0)
    if breaks_a:
        changed = _stable_matrix(rng, n_outcomes, min(target_radius + 0.1, 0.9))
        a_matrices[1:] = changed
    b_base = rng.normal(scale=0.28, size=(n_outcomes, n_features))
    b_base[rng.random(b_base.shape) < 0.65] = 0.0
    if not np.any(b_base):
        b_base[0, 0] = 0.5
    b_matrices = np.repeat(b_base[None, :, :], segment_count, axis=0)
    direction = rng.normal(size=b_base.shape)
    direction[rng.random(direction.shape) < 0.65] = 0.0
    direction /= max(np.linalg.norm(direction), EPS)
    jump = 0.32 if scenario == "weak_B_break" else 1.0
    if breaks_b:
        for segment in range(1, segment_count):
            b_matrices[segment] = b_matrices[segment - 1] + jump * direction * (-1) ** (segment + 1)
    if scenario == "B_zero_to_nonzero":
        b_matrices[0] = 0.0
        b_matrices[1] = b_base
    if scenario == "B_nonzero_to_zero":
        b_matrices[0] = b_base
        b_matrices[1] = 0.0
    if scenario == "contemporaneous_only_relation":
        # The response uses X_t below; the direct lagged-X coefficient is truly zero.
        b_matrices[:] = 0.0
    intercepts = np.zeros((segment_count, n_outcomes))
    if breaks_c:
        intercepts[1] = np.linspace(0.6, 1.2, n_outcomes)
    base_covariance = np.fromfunction(
        lambda i, j: 0.3 ** np.abs(i - j), (n_outcomes, n_outcomes)
    ) * 0.55**2
    noise_covariances = np.repeat(base_covariance[None, :, :], segment_count, axis=0)
    if breaks_noise:
        noise_covariances[1] *= 3.5
    feature_phi = 0.85 if scenario == "high_persistence" else 0.45
    f_matrices = np.repeat(
        (np.eye(n_features) * feature_phi)[None, :, :],
        2 if scenario == "X_process_break" else 1,
        axis=0,
    )
    feature_intercepts = np.zeros((len(f_matrices), n_features))
    if scenario == "X_process_break":
        f_matrices[1] = np.eye(n_features) * 0.7
        feature_intercepts[1] = 0.5 * (1 - 0.7)
    lag2 = np.zeros((n_outcomes, n_features))
    if scenario == "wrong_lag_order":
        lag2 = rng.normal(scale=0.25, size=lag2.shape)
        lag2[rng.random(lag2.shape) < 0.7] = 0.0

    def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = length + burn_in
        x = np.zeros((total, n_features))
        y = np.zeros((total, n_outcomes))
        segments = segment_labels(length, parameter_breaks)
        for t in range(2, total):
            reported_t = t - burn_in
            x_regime = int(
                scenario == "X_process_break" and reported_t >= midpoint
            )
            x[t] = (
                feature_intercepts[x_regime]
                + f_matrices[x_regime] @ x[t - 1]
                + rng.normal(scale=np.sqrt(max(1 - feature_phi**2, 0.15)), size=n_features)
            )
            segment = segments[max(0, min(reported_t, length - 1))] if reported_t >= 0 else 0
            if scenario == "contemporaneous_only_relation":
                mean = a_matrices[segment] @ y[t - 1] + b_base @ x[t]
            else:
                mean = (
                    intercepts[segment]
                    + a_matrices[segment] @ y[t - 1]
                    + b_matrices[segment] @ x[t - 1]
                    + lag2 @ x[t - 2]
                )
            y[t] = mean + rng.multivariate_normal(
                np.zeros(n_outcomes), noise_covariances[segment]
            )
        return x[burn_in:], y[burn_in:], segments

    xd, yd, sd = sample()
    xe, ye, se = sample()
    support = np.abs(b_matrices) > EPS
    metadata = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "feature_names": [f"X{j + 1}" for j in range(n_features)],
        "outcome_names": [f"Y{j + 1}" for j in range(n_outcomes)],
        "burn_in": burn_in,
        "lag_order_truth": 2 if scenario == "wrong_lag_order" else 1,
        "lag_order_fitted": 1,
        "breakpoint_convention": "break after one-indexed row tau; arrays split at Python index tau",
        "interpretation": "Granger_predictive_not_interventional_causal",
        "X_process_break": midpoint if scenario == "X_process_break" else None,
    }
    truth = {
        "true_breakpoints_all": np.asarray(breaks_all, dtype=int),
        "true_breakpoints_B": np.asarray(breaks_b, dtype=int),
        "true_breakpoints_A": np.asarray(breaks_a, dtype=int),
        "true_breakpoints_c": np.asarray(breaks_c, dtype=int),
        "true_breakpoints_noise": np.asarray(breaks_noise, dtype=int),
        "segment_discovery": sd,
        "segment_evaluation": se,
        "c": intercepts,
        "A": a_matrices,
        "B": b_matrices,
        "Sigma_eps": noise_covariances,
        "B_support": support,
        "F": f_matrices,
        "d": feature_intercepts,
        "lag2_B": lag2,
    }
    observed_d = {
        "X": xd,
        "Y": yd,
        "timestamps": np.arange(1, length + 1),
        "sequence_id": np.zeros(length, dtype=int),
    }
    observed_e = {
        "X": xe,
        "Y": ye,
        "timestamps": np.arange(1, length + 1),
        "sequence_id": np.zeros(length, dtype=int),
    }
    return VarxPair(observed_d, observed_e, truth, metadata)


def make_lagged(
    x: np.ndarray, y: np.ndarray, sequence_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = sequence_id[1:] == sequence_id[:-1]
    design = np.column_stack([y[:-1][valid], x[:-1][valid]])
    response = y[1:][valid]
    assert len(design) == int(np.sum(valid))
    return design, response


def _classify_b_breaks(
    design: np.ndarray,
    response: np.ndarray,
    raw_breaks_lagged: list[int],
    n_outcomes: int,
    n_features: int,
) -> tuple[list[int], dict[int, float]]:
    boundaries = [0] + raw_breaks_lagged + [len(design)]
    retained, p_values = [], {}
    df = n_features * n_outcomes
    for position, breakpoint in enumerate(raw_breaks_lagged, start=1):
        start, end = boundaries[position - 1], boundaries[position + 1]
        _, left_sse = _segment_fit(design, response, start, breakpoint)
        _, right_sse = _segment_fit(design, response, breakpoint, end)
        full_sse = left_sse + right_sse
        adjacent_y_lag = design[start:end, :n_outcomes]
        adjacent_x_lag = design[start:end, n_outcomes:]
        side = np.zeros((end - start, 1))
        side[breakpoint - start :] = 1.0
        left_a = adjacent_y_lag * (1.0 - side)
        right_a = adjacent_y_lag * side
        restricted_design = np.column_stack([side, left_a, right_a, adjacent_x_lag])
        _, restricted_sse = _segment_fit(
            restricted_design, response[start:end], 0, end - start
        )
        statistic = (end - start) * n_outcomes * np.log(
            max(restricted_sse, EPS) / max(full_sse, EPS)
        )
        p_value = float(chi2.sf(max(statistic, 0.0), df))
        original_breakpoint = breakpoint + 1
        p_values[original_breakpoint] = p_value
        if p_value <= 0.05 / max(len(raw_breaks_lagged), 1):
            retained.append(original_breakpoint)
    return retained, p_values


def _segment_relevance(
    design: np.ndarray,
    response: np.ndarray,
    breaks_lagged: list[int],
    n_outcomes: int,
) -> tuple[list[bool], list[float]]:
    boundaries = [0] + breaks_lagged + [len(design)]
    relevance, gains = [], []
    n_features = design.shape[1] - n_outcomes
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        _, full_sse = _segment_fit(design, response, start, end)
        _, restricted_sse = _segment_fit(
            design[:, :n_outcomes], response, start, end
        )
        statistic = (end - start) * n_outcomes * np.log(
            max(restricted_sse, EPS) / max(full_sse, EPS)
        )
        p_value = float(chi2.sf(max(statistic, 0), n_features * n_outcomes))
        relevance.append(p_value <= 0.05)
        gains.append(float((restricted_sse - full_sse) / max(restricted_sse, EPS)))
    return relevance, gains


def _truth_for_estimated_segments(
    breaks_original: list[int],
    true_breaks: list[int],
    b_matrices: np.ndarray,
    length: int,
) -> list[np.ndarray]:
    boundaries = [0] + breaks_original + [length]
    truths = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        midpoint = (start + end - 1) // 2
        segment = int(np.searchsorted(true_breaks, midpoint, side="right"))
        truths.append(b_matrices[min(segment, len(b_matrices) - 1)])
    return truths


def _coefficient_metrics(
    coefficients: list[np.ndarray],
    truths_b: list[np.ndarray],
    truths_a: list[np.ndarray],
    n_outcomes: int,
) -> dict[str, float]:
    b_errors, a_errors, b_denominator, a_denominator = [], [], 0.0, 0.0
    true_edges, estimated_edges, signs = [], [], []
    nonzero_errors = []
    for segment, (coef, truth_b, truth_a) in enumerate(zip(coefficients, truths_b, truths_a)):
        estimated_a = coef[1 : 1 + n_outcomes].T
        estimated_b = coef[1 + n_outcomes :].T
        a_errors.append(np.linalg.norm(estimated_a - truth_a))
        b_errors.append(np.linalg.norm(estimated_b - truth_b))
        a_denominator += np.linalg.norm(truth_a)
        b_denominator += np.linalg.norm(truth_b)
        for output, feature in zip(*np.where(np.abs(truth_b) > EPS)):
            true_edges.append((segment, int(output), int(feature)))
            nonzero_errors.append((estimated_b[output, feature] - truth_b[output, feature]) ** 2)
            if abs(estimated_b[output, feature]) >= 0.1:
                signs.append(
                    np.sign(estimated_b[output, feature]) == np.sign(truth_b[output, feature])
                )
        for output, feature in zip(*np.where(np.abs(estimated_b) >= 0.1)):
            estimated_edges.append((segment, int(output), int(feature)))
    support = set_scores(true_edges, estimated_edges)
    return {
        "relative_A_error": safe_div(sum(a_errors), a_denominator + EPS),
        "relative_B_error": safe_div(sum(b_errors), b_denominator + EPS),
        "B_support_precision": support["precision"],
        "B_support_recall": support["recall"],
        "B_support_f1": support["f1"],
        "B_support_fdr": support["fdr"],
        "B_sign_accuracy": float(np.mean(signs)) if signs else np.nan,
        "B_nonzero_rmse": float(np.sqrt(np.mean(nonzero_errors))) if nonzero_errors else np.nan,
    }


def _delta_b_error(
    coefficients: list[np.ndarray],
    raw_breaks: list[int],
    retained_b_breaks: list[int],
    true_breaks_b: list[int],
    truth_b_standardized_by_true_segment: list[np.ndarray],
    n_outcomes: int,
    tolerance: int,
) -> float:
    if not true_breaks_b:
        return np.nan
    errors, denominator = [], 0.0
    for true_position, true_break in enumerate(true_breaks_b):
        candidates = [
            (abs(estimated - true_break), estimated)
            for estimated in retained_b_breaks
            if abs(estimated - true_break) <= tolerance and estimated in raw_breaks
        ]
        if not candidates:
            continue
        _, estimated_break = min(candidates)
        boundary_position = raw_breaks.index(estimated_break)
        left_b = coefficients[boundary_position][1 + n_outcomes :].T
        right_b = coefficients[boundary_position + 1][1 + n_outcomes :].T
        estimated_delta = right_b - left_b
        true_delta = (
            truth_b_standardized_by_true_segment[true_position + 1]
            - truth_b_standardized_by_true_segment[true_position]
        )
        errors.append(np.linalg.norm(estimated_delta - true_delta))
        denominator += np.linalg.norm(true_delta)
    return safe_div(sum(errors), denominator + EPS) if errors else np.nan


def _bootstrap_varx_stability(
    design: np.ndarray,
    response: np.ndarray,
    fitted: np.ndarray,
    raw_breaks_lagged: list[int],
    retained_b_breaks: list[int],
    critical: float,
    min_segment: int,
    scan_step: int,
    block_size: int,
    n_outcomes: int,
    n_features: int,
    rng: np.random.Generator,
    n_replicates: int,
) -> dict[str, float]:
    residuals = response - fitted
    raw_sets: list[list[int]] = []
    b_sets: list[list[int]] = []
    tolerance = max(5, int(np.ceil(0.01 * (len(design) + 1))))
    for _ in range(n_replicates):
        boot_response = fitted + _resample_residuals(residuals, rng, block_size)
        boot_raw_lagged = _binary_segmentation(
            design, boot_response, critical, min_segment, scan_step
        )
        boot_raw = [value + 1 for value in boot_raw_lagged]
        boot_b, _ = _classify_b_breaks(
            design, boot_response, boot_raw_lagged, n_outcomes, n_features
        )
        raw_sets.append(boot_raw)
        b_sets.append(boot_b)
    similarities = [
        _breakpoint_set_similarity(b_sets[i], b_sets[j], tolerance)
        for i in range(len(b_sets))
        for j in range(i + 1, len(b_sets))
    ]
    retained_frequencies = [
        np.mean([any(abs(value - target) <= tolerance for value in values) for values in b_sets])
        for target in retained_b_breaks
    ]
    return {
        "bootstrap_B_break_jaccard": float(np.mean(similarities)) if similarities else np.nan,
        "bootstrap_any_raw_break_frequency": float(np.mean([bool(values) for values in raw_sets])),
        "bootstrap_any_B_break_frequency": float(np.mean([bool(values) for values in b_sets])),
        "bootstrap_mean_B_break_count": float(np.mean([len(values) for values in b_sets])),
        "bootstrap_retained_B_change_probability": float(np.mean(retained_frequencies))
        if retained_frequencies
        else np.nan,
    }


def evaluate_pair(pair: VarxPair, seed: int, n_bootstrap: int = 19) -> dict[str, Any]:
    timer = Timer()
    xd_raw, yd_raw = pair.discovery["X"], pair.discovery["Y"]
    xe_raw, ye_raw = pair.evaluation["X"], pair.evaluation["Y"]
    xm, xs = standardize_fit(xd_raw)
    ym, ys = standardize_fit(yd_raw)
    xd, xe = standardize_apply(xd_raw, xm, xs), standardize_apply(xe_raw, xm, xs)
    yd, ye = standardize_apply(yd_raw, ym, ys), standardize_apply(ye_raw, ym, ys)
    qd, rd = make_lagged(xd, yd, pair.discovery["sequence_id"])
    qe, re = make_lagged(xe, ye, pair.evaluation["sequence_id"])
    n_outcomes, n_features = yd.shape[1], xd.shape[1]
    min_segment = max(90, 3 * (1 + n_outcomes + n_features))
    scan_step = 5
    block_size = 12 if pair.metadata["scenario"] == "high_persistence" else 1
    rng = np.random.default_rng(seed + 49031)
    critical, p_value, _ = _calibrate_full_search(
        qd, rd, min_segment, scan_step, rng, n_bootstrap, block_size
    )
    raw_breaks_lagged = (
        _binary_segmentation(qd, rd, critical, min_segment, scan_step)
        if p_value <= 0.05
        else []
    )
    raw_breaks = [value + 1 for value in raw_breaks_lagged]
    b_breaks, b_p_values = _classify_b_breaks(
        qd, rd, raw_breaks_lagged, n_outcomes, n_features
    )
    relevance, relevance_gains = _segment_relevance(
        qd, rd, raw_breaks_lagged, n_outcomes
    )
    coefficients, discovery_fitted = _fit_piecewise(
        qd, rd, raw_breaks_lagged
    )
    piecewise_pred = _predict_piecewise(qe, coefficients, raw_breaks_lagged)
    pooled_varx, _ = _segment_fit(qd, rd, 0, len(qd))
    pooled_varx_pred = ridge_predict(qe, pooled_varx)
    pooled_ar, _ = _segment_fit(qd[:, :n_outcomes], rd, 0, len(qd))
    pooled_ar_pred = ridge_predict(qe[:, :n_outcomes], pooled_ar)
    true_all = pair.truth["true_breakpoints_all"].tolist()
    true_b = pair.truth["true_breakpoints_B"].tolist()
    oracle_breaks_lagged = [value - 1 for value in true_all]
    oracle_coefficients, _ = _fit_piecewise(qd, rd, oracle_breaks_lagged)
    oracle_pred = _predict_piecewise(qe, oracle_coefficients, oracle_breaks_lagged)
    ar_loss = normalized_joint_loss(re, pooled_ar_pred, np.mean(rd, axis=0))
    varx_loss = normalized_joint_loss(re, pooled_varx_pred, np.mean(rd, axis=0))
    piecewise_loss = normalized_joint_loss(re, piecewise_pred, np.mean(rd, axis=0))
    oracle_loss = normalized_joint_loss(re, oracle_pred, np.mean(rd, axis=0))
    tolerance = max(5, int(np.ceil(0.01 * len(xd))))
    raw_metrics = breakpoint_scores(
        true_all, raw_breaks, tolerance, len(xd), "raw_break"
    )
    b_metrics = breakpoint_scores(
        true_b, b_breaks, tolerance, len(xd), "B_break"
    )
    partition = partition_scores(
        segment_labels(len(xd), true_all), segment_labels(len(xd), raw_breaks)
    )
    true_b_by_estimated = _truth_for_estimated_segments(
        raw_breaks, true_all, pair.truth["B"], len(xd)
    )
    true_a_by_estimated = _truth_for_estimated_segments(
        raw_breaks, true_all, pair.truth["A"], len(xd)
    )

    # Convert raw generator coefficients to the standardized coordinate system.
    standardized_b = [
        truth_b * xs[None, :] / ys[:, None] for truth_b in true_b_by_estimated
    ]
    standardized_a = [
        truth_a * ys[None, :] / ys[:, None] for truth_a in true_a_by_estimated
    ]
    coefficient_metrics = _coefficient_metrics(
        coefficients, standardized_b, standardized_a, n_outcomes
    )
    true_relevance = [bool(np.linalg.norm(value) > EPS) for value in standardized_b]
    relevance_metrics = binary_scores(true_relevance, relevance)
    standardized_b_by_true_segment = [
        truth_b * xs[None, :] / ys[:, None] for truth_b in pair.truth["B"]
    ]
    delta_b_error = _delta_b_error(
        coefficients,
        raw_breaks,
        b_breaks,
        true_b,
        standardized_b_by_true_segment,
        n_outcomes,
        tolerance,
    )
    scenario = pair.metadata["scenario"]
    result: dict[str, Any] = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "status": "ok",
        "global_search_p": p_value,
        "critical_score": critical,
        "n_true_breaks_all": float(len(true_all)),
        "n_true_breaks_B": float(len(true_b)),
        "n_raw_breaks": float(len(raw_breaks)),
        "n_B_breaks": float(len(b_breaks)),
        "segment_ari": partition["ari"],
        "segment_nmi": partition["nmi"],
        "boundary_displacement": 1.0 - partition["row_accuracy"],
        "segment_relevance_precision": relevance_metrics["precision"],
        "segment_relevance_recall": relevance_metrics["recall"],
        "segment_relevance_f1": relevance_metrics["f1"],
        "segment_relevance_loss_gain": float(np.mean(relevance_gains)),
        "pooled_AR_test_loss": ar_loss,
        "pooled_VARX_test_loss": varx_loss,
        "piecewise_VARX_test_loss": piecewise_loss,
        "oracle_piecewise_VARX_test_loss": oracle_loss,
        "VARX_improvement_over_AR": ar_loss - varx_loss,
        "piecewise_improvement_over_VARX": varx_loss - piecewise_loss,
        "oracle_improvement_recovered": safe_div(
            varx_loss - piecewise_loss, varx_loss - oracle_loss
        ),
        "false_B_attribution": float(not true_b and bool(b_breaks)),
        "delta_B_relative_error": delta_b_error,
        "runtime_seconds": timer.seconds,
    }
    result.update(raw_metrics)
    result.update(b_metrics)
    result.update(coefficient_metrics)
    result.update(
        _bootstrap_varx_stability(
            qd,
            rd,
            discovery_fitted,
            raw_breaks_lagged,
            b_breaks,
            critical,
            min_segment,
            scan_step,
            block_size,
            n_outcomes,
            n_features,
            rng,
            min(5, n_bootstrap),
        )
    )
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
