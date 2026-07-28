from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from .common import (
    EPS,
    Timer,
    normalized_joint_loss,
    partition_scores,
    ridge_fit,
    ridge_predict,
    safe_div,
    save_dataset,
    set_scores,
    standardize_apply,
    standardize_fit,
    summarize_records,
)


METHOD = "causal_model_determination"
SCENARIOS = [
    "null",
    "clean_one_split",
    "weak_change",
    "correlated_proxy",
    "unbalanced_regimes",
    "multiple_outcomes",
    "multiple_splits",
    "global_nonlinearity_correct",
    "global_nonlinearity_misspecified",
    "main_effect_distractor",
    "variance_only_change",
]


@dataclass
class CausalPair:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _covariance(n_features: int, rho: float) -> np.ndarray:
    idx = np.arange(n_features)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_pair(scenario: str, seed: int, n_discovery: int = 500, n_evaluation: int = 700) -> CausalPair:
    rng = np.random.default_rng(seed)
    n_features, n_outcomes, n_predictors = 20, 3, 4
    regime_feature, proxy_feature, second_regime_feature = 4, 5, 6
    rho = 0.3
    sigma_x = _covariance(n_features, rho)
    baseline = rng.normal(0, 0.55, size=(n_predictors, n_outcomes))
    baseline[0, 0] += 0.8
    direction = rng.normal(size=baseline.shape)
    direction /= np.linalg.norm(direction)
    delta = 1.0
    threshold = 0.0
    true_features: list[int] = [regime_feature]
    nonlinear = scenario.startswith("global_nonlinearity")
    if scenario == "weak_change":
        delta = 0.3
    if scenario == "unbalanced_regimes":
        threshold = 0.994457883209753  # standard-normal 84th percentile
    if scenario in {"null", "main_effect_distractor"} or nonlinear:
        delta = 0.0
        true_features = []
    if scenario == "variance_only_change":
        delta = 0.0
        true_features = []
    if scenario == "multiple_splits":
        true_features = [regime_feature, second_regime_feature]

    def sample_x(n_rows: int) -> np.ndarray:
        x = rng.multivariate_normal(np.zeros(n_features), sigma_x, size=n_rows)
        if scenario == "correlated_proxy":
            x[:, proxy_feature] = 0.9 * x[:, regime_feature] + np.sqrt(1 - 0.9**2) * rng.normal(size=n_rows)
        return x

    def labels(x: np.ndarray) -> np.ndarray:
        if scenario == "multiple_splits":
            return np.where(
                x[:, regime_feature] > threshold,
                2,
                np.where(x[:, second_regime_feature] <= 0, 0, 1),
            ).astype(int)
        if scenario in {"null", "main_effect_distractor", "variance_only_change"} or nonlinear:
            return np.zeros(len(x), dtype=int)
        return (x[:, regime_feature] > threshold).astype(int)

    n_regimes = 3 if scenario == "multiple_splits" else (1 if not true_features else 2)
    coefficients = np.repeat(baseline[None, :, :], n_regimes, axis=0)
    if n_regimes >= 2:
        coefficients[1] += delta * direction
    if n_regimes == 3:
        direction2 = rng.normal(size=baseline.shape)
        direction2 /= np.linalg.norm(direction2)
        coefficients[2] += 1.25 * direction2
    if scenario == "multiple_outcomes":
        coefficients[1, :, 1:] = coefficients[0, :, 1:]
    if scenario == "main_effect_distractor":
        coefficients[0, 3, :] += 2.0
    intercepts = np.zeros((n_regimes, n_outcomes))
    base_noise = np.fromfunction(lambda i, j: 0.35 ** np.abs(i - j), (n_outcomes, n_outcomes))
    base_noise *= 0.65**2

    def sample(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = sample_x(n_rows)
        regime = labels(x)
        if nonlinear:
            y = np.column_stack(
                [
                    x[:, 0] ** 2,
                    0.7 * x[:, 0] ** 2 - 0.4 * x[:, 1],
                    -0.5 * x[:, 0] ** 2 + 0.3 * x[:, 2],
                ]
            )
        else:
            y = np.empty((n_rows, n_outcomes))
            for row in range(n_rows):
                k = regime[row] if n_regimes > 1 else 0
                y[row] = intercepts[k] + x[row, :n_predictors] @ coefficients[k]
        noise = np.empty_like(y)
        for row in range(n_rows):
            scale = 2.0 if scenario == "variance_only_change" and x[row, regime_feature] > threshold else 1.0
            noise[row] = rng.multivariate_normal(np.zeros(n_outcomes), base_noise * scale**2)
        return x, y + noise, regime

    xd, yd, rd = sample(n_discovery)
    xe, ye, re = sample(n_evaluation)
    metadata = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "feature_names": [f"X{j + 1}" for j in range(n_features)],
        "outcome_names": [f"Y{j + 1}" for j in range(n_outcomes)],
        "ordinary_predictors": list(range(n_predictors)),
        "true_regime_features": true_features,
        "proxy_groups": {str(regime_feature): [proxy_feature]} if scenario == "correlated_proxy" else {},
        "true_tree": (
            [
                {"feature": regime_feature, "operator": "<=", "threshold": threshold},
                {"feature": second_regime_feature, "operator": "<=", "threshold": 0.0, "parent": "left"},
            ]
            if scenario == "multiple_splits"
            else ([{"feature": regime_feature, "operator": "<=", "threshold": threshold}] if true_features else [])
        ),
        "correct_nonlinear_basis": scenario == "global_nonlinearity_correct",
        "estimand": "conditional_mean_coefficients",
    }
    truth = {
        "regime_discovery": rd,
        "regime_evaluation": re,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "true_regime_features": np.asarray(true_features, dtype=int),
        "thresholds": np.asarray(
            [threshold, 0.0] if scenario == "multiple_splits" else ([threshold] if true_features else []),
            dtype=float,
        ),
        "ordinary_predictors": np.arange(n_predictors, dtype=int),
        "noise_covariance": base_noise,
    }
    return CausalPair({"X": xd, "Y": yd}, {"X": xe, "Y": ye}, truth, metadata)


def _local_design(x: np.ndarray, correct_nonlinear_basis: bool) -> np.ndarray:
    design = x[:, :4]
    if correct_nonlinear_basis:
        design = np.column_stack([design, x[:, 0] ** 2])
    return design


def _best_split(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    correct_nonlinear_basis: bool,
    min_fraction: float = 0.15,
) -> dict[str, Any]:
    local_x = _local_design(x[rows], correct_nonlinear_basis)
    local_y = y[rows]
    pooled = ridge_fit(local_x, local_y, alpha=1e-5)
    pooled_sse = float(np.sum((local_y - ridge_predict(local_x, pooled)) ** 2))
    n_rows, n_outcomes = local_y.shape
    extra_df = (local_x.shape[1] + 1) * n_outcomes
    best: dict[str, Any] = {"score": -np.inf, "feature": None, "threshold": None}
    for feature in range(x.shape[1]):
        values = x[rows, feature]
        thresholds = np.unique(np.quantile(values, np.linspace(min_fraction, 1 - min_fraction, 15)))
        for threshold in thresholds:
            left = values <= threshold
            n_left = int(np.sum(left))
            if n_left < max(20, int(min_fraction * n_rows)) or n_rows - n_left < max(20, int(min_fraction * n_rows)):
                continue
            left_coef = ridge_fit(local_x[left], local_y[left], alpha=1e-5)
            right_coef = ridge_fit(local_x[~left], local_y[~left], alpha=1e-5)
            split_sse = float(
                np.sum((local_y[left] - ridge_predict(local_x[left], left_coef)) ** 2)
                + np.sum((local_y[~left] - ridge_predict(local_x[~left], right_coef)) ** 2)
            )
            statistic = n_rows * n_outcomes * np.log(max(pooled_sse, EPS) / max(split_sse, EPS))
            score = statistic - extra_df * np.log(n_rows)
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "raw_statistic": float(statistic),
                    "feature": int(feature),
                    "threshold": float(threshold),
                    "pooled_sse": pooled_sse,
                    "split_sse": split_sse,
                }
    return best


def _grow_tree(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    correct_nonlinear_basis: bool,
    depth: int,
    max_depth: int,
    force_root: bool,
) -> dict[str, Any]:
    split = _best_split(x, y, rows, correct_nonlinear_basis)
    accepted = split["feature"] is not None and (force_root or split["score"] > 6.0)
    if depth >= max_depth or not accepted:
        coef = ridge_fit(_local_design(x[rows], correct_nonlinear_basis), y[rows], alpha=1e-5)
        return {"leaf": True, "coef": coef, "n": int(len(rows))}
    mask = x[rows, split["feature"]] <= split["threshold"]
    return {
        "leaf": False,
        "feature": split["feature"],
        "threshold": split["threshold"],
        "score": split["score"],
        "left": _grow_tree(
            x, y, rows[mask], correct_nonlinear_basis, depth + 1, max_depth, False
        ),
        "right": _grow_tree(
            x, y, rows[~mask], correct_nonlinear_basis, depth + 1, max_depth, False
        ),
    }


def _apply_tree(
    tree: dict[str, Any], x: np.ndarray, correct_nonlinear_basis: bool
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty((len(x), tree_output_dim(tree)))
    labels = np.empty(len(x), dtype=int)
    next_label = 0

    def visit(node: dict[str, Any], rows: np.ndarray) -> None:
        nonlocal next_label
        if node["leaf"]:
            predictions[rows] = ridge_predict(_local_design(x[rows], correct_nonlinear_basis), node["coef"])
            labels[rows] = next_label
            next_label += 1
            return
        mask = x[rows, node["feature"]] <= node["threshold"]
        visit(node["left"], rows[mask])
        visit(node["right"], rows[~mask])

    visit(tree, np.arange(len(x)))
    return predictions, labels


def tree_output_dim(tree: dict[str, Any]) -> int:
    if tree["leaf"]:
        return int(tree["coef"].shape[1])
    return tree_output_dim(tree["left"])


def _tree_features(tree: dict[str, Any]) -> list[int]:
    if tree["leaf"]:
        return []
    return [tree["feature"]] + _tree_features(tree["left"]) + _tree_features(tree["right"])


def _leaf_coefs(tree: dict[str, Any]) -> list[np.ndarray]:
    if tree["leaf"]:
        return [tree["coef"]]
    return _leaf_coefs(tree["left"]) + _leaf_coefs(tree["right"])


def _root_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    correct_nonlinear_basis: bool,
    observed_score: float,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    design = _local_design(x, correct_nonlinear_basis)
    pooled = ridge_fit(design, y, alpha=1e-5)
    fitted = ridge_predict(design, pooled)
    residuals = y - fitted
    exceed = 0
    rows = np.arange(len(x))
    for _ in range(n_permutations):
        boot_y = fitted + residuals[rng.permutation(len(x))]
        score = _best_split(x, boot_y, rows, correct_nonlinear_basis)["score"]
        exceed += int(score >= observed_score)
    return float((exceed + 1) / (n_permutations + 1))


def _fit_detect(
    x: np.ndarray,
    y: np.ndarray,
    correct_nonlinear_basis: bool,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    rows = np.arange(len(x))
    root = _best_split(x, y, rows, correct_nonlinear_basis)
    p_value = _root_permutation_p(
        x, y, correct_nonlinear_basis, root["score"], rng, n_permutations
    )
    detected = p_value <= 0.05 and root["score"] > 0
    if detected:
        tree = _grow_tree(x, y, rows, correct_nonlinear_basis, 0, 2, True)
    else:
        tree = {
            "leaf": True,
            "coef": ridge_fit(_local_design(x, correct_nonlinear_basis), y, alpha=1e-5),
            "n": len(x),
        }
    return tree, p_value, root


def _bootstrap_stability(
    x: np.ndarray,
    y: np.ndarray,
    correct_nonlinear_basis: bool,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    selected: list[int | None] = []
    thresholds: list[float] = []
    feature_sets: list[set[int]] = []
    leaf_counts: list[int] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        root = _best_split(x[idx], y[idx], np.arange(len(idx)), correct_nonlinear_basis)
        if root["feature"] is not None and root["score"] > 0:
            tree = _grow_tree(
                x[idx], y[idx], np.arange(len(idx)), correct_nonlinear_basis, 0, 2, True
            )
            selected.append(root["feature"])
            thresholds.append(root["threshold"])
            features = set(_tree_features(tree))
            feature_sets.append(features)
            leaf_counts.append(len(_leaf_coefs(tree)))
        else:
            selected.append(None)
            feature_sets.append(set())
            leaf_counts.append(1)
    jaccards = []
    for i in range(len(feature_sets)):
        for j in range(i + 1, len(feature_sets)):
            union = feature_sets[i] | feature_sets[j]
            jaccards.append(len(feature_sets[i] & feature_sets[j]) / len(union) if union else 1.0)
    result: dict[str, float] = {
        "bootstrap_support_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
        "bootstrap_threshold_median": float(np.median(thresholds)) if thresholds else np.nan,
        "bootstrap_threshold_q05": float(np.quantile(thresholds, 0.05)) if thresholds else np.nan,
        "bootstrap_threshold_q95": float(np.quantile(thresholds, 0.95)) if thresholds else np.nan,
        "bootstrap_leaf_count_mode": float(np.bincount(leaf_counts).argmax()) if leaf_counts else np.nan,
    }
    for feature in range(x.shape[1]):
        result[f"bootstrap_root_frequency_X{feature + 1}"] = float(
            np.mean([value == feature for value in selected])
        )
    return result


def _parameter_error(
    leaf_coefs: list[np.ndarray],
    estimated_labels: np.ndarray,
    true_labels: np.ndarray,
    true_coefficients: np.ndarray,
    true_intercepts: np.ndarray,
) -> tuple[float, float]:
    if true_coefficients.shape[0] == 0 or len(leaf_coefs) == 0:
        return np.nan, np.nan
    est_values = np.unique(estimated_labels)
    true_values = np.unique(true_labels)
    overlap = np.zeros((len(est_values), len(true_values)))
    for i, ev in enumerate(est_values):
        for j, tv in enumerate(true_values):
            overlap[i, j] = np.sum((estimated_labels == ev) & (true_labels == tv))
    rows, cols = linear_sum_assignment(-overlap)
    errors, intercept_errors = [], []
    denominator = 0.0
    for est_pos, true_pos in zip(rows, cols):
        coef = leaf_coefs[int(est_pos)]
        truth = true_coefficients[int(true_values[true_pos])]
        errors.append(np.linalg.norm(coef[1 : 1 + truth.shape[0]] - truth))
        intercept_errors.append(
            np.linalg.norm(coef[0] - true_intercepts[int(true_values[true_pos])])
        )
        denominator += np.linalg.norm(truth)
    return safe_div(sum(errors), denominator + EPS), float(np.mean(intercept_errors))


def evaluate_pair(
    pair: CausalPair, seed: int, n_permutations: int = 19, n_bootstrap: int = 5
) -> dict[str, Any]:
    timer = Timer()
    xd, yd = pair.discovery["X"], pair.discovery["Y"]
    xe, ye = pair.evaluation["X"], pair.evaluation["Y"]
    x_mean, x_scale = standardize_fit(xd)
    y_mean, y_scale = standardize_fit(yd)
    xd_s = standardize_apply(xd, x_mean, x_scale)
    xe_s = standardize_apply(xe, x_mean, x_scale)
    yd_s = standardize_apply(yd, y_mean, y_scale)
    ye_s = standardize_apply(ye, y_mean, y_scale)
    nonlinear_basis = bool(pair.metadata["correct_nonlinear_basis"])
    rng = np.random.default_rng(seed + 71191)
    tree, p_value, root = _fit_detect(
        xd_s, yd_s, nonlinear_basis, rng, n_permutations
    )
    pred, estimated_regime = _apply_tree(tree, xe_s, nonlinear_basis)
    selected_features = sorted(set(_tree_features(tree)))
    true_features = pair.truth["true_regime_features"].tolist()
    feature_metrics = set_scores(true_features, selected_features)
    partition = partition_scores(pair.truth["regime_evaluation"], estimated_regime)
    pooled_coef = ridge_fit(_local_design(xd_s, nonlinear_basis), yd_s)
    pooled_pred = ridge_predict(_local_design(xe_s, nonlinear_basis), pooled_coef)
    pooled_loss = normalized_joint_loss(ye_s, pooled_pred, np.mean(yd_s, axis=0))
    partitioned_loss = normalized_joint_loss(ye_s, pred, np.mean(yd_s, axis=0))

    oracle_pred = np.empty_like(ye_s)
    for regime in np.unique(pair.truth["regime_discovery"]):
        train_mask = pair.truth["regime_discovery"] == regime
        test_mask = pair.truth["regime_evaluation"] == regime
        coef = ridge_fit(_local_design(xd_s[train_mask], nonlinear_basis), yd_s[train_mask])
        oracle_pred[test_mask] = ridge_predict(_local_design(xe_s[test_mask], nonlinear_basis), coef)
    oracle_loss = normalized_joint_loss(ye_s, oracle_pred, np.mean(yd_s, axis=0))

    predictive_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(20, int(0.1 * len(xd_s))), random_state=seed)
    predictive_tree.fit(xd_s, yd_s)
    predictive_tree_pred = predictive_tree.predict(xe_s)
    predictive_tree_loss = normalized_joint_loss(
        ye_s, predictive_tree_pred, np.mean(yd_s, axis=0)
    )
    standardized_coefficients = (
        pair.truth["coefficients"]
        * x_scale[: pair.truth["coefficients"].shape[1]][None, :, None]
        / y_scale[None, None, :]
    )
    standardized_intercepts = np.empty_like(pair.truth["intercepts"])
    for regime in range(len(standardized_intercepts)):
        standardized_intercepts[regime] = (
            pair.truth["intercepts"][regime]
            + x_mean[: pair.truth["coefficients"].shape[1]]
            @ pair.truth["coefficients"][regime]
            - y_mean
        ) / y_scale
    parameter_error, intercept_error = _parameter_error(
        _leaf_coefs(tree),
        estimated_regime,
        pair.truth["regime_evaluation"],
        standardized_coefficients,
        standardized_intercepts,
    )
    threshold_error = np.nan
    if true_features and not tree["leaf"] and root["feature"] == true_features[0]:
        true_threshold = (
            float(pair.truth["thresholds"][0]) - x_mean[true_features[0]]
        ) / x_scale[true_features[0]]
        iqr = np.subtract(*np.percentile(xd_s[:, true_features[0]], [75, 25]))
        threshold_error = abs(root["threshold"] - true_threshold) / max(iqr, EPS)
    proxy_hit = int(bool(
        root["feature"] == (true_features[0] if true_features else -1)
        or (
            pair.metadata["proxy_groups"]
            and root["feature"] in next(iter(pair.metadata["proxy_groups"].values()))
        )
    ))
    stability = _bootstrap_stability(xd_s, yd_s, nonlinear_basis, rng, n_bootstrap)
    result: dict[str, Any] = {
        "method": METHOD,
        "scenario": pair.metadata["scenario"],
        "seed": seed,
        "status": "ok",
        "split_detected": float(not tree["leaf"]),
        "n_true_splits": float(len(pair.metadata["true_tree"])),
        "n_detected_splits": float(len(_tree_features(tree))),
        "selected_root_feature": float(root["feature"]) if not tree["leaf"] else -1.0,
        "root_feature_exact_hit": float(bool(true_features) and not tree["leaf"] and root["feature"] == true_features[0]),
        "root_feature_proxy_family_hit": float(proxy_hit if not tree["leaf"] else 0),
        "permutation_p": p_value,
        "feature_precision": feature_metrics["precision"],
        "feature_recall": feature_metrics["recall"],
        "feature_f1": feature_metrics["f1"],
        "feature_fdr": feature_metrics["fdr"],
        "split_count_absolute_error": float(abs(len(pair.metadata["true_tree"]) - len(_tree_features(tree)))),
        "threshold_mae_iqr": threshold_error,
        "regime_ari": partition["ari"],
        "regime_nmi": partition["nmi"],
        "regime_row_accuracy": partition["row_accuracy"],
        "minimum_leaf_fraction": float(np.bincount(estimated_regime).min() / len(estimated_regime)),
        "relative_parameter_error": parameter_error if not pair.metadata["scenario"].startswith("global_nonlinearity") else np.nan,
        "intercept_error": intercept_error if not pair.metadata["scenario"].startswith("global_nonlinearity") else np.nan,
        "joint_test_loss_pooled": pooled_loss,
        "joint_test_loss_partitioned": partitioned_loss,
        "joint_test_loss_predictive_tree": predictive_tree_loss,
        "joint_test_loss_oracle": oracle_loss,
        "delta_loss_pooled": pooled_loss - partitioned_loss,
        "oracle_improvement_recovered": safe_div(
            pooled_loss - partitioned_loss, pooled_loss - oracle_loss
        ),
        "rmse": float(np.sqrt(mean_squared_error(ye_s, pred))),
        "runtime_seconds": timer.seconds,
    }
    result.update(stability)
    return result


def run(
    data_root: Path,
    metrics_root: Path,
    seeds: list[int],
    n_permutations: int,
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
                records.append(evaluate_pair(pair, seed, n_permutations, n_bootstrap))
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
