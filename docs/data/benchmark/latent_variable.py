from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold

from .common import (
    EPS,
    Timer,
    correlation_matrix,
    match_components,
    normalized_joint_loss,
    ridge_fit,
    ridge_predict,
    safe_div,
    save_dataset,
    set_scores,
    standardize_apply,
    standardize_fit,
    summarize_records,
)


METHOD = "latent_variable_determination"
SCENARIOS = [
    "clean_balanced_groups",
    "weak_measurement",
    "weak_outcome_relevance",
    "outcome_irrelevant_latent_block",
    "correlated_latents",
    "unequal_group_sizes",
    "correlated_nuisance_features",
    "cross_loadings",
    "high_dimensional",
    "heterogeneous_outcome_scales",
    "missing_data",
    "complete_null",
]


@dataclass
class LatentPair:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _positive_equicorrelation(k: int, rho: float) -> np.ndarray:
    return np.full((k, k), rho) + np.eye(k) * (1.0 - rho)


def generate_pair(
    scenario: str, seed: int, n_discovery: int | None = None, n_evaluation: int | None = None
) -> LatentPair:
    rng = np.random.default_rng(seed + 100_003)
    k_true, n_outcomes = 4, 4
    high_dimensional = scenario == "high_dimensional"
    n_discovery = n_discovery or (180 if high_dimensional else 500)
    n_evaluation = n_evaluation or (400 if high_dimensional else 700)
    if high_dimensional:
        group_sizes = [30] * k_true
        n_noise = 100
    elif scenario == "unequal_group_sizes":
        group_sizes = [3, 6, 12, 20]
        n_noise = 12
    else:
        group_sizes = [8] * k_true
        n_noise = 12
    n_features = sum(group_sizes) + n_noise
    sigma_z = _positive_equicorrelation(
        k_true, 0.75 if scenario == "correlated_latents" else 0.0
    )
    measurement_sd = 1.35 if scenario == "weak_measurement" else 0.55
    outcome_strength = 0.3 if scenario == "weak_outcome_relevance" else 1.0
    relevant = list(range(k_true))
    if scenario == "outcome_irrelevant_latent_block":
        relevant = list(range(k_true - 1))
    if scenario == "complete_null":
        relevant = []
    loading = np.zeros((n_features, k_true))
    true_group = np.full(n_features, -1, dtype=int)
    start = 0
    for latent, group_size in enumerate(group_sizes):
        stop = start + group_size
        signs = rng.choice([-1.0, 1.0], size=group_size)
        loading[start:stop, latent] = signs * rng.uniform(0.7, 1.0, size=group_size)
        true_group[start:stop] = latent
        start = stop
    cross_loading_features: list[int] = []
    if scenario == "cross_loadings":
        candidates = rng.choice(sum(group_sizes), size=max(2, sum(group_sizes) // 5), replace=False)
        for feature in candidates:
            secondary = (true_group[feature] + rng.integers(1, k_true)) % k_true
            loading[feature, secondary] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.55)
            cross_loading_features.append(int(feature))
    gamma = np.zeros((n_outcomes, k_true))
    for latent in relevant:
        gamma[latent % n_outcomes, latent] = outcome_strength
        gamma[(latent + 1) % n_outcomes, latent] = 0.45 * outcome_strength * (-1) ** latent
    sigma_y = _positive_equicorrelation(n_outcomes, 0.25) * 0.65**2

    def sample(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = rng.multivariate_normal(np.zeros(k_true), sigma_z, size=n_rows)
        x = z @ loading.T + rng.normal(scale=measurement_sd, size=(n_rows, n_features))
        if scenario == "correlated_nuisance_features":
            nuisance_start = sum(group_sizes)
            nuisance_factor = rng.normal(size=(n_rows, 1))
            x[:, nuisance_start:] = 0.9 * nuisance_factor + 0.35 * rng.normal(
                size=(n_rows, n_noise)
            )
        y = z @ gamma.T + rng.multivariate_normal(np.zeros(n_outcomes), sigma_y, size=n_rows)
        if scenario == "heterogeneous_outcome_scales":
            y[:, 0] *= 25.0
        if scenario == "missing_data":
            mcar = rng.random(x.shape) < 0.06
            dependent_probability = 0.12 / (1 + np.exp(-np.nan_to_num(x[:, [0]])))
            dependent = rng.random(x.shape) < dependent_probability
            x[mcar | dependent] = np.nan
        return x, y, z

    xd, yd, zd = sample(n_discovery)
    xe, ye, ze = sample(n_evaluation)
    relevant_features = np.flatnonzero(np.isin(true_group, relevant))
    metadata = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "feature_names": [f"X{j + 1}" for j in range(n_features)],
        "outcome_names": [f"Y{j + 1}" for j in range(n_outcomes)],
        "relevant_latents": relevant,
        "cross_loading_features": cross_loading_features,
        "target": "outcome_relevant_predictive_components",
        "group_sizes": group_sizes,
        "n_noise_features": n_noise,
    }
    truth = {
        "Z_discovery": zd,
        "Z_evaluation": ze,
        "Lambda": loading,
        "Gamma": gamma,
        "true_group": true_group,
        "relevant_latents": np.asarray(relevant, dtype=int),
        "relevant_feature_set": relevant_features,
        "cross_loading_features": np.asarray(cross_loading_features, dtype=int),
        "Sigma_Z": sigma_z,
        "Sigma_Y": sigma_y,
    }
    return LatentPair({"X": xd, "Y": yd}, {"X": xe, "Y": ye}, truth, metadata)


def _impute_fit(x: np.ndarray) -> np.ndarray:
    means = np.nanmean(x, axis=0)
    return np.where(np.isfinite(means), means, 0.0)


def _impute_apply(x: np.ndarray, means: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), means, x)


def fit_sparse_pls(
    x: np.ndarray, y: np.ndarray, n_components: int, n_keep: int
) -> dict[str, np.ndarray | int]:
    if n_components == 0:
        return {
            "n_components": 0,
            "n_keep": n_keep,
            "weights": np.empty((x.shape[1], 0)),
            "x_loadings": np.empty((x.shape[1], 0)),
            "score_coef": np.mean(y, axis=0, keepdims=True),
        }
    xr, yr = x.copy(), y.copy()
    weights, loadings, scores = [], [], []
    for _ in range(n_components):
        covariance = xr.T @ yr / max(len(xr) - 1, 1)
        u, _, _ = np.linalg.svd(covariance, full_matrices=False)
        weight = u[:, 0]
        keep = min(n_keep, len(weight))
        retained = np.argpartition(np.abs(weight), -keep)[-keep:]
        sparse_weight = np.zeros_like(weight)
        sparse_weight[retained] = weight[retained]
        norm = np.linalg.norm(sparse_weight)
        if norm < EPS:
            break
        sparse_weight /= norm
        score = xr @ sparse_weight
        denominator = float(score @ score) + EPS
        x_loading = xr.T @ score / denominator
        y_loading = yr.T @ score / denominator
        xr -= np.outer(score, x_loading)
        yr -= np.outer(score, y_loading)
        weights.append(sparse_weight)
        loadings.append(x_loading)
        scores.append(score)
    if not scores:
        return fit_sparse_pls(x, y, 0, n_keep)
    score_matrix = np.column_stack(scores)
    return {
        "n_components": len(scores),
        "n_keep": n_keep,
        "weights": np.column_stack(weights),
        "x_loadings": np.column_stack(loadings),
        "score_coef": ridge_fit(score_matrix, y, alpha=1e-4),
    }


def sparse_transform(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if model["n_components"] == 0:
        return np.empty((len(x), 0))
    xr = x.copy()
    scores = []
    for component in range(model["n_components"]):
        score = xr @ model["weights"][:, component]
        scores.append(score)
        xr -= np.outer(score, model["x_loadings"][:, component])
    return np.column_stack(scores)


def sparse_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if model["n_components"] == 0:
        return np.repeat(model["score_coef"], len(x), axis=0)
    return ridge_predict(sparse_transform(model, x), model["score_coef"])


def tune_sparse_pls(
    x: np.ndarray,
    y: np.ndarray,
    max_components: int = 6,
    folds: int = 3,
    seed: int = 0,
) -> tuple[int, int, list[dict[str, float]]]:
    keep_grid = sorted(
        set(min(x.shape[1], value) for value in [4, 8, 12, 20, max(4, x.shape[1] // 4)])
    )
    candidates = [(0, keep_grid[0])] + [
        (components, keep)
        for components in range(1, min(max_components, y.shape[1] + 2) + 1)
        for keep in keep_grid
    ]
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    results: list[dict[str, float]] = []
    for components, keep in candidates:
        losses = []
        for train_idx, val_idx in splitter.split(x):
            xm, xs = standardize_fit(x[train_idx])
            ym, ys = standardize_fit(y[train_idx])
            xtrain = standardize_apply(x[train_idx], xm, xs)
            xval = standardize_apply(x[val_idx], xm, xs)
            ytrain = standardize_apply(y[train_idx], ym, ys)
            yval = standardize_apply(y[val_idx], ym, ys)
            model = fit_sparse_pls(xtrain, ytrain, components, keep)
            pred = sparse_predict(model, xval)
            losses.append(normalized_joint_loss(yval, pred, np.mean(ytrain, axis=0)))
        results.append(
            {
                "components": float(components),
                "keep": float(keep),
                "mean_loss": float(np.mean(losses)),
                "se_loss": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
            }
        )
    best = min(results, key=lambda row: row["mean_loss"])
    cutoff = best["mean_loss"] + best["se_loss"]
    eligible = [row for row in results if row["mean_loss"] <= cutoff]
    chosen = min(eligible, key=lambda row: (row["components"], row["keep"]))
    return int(chosen["components"]), int(chosen["keep"]), results


def hard_groups(weights: np.ndarray) -> np.ndarray:
    if weights.shape[1] == 0:
        return np.full(weights.shape[0], -1, dtype=int)
    maximum = np.max(np.abs(weights), axis=1)
    return np.where(maximum > EPS, np.argmax(np.abs(weights), axis=1), -1)


def _component_evaluation(
    model: dict[str, Any],
    x_test: np.ndarray,
    z_test: np.ndarray,
    loading_true: np.ndarray,
    true_group: np.ndarray,
    relevant: np.ndarray,
) -> dict[str, float]:
    scores = sparse_transform(model, x_test)
    estimated_groups = hard_groups(model["weights"])
    relevant_features = np.flatnonzero(np.isin(true_group, relevant))
    selected_features = np.flatnonzero(estimated_groups >= 0)
    support = set_scores(relevant_features.tolist(), selected_features.tolist())
    output: dict[str, float] = {
        "support_precision": support["precision"],
        "support_recall": support["recall"],
        "support_f1": support["f1"],
        "support_fdr": support["fdr"],
        "noise_features_selected": float(np.sum((true_group == -1) & (estimated_groups >= 0))),
        "assignment_coverage": float(np.mean(estimated_groups[relevant_features] >= 0))
        if len(relevant_features)
        else 1.0,
    }
    if len(relevant_features):
        output["ari_signal"] = float(
            adjusted_rand_score(true_group[relevant_features], estimated_groups[relevant_features])
        )
        output["nmi_signal"] = float(
            normalized_mutual_info_score(
                true_group[relevant_features], estimated_groups[relevant_features]
            )
        )
    else:
        output["ari_signal"] = np.nan
        output["nmi_signal"] = np.nan
    true_with_noise = np.where(np.isin(true_group, relevant), true_group, -1)
    output["ari_with_noise"] = float(adjusted_rand_score(true_with_noise, estimated_groups))
    if scores.shape[1] == 0 or len(relevant) == 0:
        output.update(
            {
                "mean_group_jaccard": np.nan,
                "mean_score_correlation": np.nan,
                "mean_score_r2": np.nan,
                "mean_loading_cosine": np.nan,
                "subspace_error": np.nan,
            }
        )
        return output
    corr = np.abs(correlation_matrix(scores, z_test[:, relevant]))
    matching = match_components(corr)
    group_jaccards, score_corrs, score_r2, loading_cosines = [], [], [], []
    for est_component, rel_position in matching:
        latent = int(relevant[rel_position])
        estimated_support = set(np.flatnonzero(estimated_groups == est_component))
        true_support = set(np.flatnonzero(true_group == latent))
        union = estimated_support | true_support
        group_jaccards.append(len(estimated_support & true_support) / len(union) if union else 1.0)
        signed_corr = float(
            correlation_matrix(scores[:, [est_component]], z_test[:, [latent]])[0, 0]
        )
        score_corrs.append(abs(signed_corr))
        score_r2.append(signed_corr**2)
        weight = model["weights"][:, est_component] * (1 if signed_corr >= 0 else -1)
        true_loading = loading_true[:, latent]
        loading_cosines.append(
            safe_div(float(weight @ true_loading), np.linalg.norm(weight) * np.linalg.norm(true_loading))
        )
    output["mean_group_jaccard"] = float(np.mean(group_jaccards))
    output["mean_score_correlation"] = float(np.mean(score_corrs))
    output["mean_score_r2"] = float(np.mean(score_r2))
    output["mean_loading_cosine"] = float(np.mean(loading_cosines))
    common = min(scores.shape[1], len(relevant))
    angles = subspace_angles(scores[:, :common], z_test[:, relevant[:common]])
    output["subspace_error"] = float(np.mean(np.sin(angles))) if len(angles) else np.nan
    return output


def _conditional_bootstrap_stability(
    x: np.ndarray,
    y: np.ndarray,
    components: int,
    keep: int,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    supports, groupings = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        model = fit_sparse_pls(x[idx], y[idx], components, keep)
        grouping = hard_groups(model["weights"])
        supports.append(set(np.flatnonzero(grouping >= 0)))
        groupings.append(grouping)
    jaccards, aris = [], []
    for i in range(len(supports)):
        for j in range(i + 1, len(supports)):
            union = supports[i] | supports[j]
            jaccards.append(len(supports[i] & supports[j]) / len(union) if union else 1.0)
            aris.append(adjusted_rand_score(groupings[i], groupings[j]))
    return {
        "bootstrap_support_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
        "bootstrap_group_ari": float(np.mean(aris)) if aris else np.nan,
    }


def evaluate_pair(pair: LatentPair, seed: int, n_bootstrap: int = 5) -> dict[str, Any]:
    timer = Timer()
    xd_raw, yd_raw = pair.discovery["X"], pair.discovery["Y"]
    xe_raw, ye_raw = pair.evaluation["X"], pair.evaluation["Y"]
    impute = _impute_fit(xd_raw)
    xd_imputed, xe_imputed = _impute_apply(xd_raw, impute), _impute_apply(xe_raw, impute)
    xm, xs = standardize_fit(xd_imputed)
    ym, ys = standardize_fit(yd_raw)
    xd = standardize_apply(xd_imputed, xm, xs)
    xe = standardize_apply(xe_imputed, xm, xs)
    yd = standardize_apply(yd_raw, ym, ys)
    ye = standardize_apply(ye_raw, ym, ys)
    components, keep, search = tune_sparse_pls(xd_imputed, yd_raw, seed=seed)
    model = fit_sparse_pls(xd, yd, components, keep)
    pred = sparse_predict(model, xe)
    joint_loss = normalized_joint_loss(ye, pred, np.mean(yd, axis=0))
    relevant = pair.truth["relevant_latents"]
    component_metrics = _component_evaluation(
        model,
        xe,
        pair.truth["Z_evaluation"],
        pair.truth["Lambda"],
        pair.truth["true_group"],
        relevant,
    )

    pca_components = max(1, min(len(relevant) if len(relevant) else 1, xd.shape[1], len(xd) - 1))
    pca = PCA(n_components=pca_components, random_state=seed).fit(xd)
    pca_train, pca_test = pca.transform(xd), pca.transform(xe)
    pca_coef = ridge_fit(pca_train, yd)
    pca_loss = normalized_joint_loss(
        ye, ridge_predict(pca_test, pca_coef), np.mean(yd, axis=0)
    )
    dense_components = max(1, min(components if components else 1, yd.shape[1], xd.shape[1]))
    dense_pls = PLSRegression(n_components=dense_components, scale=False).fit(xd, yd)
    dense_loss = normalized_joint_loss(ye, dense_pls.predict(xe), np.mean(yd, axis=0))
    try:
        sparse_regression = MultiTaskLassoCV(
            alphas=np.logspace(-3, 0, 7), cv=3, max_iter=3000, random_state=seed
        ).fit(xd, yd)
        sparse_regression_loss = normalized_joint_loss(
            ye, sparse_regression.predict(xe), np.mean(yd, axis=0)
        )
    except Exception:
        sparse_regression_loss = np.nan
    if len(relevant):
        oracle_train = pair.truth["Z_discovery"][:, relevant]
        oracle_test = pair.truth["Z_evaluation"][:, relevant]
        oracle_coef = ridge_fit(oracle_train, yd)
        oracle_loss = normalized_joint_loss(
            ye, ridge_predict(oracle_test, oracle_coef), np.mean(yd, axis=0)
        )
    else:
        oracle_loss = normalized_joint_loss(
            ye, np.repeat(np.mean(yd, axis=0, keepdims=True), len(ye), axis=0), np.mean(yd, axis=0)
        )
    variance = np.sum((ye - np.mean(yd, axis=0)) ** 2, axis=0)
    r2 = 1 - np.sum((ye - pred) ** 2, axis=0) / np.maximum(variance, EPS)
    rmse = np.sqrt(np.mean((ye - pred) ** 2, axis=0))
    stability = _conditional_bootstrap_stability(
        xd, yd, components, keep, np.random.default_rng(seed + 91009), n_bootstrap
    )
    result: dict[str, Any] = {
        "method": METHOD,
        "scenario": pair.metadata["scenario"],
        "seed": seed,
        "status": "ok",
        "K_true_relevant": float(len(relevant)),
        "K_selected": float(model["n_components"]),
        "K_absolute_error": float(abs(model["n_components"] - len(relevant))),
        "K_exact_recovery": float(model["n_components"] == len(relevant)),
        "selected_features": float(np.sum(hard_groups(model["weights"]) >= 0)),
        "any_component_selected": float(model["n_components"] > 0),
        "joint_test_nrmse": joint_loss,
        "mean_test_rmse": float(np.mean(rmse)),
        "mean_test_r2": float(np.mean(r2)),
        "pca_joint_test_nrmse": pca_loss,
        "dense_pls_joint_test_nrmse": dense_loss,
        "sparse_regression_joint_test_nrmse": sparse_regression_loss,
        "oracle_joint_test_nrmse": oracle_loss,
        "runtime_seconds": timer.seconds,
    }
    for index, value in enumerate(rmse):
        result[f"rmse_Y{index + 1}"] = float(value)
    for index, value in enumerate(r2):
        result[f"r2_Y{index + 1}"] = float(value)
    result.update(component_metrics)
    result.update(stability)
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
