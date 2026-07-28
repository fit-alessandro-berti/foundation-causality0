from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.sparse import csr_matrix
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    EPS,
    Timer,
    binary_scores,
    normalized_joint_loss,
    safe_div,
    save_dataset,
    set_scores,
    summarize_records,
)


METHOD = "causal_interference_detection"
SCENARIOS = [
    "direct_effect_null",
    "complete_null",
    "linear_spillover",
    "weak_effect_0_15",
    "weak_effect_0_30",
    "weak_effect_0_45",
    "direct_spillover_interaction",
    "cancellation",
    "threshold_saturation",
    "zero_selected_contrast",
    "sparse_multivariate_effects",
    "mixed_outcome_types",
    "poor_exposure_overlap",
    "degree_heterogeneity",
    "shared_outcome_shock",
    "observed_homophily",
    "hidden_confounding",
    "misspecified_exposure",
    "noisy_relation_matrix",
    "interference_outside_W",
    "policy_shift",
]

TARGET_G0 = 0.25
TARGET_G1 = 0.75
CURVE_GRID = np.linspace(0.1, 0.9, 9)


@dataclass
class InterferencePair:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]


class ConstantModel(BaseEstimator):
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.value, len(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        value = np.clip(self.value, 0.0, 1.0)
        return np.column_stack(
            [np.repeat(1 - value, len(x)), np.repeat(value, len(x))]
        )


class LogCountModel(BaseEstimator):
    """Fast response-scale count model used inside repeated cluster bootstraps."""

    def __init__(self) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogCountModel":
        self.model.fit(x, np.log1p(np.clip(y, 0.0, None)))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(np.clip(self.model.predict(x), -5.0, 5.0))


def compute_exposure(a: np.ndarray, w: np.ndarray, mapping: str) -> np.ndarray:
    weights = np.asarray(w, dtype=float).copy()
    np.fill_diagonal(weights, 0.0)
    if mapping == "unweighted":
        weights = (weights > 0).astype(float)
    elif mapping == "distance_two":
        adjacency = csr_matrix(weights > 0, dtype=float)
        reach = adjacency + adjacency @ adjacency
        reach.setdiag(0)
        reach.eliminate_zeros()
        reach.data[:] = 1.0
        numerator = np.asarray(reach @ a).ravel()
        denominator = np.asarray(reach.sum(axis=1)).ravel()
        return numerator / np.maximum(denominator, 1.0)
    elif mapping == "threshold":
        direct = compute_exposure(a, weights, "unweighted")
        return (direct >= 0.5).astype(float)
    elif mapping == "top_k":
        top_weights = np.zeros_like(weights)
        for row in range(len(weights)):
            nonzero = np.flatnonzero(weights[row] > 0)
            if len(nonzero):
                keep = nonzero[
                    np.argsort(weights[row, nonzero])[-min(3, len(nonzero)) :]
                ]
                top_weights[row, keep] = weights[row, keep]
        weights = top_weights
    elif mapping != "weighted":
        raise ValueError(f"Unknown exposure mapping: {mapping}")
    denominator = weights.sum(axis=1)
    return weights @ a / np.maximum(denominator, EPS)


def _make_graph(
    rng: np.random.Generator,
    cluster_id: np.ndarray,
    x: np.ndarray,
    scenario: str,
) -> np.ndarray:
    size = len(cluster_id)
    w = np.zeros((size, size), dtype=np.float32)
    for cluster in np.unique(cluster_id):
        rows = np.flatnonzero(cluster_id == cluster)
        n = len(rows)
        if scenario == "poor_exposure_overlap":
            adjacency = np.ones((n, n), dtype=bool)
            np.fill_diagonal(adjacency, False)
        elif scenario == "degree_heterogeneity":
            adjacency = rng.random((n, n)) < 0.08
            adjacency = np.triu(adjacency, 1)
            adjacency = adjacency | adjacency.T
            adjacency[0, 1:] = True
            adjacency[1:, 0] = True
        elif scenario in {"observed_homophily", "hidden_confounding"}:
            similarity = np.abs(x[rows, 0, None] - x[rows, 0][None, :])
            probability = 0.08 + 0.48 * np.exp(-similarity)
            adjacency = rng.random((n, n)) < probability
            adjacency = np.triu(adjacency, 1)
            adjacency = adjacency | adjacency.T
        else:
            adjacency = rng.random((n, n)) < 0.24
            adjacency = np.triu(adjacency, 1)
            adjacency = adjacency | adjacency.T
        np.fill_diagonal(adjacency, False)
        for local_row in range(n):
            if not np.any(adjacency[local_row]):
                other = (local_row + rng.integers(1, n)) % n
                adjacency[local_row, other] = adjacency[other, local_row] = True
        block_weights = rng.uniform(0.5, 1.5, size=(n, n))
        block_weights = (block_weights + block_weights.T) / 2
        block = adjacency * block_weights
        w[np.ix_(rows, rows)] = block.astype(np.float32)
    np.fill_diagonal(w, 0.0)
    return w


def _corrupt_relation_matrix(
    rng: np.random.Generator, w_true: np.ndarray, cluster_id: np.ndarray
) -> np.ndarray:
    w = w_true.copy()
    upper_edges = np.column_stack(np.where(np.triu(w > 0, 1)))
    if len(upper_edges):
        delete = upper_edges[
            rng.choice(len(upper_edges), size=max(1, len(upper_edges) // 7), replace=False)
        ]
        for i, j in delete:
            w[i, j] = w[j, i] = 0.0
    same_cluster = cluster_id[:, None] == cluster_id[None, :]
    nonedges = np.column_stack(
        np.where(np.triu((w == 0) & same_cluster, 1))
    )
    if len(nonedges):
        valid = nonedges[
            np.asarray(
                [
                    np.any(w_true[i] > 0) and np.any(w_true[j] > 0)
                    for i, j in nonedges
                ]
            )
        ]
        if len(valid):
            add = valid[
                rng.choice(
                    len(valid),
                    size=min(max(1, len(upper_edges) // 20), len(valid)),
                    replace=False,
                )
            ]
            for i, j in add:
                value = rng.uniform(0.4, 1.2)
                w[i, j] = w[j, i] = value
    existing = w > 0
    w[existing] *= rng.uniform(0.75, 1.25, size=np.sum(existing))
    np.fill_diagonal(w, 0.0)
    return w.astype(np.float32)


def _add_omitted_relations(
    rng: np.random.Generator, w_observed: np.ndarray, cluster_id: np.ndarray
) -> np.ndarray:
    w = w_observed.copy()
    for cluster in np.unique(cluster_id):
        rows = np.flatnonzero(cluster_id == cluster)
        candidates = np.column_stack(
            np.where(np.triu(w[np.ix_(rows, rows)] == 0, 1))
        )
        if len(candidates):
            add = candidates[
                rng.choice(
                    len(candidates), size=min(max(2, len(rows) // 3), len(candidates)), replace=False
                )
            ]
            for local_i, local_j in add:
                i, j = rows[local_i], rows[local_j]
                value = rng.uniform(0.5, 1.3)
                w[i, j] = w[j, i] = value
    return w.astype(np.float32)


def _response_function(g: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        return g
    if kind == "threshold":
        return (g > 0.5).astype(float)
    if kind == "saturating":
        return 1.0 - np.exp(-4.0 * g)
    if kind == "nonmonotone":
        return (g - 0.5) ** 2
    raise ValueError(kind)


def _mean_response(
    x: np.ndarray,
    a: np.ndarray | float,
    g: np.ndarray | float,
    alpha: np.ndarray,
    beta: np.ndarray,
    tau: np.ndarray,
    delta: np.ndarray,
    kappa: np.ndarray,
    cluster_effect: np.ndarray,
    response_kind: str,
    outcome_types: list[str],
) -> np.ndarray:
    a_values = np.broadcast_to(np.asarray(a, dtype=float), len(x))
    g_values = np.broadcast_to(np.asarray(g, dtype=float), len(x))
    fg = _response_function(g_values, response_kind)
    eta = (
        alpha[None, :]
        + x @ beta
        + a_values[:, None] * tau[None, :]
        + fg[:, None] * delta[None, :]
        + (a_values * fg)[:, None] * kappa[None, :]
        + cluster_effect
    )
    mean = eta.copy()
    for outcome, outcome_type in enumerate(outcome_types):
        if outcome_type == "binary":
            mean[:, outcome] = expit(eta[:, outcome])
        elif outcome_type == "count":
            mean[:, outcome] = np.exp(np.clip(eta[:, outcome], -4, 3.5))
    return mean


def _sample_outcomes(
    rng: np.random.Generator,
    mean: np.ndarray,
    outcome_types: list[str],
    residual_covariance: np.ndarray,
) -> np.ndarray:
    y = mean.copy()
    continuous = [i for i, value in enumerate(outcome_types) if value == "continuous"]
    if continuous:
        covariance = residual_covariance[np.ix_(continuous, continuous)]
        y[:, continuous] += rng.multivariate_normal(
            np.zeros(len(continuous)), covariance, size=len(y)
        )
    for outcome, outcome_type in enumerate(outcome_types):
        if outcome_type == "binary":
            y[:, outcome] = rng.binomial(1, np.clip(mean[:, outcome], 1e-4, 1 - 1e-4))
        elif outcome_type == "count":
            y[:, outcome] = rng.poisson(np.clip(mean[:, outcome], 1e-4, 30))
    return y


def _scenario_parameters(
    rng: np.random.Generator,
    scenario: str,
    n_features: int,
) -> dict[str, Any]:
    outcome_types = (
        ["continuous", "binary", "count"]
        if scenario == "mixed_outcome_types"
        else ["continuous"] * 4
    )
    m = len(outcome_types)
    alpha = np.zeros(m)
    if scenario == "mixed_outcome_types":
        alpha[:] = [0.0, -0.5, -0.3]
    beta = rng.normal(scale=0.25, size=(n_features, m))
    tau = np.full(m, 0.55)
    delta = np.asarray([0.8, -0.65, 0.5, 0.0][:m], dtype=float)
    kappa = np.zeros(m)
    response_kind = "linear"
    if scenario == "complete_null":
        tau[:] = 0.0
        delta[:] = 0.0
    elif scenario in {
        "direct_effect_null",
        "shared_outcome_shock",
        "observed_homophily",
    }:
        delta[:] = 0.0
    elif scenario.startswith("weak_effect_"):
        magnitude = float(scenario.rsplit("_", 1)[1]) / 100
        signs = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
        delta = magnitude * signs
    elif scenario == "direct_spillover_interaction":
        delta[:] = 0.25
        kappa[:] = np.asarray([0.85, -0.7, 0.6, 0.0][:m])
    elif scenario == "cancellation":
        delta[:] = np.asarray([0.8, -0.7, 0.6, 0.0][:m])
        kappa[:] = -2.0 * delta
    elif scenario == "threshold_saturation":
        response_kind = "threshold"
    elif scenario == "zero_selected_contrast":
        response_kind = "nonmonotone"
        delta[:] = np.asarray([2.5, -2.0, 1.8, 0.0][:m])
    elif scenario == "sparse_multivariate_effects":
        delta[:] = 0.0
        delta[0] = 0.85
        if m > 2:
            delta[2] = -0.65
    elif scenario in {
        "hidden_confounding",
        "misspecified_exposure",
        "noisy_relation_matrix",
        "interference_outside_W",
        "policy_shift",
    }:
        delta[:] = np.asarray([0.75, -0.55, 0.45, 0.0][:m])
    return {
        "outcome_types": outcome_types,
        "alpha": alpha,
        "beta": beta,
        "tau": tau,
        "delta": delta,
        "kappa": kappa,
        "response_kind": response_kind,
    }


def _make_sample(
    rng: np.random.Generator,
    scenario: str,
    params: dict[str, Any],
    n_clusters: int,
    cluster_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    s = n_clusters * cluster_size
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    h_cluster = rng.normal(size=(n_clusters, 1))
    x = rng.normal(size=(s, params["beta"].shape[0]))
    if scenario in {
        "observed_homophily",
        "hidden_confounding",
        "shared_outcome_shock",
    }:
        x[:, :2] += np.repeat(h_cluster, cluster_size, axis=0) * np.asarray([0.9, -0.5])
    w_base = _make_graph(rng, cluster_id, x, scenario)
    if scenario == "interference_outside_W":
        w_observed = w_base
        w_true = _add_omitted_relations(rng, w_base, cluster_id)
    else:
        w_true = w_base
        w_observed = (
            _corrupt_relation_matrix(rng, w_true, cluster_id)
            if scenario == "noisy_relation_matrix"
            else w_true.copy()
        )
    neighbor_x = compute_exposure(x[:, 0], w_true, "weighted")
    saturation_levels = (
        np.asarray([0.05, 0.95])
        if scenario == "poor_exposure_overlap"
        else np.asarray([0.2, 0.5, 0.8])
    )
    saturation = rng.choice(saturation_levels, size=n_clusters)
    p_cluster = saturation[cluster_id]
    assignment_design = "randomized_saturation"
    propensity = p_cluster.copy()
    if scenario in {"observed_homophily", "hidden_confounding"}:
        assignment_design = "observational"
        hidden = np.repeat(h_cluster[:, 0], cluster_size)
        logits = -0.1 + 0.55 * x[:, 0] + 0.35 * neighbor_x + 0.9 * hidden
        propensity = np.clip(expit(logits), 0.05, 0.95)
    a = rng.binomial(1, propensity)
    true_mapping = "distance_two" if scenario == "misspecified_exposure" else "weighted"
    primary_mapping = "unweighted" if scenario == "misspecified_exposure" else "weighted"
    g_true = compute_exposure(a, w_true, true_mapping)
    g_observed = compute_exposure(a, w_observed, primary_mapping)
    cluster_effect = np.zeros((s, len(params["outcome_types"])))
    if scenario == "shared_outcome_shock":
        shock = rng.normal(scale=1.0, size=(n_clusters, len(params["outcome_types"])))
        cluster_effect += shock[cluster_id]
    if scenario == "hidden_confounding":
        hidden = np.repeat(h_cluster, cluster_size, axis=0)
        cluster_effect += hidden * np.linspace(0.8, 1.1, len(params["outcome_types"]))
    mean = _mean_response(
        x,
        a,
        g_true,
        params["alpha"],
        params["beta"],
        params["tau"],
        params["delta"],
        params["kappa"],
        cluster_effect,
        params["response_kind"],
        params["outcome_types"],
    )
    m = len(params["outcome_types"])
    residual_covariance = np.fromfunction(
        lambda i, j: 0.3 ** np.abs(i - j), (m, m)
    ) * 0.7**2
    y = _sample_outcomes(
        rng, mean, params["outcome_types"], residual_covariance
    )
    degree_unweighted = np.sum(w_observed > 0, axis=1)
    degree_weighted = np.sum(w_observed, axis=1)
    observed_x = x.copy()
    observed_confounders = list(range(x.shape[1]))
    if scenario == "observed_homophily":
        observed_x = np.column_stack(
            [x, np.repeat(h_cluster[:, 0], cluster_size), neighbor_x]
        )
        observed_confounders = list(range(observed_x.shape[1]))
    observed = {
        "X": observed_x,
        "W_observed": w_observed.astype(np.float32),
        "A": a.astype(np.int8),
        "Y": y,
        "cluster_id": cluster_id.astype(np.int16),
        "degree_unweighted": degree_unweighted.astype(np.float32),
        "degree_weighted": degree_weighted.astype(np.float32),
        "assignment_saturation": p_cluster.astype(np.float32),
    }
    truth = {
        "X_structural": x,
        "W_true": w_true.astype(np.float32),
        "G_true": g_true,
        "G_primary_observed": g_observed,
        "G_unweighted": compute_exposure(a, w_observed, "unweighted"),
        "G_weighted": compute_exposure(a, w_observed, "weighted"),
        "G_distance_two": compute_exposure(a, w_observed, "distance_two"),
        "assignment_probability": propensity,
        "hidden_cluster_confounder": h_cluster[:, 0],
        "cluster_effect": cluster_effect,
        "conditional_mean_observed": mean,
        "residual_covariance": residual_covariance,
    }
    metadata = {
        "assignment_design": assignment_design,
        "primary_mapping": primary_mapping,
        "true_mapping": true_mapping,
        "observed_confounders": observed_confounders,
        "hidden_confounders": (
            ["cluster_H"] if scenario == "hidden_confounding" else []
        ),
    }
    return observed, truth, metadata


def _truth_estimands(
    observed: dict[str, np.ndarray],
    structural_truth: dict[str, np.ndarray],
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    x = structural_truth["X_structural"]
    cluster_effect = structural_truth["cluster_effect"]
    m = len(params["outcome_types"])
    mu = np.empty((2, 2, m))
    for a in [0, 1]:
        for g_index, g in enumerate([TARGET_G0, TARGET_G1]):
            mu[a, g_index] = np.mean(
                _mean_response(
                    x,
                    a,
                    g,
                    params["alpha"],
                    params["beta"],
                    params["tau"],
                    params["delta"],
                    params["kappa"],
                    cluster_effect,
                    params["response_kind"],
                    params["outcome_types"],
                ),
                axis=0,
            )
    ase = mu[:, 1] - mu[:, 0]
    ade = mu[1] - mu[0]
    interaction = ase[1] - ase[0]
    curve = np.empty((2, len(CURVE_GRID), m))
    for a in [0, 1]:
        for grid_index, g in enumerate(CURVE_GRID):
            curve[a, grid_index] = np.mean(
                _mean_response(
                    x,
                    a,
                    g,
                    params["alpha"],
                    params["beta"],
                    params["tau"],
                    params["delta"],
                    params["kappa"],
                    cluster_effect,
                    params["response_kind"],
                    params["outcome_types"],
                ),
                axis=0,
            )
    functional_d = np.mean(
        (curve - np.mean(curve, axis=1, keepdims=True)) ** 2, axis=1
    )
    outcome_scale = np.std(observed["Y"], axis=0)
    affected = np.abs(ase) > 1e-10
    functional_affected = functional_d > 1e-8
    return {
        "mu": mu,
        "ASE": ase,
        "SASE": ase / np.maximum(outcome_scale[None, :], EPS),
        "ADE": ade,
        "INT": interaction,
        "curve": curve,
        "functional_D": functional_d,
        "affected_outcomes": affected.astype(np.int8),
        "functional_affected_outcomes": functional_affected.astype(np.int8),
        "outcome_scale": outcome_scale,
    }


def generate_pair(
    scenario: str,
    seed: int,
    n_clusters: int = 24,
    cluster_size: int = 20,
) -> InterferencePair:
    rng = np.random.default_rng(seed + 500_009)
    n_features = 6
    params = _scenario_parameters(rng, scenario, n_features)
    discovery, truth_d, meta_d = _make_sample(
        rng, scenario, params, n_clusters, cluster_size
    )
    evaluation, truth_e, meta_e = _make_sample(
        rng, scenario, params, n_clusters, cluster_size
    )
    estimands = _truth_estimands(discovery, truth_d, params)
    policy_values = np.empty((2, len(params["outcome_types"])))
    for policy_index, probability in enumerate([0.2, 0.8]):
        draws = []
        for _ in range(30):
            allocation = rng.binomial(1, probability, size=len(evaluation["A"]))
            exposure = compute_exposure(
                allocation, truth_e["W_true"], meta_e["true_mapping"]
            )
            draws.append(
                np.mean(
                    _mean_response(
                        truth_e["X_structural"],
                        allocation,
                        exposure,
                        params["alpha"],
                        params["beta"],
                        params["tau"],
                        params["delta"],
                        params["kappa"],
                        truth_e["cluster_effect"],
                        params["response_kind"],
                        params["outcome_types"],
                    ),
                    axis=0,
                )
            )
        policy_values[policy_index] = np.mean(draws, axis=0)
    truth: dict[str, np.ndarray] = {
        "W_true_discovery": truth_d["W_true"],
        "W_true_evaluation": truth_e["W_true"],
        "G_true_discovery": truth_d["G_true"],
        "G_true_evaluation": truth_e["G_true"],
        "G_primary_observed_discovery": truth_d["G_primary_observed"],
        "G_primary_observed_evaluation": truth_e["G_primary_observed"],
        "assignment_probability_discovery": truth_d["assignment_probability"],
        "assignment_probability_evaluation": truth_e["assignment_probability"],
        "hidden_cluster_confounder_discovery": truth_d["hidden_cluster_confounder"],
        "hidden_cluster_confounder_evaluation": truth_e["hidden_cluster_confounder"],
        "cluster_effect_discovery": truth_d["cluster_effect"],
        "cluster_effect_evaluation": truth_e["cluster_effect"],
        "X_structural_discovery": truth_d["X_structural"],
        "X_structural_evaluation": truth_e["X_structural"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "tau": params["tau"],
        "delta": params["delta"],
        "kappa": params["kappa"],
        "policy_value": policy_values,
        "policy_OE": policy_values[1] - policy_values[0],
        **estimands,
    }
    for name in ["G_unweighted", "G_weighted", "G_distance_two"]:
        truth[f"{name}_discovery"] = truth_d[name]
        truth[f"{name}_evaluation"] = truth_e[name]
    metadata = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "feature_names": [f"X{j + 1}" for j in range(discovery["X"].shape[1])],
        "outcome_names": [f"Y{j + 1}" for j in range(len(params["outcome_types"]))],
        "outcome_types": params["outcome_types"],
        "response_function": params["response_kind"],
        "assignment_design": meta_d["assignment_design"],
        "assignment_metadata": {
            "saturation_levels": [0.05, 0.95]
            if scenario == "poor_exposure_overlap"
            else [0.2, 0.5, 0.8],
            "cluster_randomization": True,
        },
        "primary_mapping": meta_d["primary_mapping"],
        "true_exposure_mapping": meta_d["true_mapping"],
        "sensitivity_mappings": ["unweighted", "weighted", "distance_two"],
        "target_levels": {
            "a": [0, 1],
            "g0": TARGET_G0,
            "g1": TARGET_G1,
            "grid": CURVE_GRID.tolist(),
        },
        "observed_confounders": meta_d["observed_confounders"],
        "hidden_confounders": meta_d["hidden_confounders"],
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "interpretation": "causal only under assignment, overlap, consistency, and exposure-mapping assumptions",
    }
    return InterferencePair(discovery, evaluation, truth, metadata)


def _feature_matrix(
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    rows: np.ndarray,
    *,
    a_override: float | np.ndarray | None = None,
    g_override: float | np.ndarray | None = None,
    mode: str = "full",
) -> np.ndarray:
    a = observed["A"][rows].astype(float)
    exposure = np.asarray(g[rows], dtype=float)
    if a_override is not None:
        a = np.broadcast_to(np.asarray(a_override, dtype=float), len(rows))
    if g_override is not None:
        exposure = np.broadcast_to(np.asarray(g_override, dtype=float), len(rows))
    if mode == "naive":
        return np.column_stack([a, exposure, a * exposure])
    x = observed["X"][rows]
    degree = np.log1p(observed["degree_unweighted"][rows])
    saturation = observed["assignment_saturation"][rows]
    if mode == "no_g":
        return np.column_stack([a, x, degree, saturation])
    basis = np.column_stack(
        [
            exposure,
            exposure**2,
            exposure**3,
            (exposure > 0.5).astype(float),
            1.0 - np.exp(-4.0 * exposure),
        ]
    )
    return np.column_stack(
        [a, basis, a[:, None] * basis, x, degree, saturation]
    )


def _fit_one_model(
    x: np.ndarray, y: np.ndarray, outcome_type: str
) -> BaseEstimator:
    if len(y) == 0 or np.nanstd(y) < EPS:
        return ConstantModel(float(np.nanmean(y)) if len(y) else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if outcome_type == "binary":
            if len(np.unique(y)) < 2:
                return ConstantModel(float(np.mean(y)))
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            )
        elif outcome_type == "count":
            model = LogCountModel()
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x, y)
    return model


def _predict_one(
    model: BaseEstimator, x: np.ndarray, outcome_type: str
) -> np.ndarray:
    if outcome_type == "binary" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    prediction = np.asarray(model.predict(x), dtype=float)
    return np.clip(prediction, 0.0, None) if outcome_type == "count" else prediction


def _fit_models(
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    rows: np.ndarray,
    outcome_types: list[str],
    mode: str = "full",
) -> list[BaseEstimator]:
    features = _feature_matrix(observed, g, rows, mode=mode)
    return [
        _fit_one_model(features, observed["Y"][rows, outcome], outcome_type)
        for outcome, outcome_type in enumerate(outcome_types)
    ]


def _predict_models(
    models: list[BaseEstimator],
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    rows: np.ndarray,
    outcome_types: list[str],
    *,
    a_override: float | np.ndarray | None = None,
    g_override: float | np.ndarray | None = None,
    mode: str = "full",
) -> np.ndarray:
    features = _feature_matrix(
        observed,
        g,
        rows,
        a_override=a_override,
        g_override=g_override,
        mode=mode,
    )
    return np.column_stack(
        [
            _predict_one(model, features, outcome_type)
            for model, outcome_type in zip(models, outcome_types)
        ]
    )


def _crossfit_outcome_regression(
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    outcome_types: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    clusters = np.unique(observed["cluster_id"])
    folds = KFold(n_splits=3, shuffle=True, random_state=seed)
    n, m = observed["Y"].shape
    observed_prediction = np.empty((n, m))
    counter = np.empty((2, 2, n, m))
    curve = np.empty((2, len(CURVE_GRID), n, m))
    for train_cluster_positions, validation_cluster_positions in folds.split(clusters):
        train_clusters = clusters[train_cluster_positions]
        validation_clusters = clusters[validation_cluster_positions]
        train_rows = np.flatnonzero(np.isin(observed["cluster_id"], train_clusters))
        validation_rows = np.flatnonzero(
            np.isin(observed["cluster_id"], validation_clusters)
        )
        models = _fit_models(
            observed, g, train_rows, outcome_types, mode="full"
        )
        observed_prediction[validation_rows] = _predict_models(
            models,
            observed,
            g,
            validation_rows,
            outcome_types,
            mode="full",
        )
        for own_treatment in [0, 1]:
            for g_index, target in enumerate([TARGET_G0, TARGET_G1]):
                counter[own_treatment, g_index, validation_rows] = _predict_models(
                    models,
                    observed,
                    g,
                    validation_rows,
                    outcome_types,
                    a_override=own_treatment,
                    g_override=target,
                    mode="full",
                )
            for grid_index, target in enumerate(CURVE_GRID):
                curve[own_treatment, grid_index, validation_rows] = _predict_models(
                    models,
                    observed,
                    g,
                    validation_rows,
                    outcome_types,
                    a_override=own_treatment,
                    g_override=target,
                    mode="full",
                )
    mu = np.mean(counter, axis=2)
    curve_mu = np.mean(curve, axis=2)
    return {
        "observed_prediction": observed_prediction,
        "mu": mu,
        "ASE": mu[:, 1] - mu[:, 0],
        "curve": curve_mu,
    }


def _cluster_bootstrap(
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    outcome_types: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray]:
    clusters = np.unique(observed["cluster_id"])
    all_rows = np.arange(len(g))
    ase_draws = np.empty((n_bootstrap, 2, len(outcome_types)))
    curve_draws = np.empty(
        (n_bootstrap, 2, len(CURVE_GRID), len(outcome_types))
    )
    for bootstrap in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate(
            [
                np.flatnonzero(observed["cluster_id"] == cluster)
                for cluster in sampled_clusters
            ]
        )
        models = _fit_models(observed, g, rows, outcome_types, mode="full")
        for own_treatment in [0, 1]:
            target_predictions = []
            for target in [TARGET_G0, TARGET_G1]:
                target_predictions.append(
                    np.mean(
                        _predict_models(
                            models,
                            observed,
                            g,
                            all_rows,
                            outcome_types,
                            a_override=own_treatment,
                            g_override=target,
                            mode="full",
                        ),
                        axis=0,
                    )
                )
            ase_draws[bootstrap, own_treatment] = (
                target_predictions[1] - target_predictions[0]
            )
            for grid_index, target in enumerate(CURVE_GRID):
                curve_draws[bootstrap, own_treatment, grid_index] = np.mean(
                    _predict_models(
                        models,
                        observed,
                        g,
                        all_rows,
                        outcome_types,
                        a_override=own_treatment,
                        g_override=target,
                        mode="full",
                    ),
                    axis=0,
                )
    return ase_draws, curve_draws


def _null_cluster_bootstrap(
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    outcome_types: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(g))
    reduced_models = _fit_models(
        observed, g, rows, outcome_types, mode="no_g"
    )
    fitted = _predict_models(
        reduced_models,
        observed,
        g,
        rows,
        outcome_types,
        mode="no_g",
    )
    residuals = observed["Y"] - fitted
    clusters = np.unique(observed["cluster_id"])
    ase_draws = np.empty((n_bootstrap, 2, len(outcome_types)))
    curve_draws = np.empty(
        (n_bootstrap, 2, len(CURVE_GRID), len(outcome_types))
    )
    for bootstrap in range(n_bootstrap):
        y_bootstrap = fitted.copy()
        signs_by_cluster = {
            int(cluster): rng.choice([-1.0, 1.0]) for cluster in clusters
        }
        row_signs = np.asarray(
            [signs_by_cluster[int(cluster)] for cluster in observed["cluster_id"]]
        )
        for outcome, outcome_type in enumerate(outcome_types):
            if outcome_type == "binary":
                y_bootstrap[:, outcome] = rng.binomial(
                    1, np.clip(fitted[:, outcome], 1e-4, 1 - 1e-4)
                )
            elif outcome_type == "count":
                y_bootstrap[:, outcome] = rng.poisson(
                    np.clip(fitted[:, outcome], 1e-4, 30)
                )
            else:
                y_bootstrap[:, outcome] = (
                    fitted[:, outcome] + residuals[:, outcome] * row_signs
                )
        boot_observed = {**observed, "Y": y_bootstrap}
        fitted_bootstrap = _crossfit_outcome_regression(
            boot_observed, g, outcome_types, seed + bootstrap + 1
        )
        ase_draws[bootstrap] = fitted_bootstrap["ASE"]
        curve_draws[bootstrap] = fitted_bootstrap["curve"]
    return ase_draws, curve_draws


def _bh_rejections(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(p_values)
    ordered = p_values[order]
    threshold_positions = np.flatnonzero(
        ordered <= alpha * (np.arange(len(ordered)) + 1) / len(ordered)
    )
    rejected = np.zeros(len(p_values), dtype=bool)
    if len(threshold_positions):
        rejected[order[: threshold_positions[-1] + 1]] = True
    return rejected


def _global_test(
    estimate: np.ndarray, null_bootstrap: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    null_center = np.mean(null_bootstrap, axis=0)
    deviations = null_bootstrap - null_center
    covariance = np.atleast_2d(np.cov(deviations, rowvar=False))
    covariance += np.eye(len(estimate)) * 1e-8
    inverse = np.linalg.pinv(covariance)
    centered_estimate = estimate - null_center
    statistic = float(centered_estimate @ inverse @ centered_estimate)
    null_statistics = np.einsum(
        "bi,ij,bj->b", deviations, inverse, deviations
    )
    p_value = float(
        (1 + np.sum(null_statistics >= statistic))
        / (len(null_bootstrap) + 1)
    )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), EPS))
    outcome_p = 2 * norm.sf(
        np.abs(centered_estimate) / np.maximum(standard_error, EPS)
    )
    return statistic, p_value, standard_error, outcome_p


def _support_diagnostics(a: np.ndarray, g: np.ndarray, own_treatment: int) -> dict[str, float]:
    mask = a == own_treatment
    values = g[mask]
    if not len(values):
        return {
            "supported": 0.0,
            "ess_g0": 0.0,
            "ess_g1": 0.0,
            "raw_g0": 0.0,
            "raw_g1": 0.0,
        }
    diagnostics: dict[str, float] = {}
    supported = True
    lower, upper = np.quantile(values, [0.02, 0.98])
    for name, target in [("g0", TARGET_G0), ("g1", TARGET_G1)]:
        weights = np.exp(-0.5 * ((values - target) / 0.12) ** 2)
        ess = safe_div(np.sum(weights) ** 2, np.sum(weights**2), 0.0)
        raw = int(np.sum(np.abs(values - target) <= 0.1))
        diagnostics[f"ess_{name}"] = ess
        diagnostics[f"raw_{name}"] = float(raw)
        supported &= lower <= target <= upper and raw >= 10 and ess >= 10
    diagnostics["supported"] = float(supported)
    diagnostics["exposure_q05"] = float(np.quantile(values, 0.05))
    diagnostics["exposure_median"] = float(np.median(values))
    diagnostics["exposure_q95"] = float(np.quantile(values, 0.95))
    return diagnostics


def _adjustment_matrix(
    observed: dict[str, np.ndarray], rows: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            observed["X"][rows],
            np.log1p(observed["degree_unweighted"][rows]),
            observed["assignment_saturation"][rows],
        ]
    )


def _discrete_aipw(
    observed: dict[str, np.ndarray],
    g: np.ndarray,
    outcome_types: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    clusters = np.unique(observed["cluster_id"])
    folds = KFold(n_splits=3, shuffle=True, random_state=seed + 17)
    h = (g >= 0.5).astype(int)
    joint = 2 * observed["A"].astype(int) + h
    n, m = observed["Y"].shape
    phi = np.empty((2, n, m))
    weights_used: list[float] = []
    for train_positions, validation_positions in folds.split(clusters):
        train_clusters = clusters[train_positions]
        validation_clusters = clusters[validation_positions]
        train = np.flatnonzero(np.isin(observed["cluster_id"], train_clusters))
        validation = np.flatnonzero(
            np.isin(observed["cluster_id"], validation_clusters)
        )
        z_train = _adjustment_matrix(observed, train)
        z_validation = _adjustment_matrix(observed, validation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            propensity = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0),
            ).fit(z_train, joint[train])
        probabilities_raw = propensity.predict_proba(z_validation)
        probabilities = np.full((len(validation), 4), 0.02)
        for position, label in enumerate(propensity[-1].classes_):
            probabilities[:, int(label)] = probabilities_raw[:, position]
        probabilities = np.clip(probabilities, 0.02, 0.98)
        for h_value in [0, 1]:
            condition = h_value  # own treatment a=0
            condition_train = train[joint[train] == condition]
            for outcome, outcome_type in enumerate(outcome_types):
                model = _fit_one_model(
                    _adjustment_matrix(observed, condition_train),
                    observed["Y"][condition_train, outcome],
                    outcome_type,
                )
                regression = _predict_one(model, z_validation, outcome_type)
                indicator = (joint[validation] == condition).astype(float)
                correction_weight = indicator / probabilities[:, condition]
                phi[h_value, validation, outcome] = regression + correction_weight * (
                    observed["Y"][validation, outcome] - regression
                )
                weights_used.extend(correction_weight[indicator > 0].tolist())
    estimate = np.mean(phi[1] - phi[0], axis=0)
    weights_array = np.asarray(weights_used)
    diagnostics = {
        "aipw_weight_min": float(np.min(weights_array)) if len(weights_array) else np.nan,
        "aipw_weight_median": float(np.median(weights_array)) if len(weights_array) else np.nan,
        "aipw_weight_q95": float(np.quantile(weights_array, 0.95)) if len(weights_array) else np.nan,
        "aipw_weight_max": float(np.max(weights_array)) if len(weights_array) else np.nan,
    }
    return estimate, diagnostics


def _mapping_estimate(
    observed: dict[str, np.ndarray],
    mapping: str,
    outcome_types: list[str],
) -> np.ndarray:
    exposure = compute_exposure(observed["A"], observed["W_observed"], mapping)
    rows = np.arange(len(exposure))
    models = _fit_models(observed, exposure, rows, outcome_types, mode="full")
    estimates = []
    for target in [TARGET_G0, TARGET_G1]:
        estimates.append(
            np.mean(
                _predict_models(
                    models,
                    observed,
                    exposure,
                    rows,
                    outcome_types,
                    a_override=0,
                    g_override=target,
                    mode="full",
                ),
                axis=0,
            )
        )
    return estimates[1] - estimates[0]


def _simple_ase(
    observed: dict[str, np.ndarray],
    exposure: np.ndarray,
    outcome_types: list[str],
    mode: str,
) -> np.ndarray:
    rows = np.arange(len(exposure))
    models = _fit_models(observed, exposure, rows, outcome_types, mode=mode)
    predictions = []
    for target in [TARGET_G0, TARGET_G1]:
        predictions.append(
            np.mean(
                _predict_models(
                    models,
                    observed,
                    exposure,
                    rows,
                    outcome_types,
                    a_override=0,
                    g_override=target,
                    mode=mode,
                ),
                axis=0,
            )
        )
    return predictions[1] - predictions[0]


def _predictive_diagnostics(
    discovery: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    g_discovery: np.ndarray,
    g_evaluation: np.ndarray,
    outcome_types: list[str],
) -> dict[str, float]:
    train_rows = np.arange(len(g_discovery))
    test_rows = np.arange(len(g_evaluation))
    losses: dict[str, float] = {}
    reference = np.mean(discovery["Y"], axis=0)
    for mode, name in [
        ("full", "predictive_loss_with_G"),
        ("no_g", "predictive_loss_without_G"),
        ("naive", "predictive_loss_naive_A_G"),
    ]:
        models = _fit_models(
            discovery, g_discovery, train_rows, outcome_types, mode=mode
        )
        prediction = _predict_models(
            models,
            evaluation,
            g_evaluation,
            test_rows,
            outcome_types,
            mode=mode,
        )
        losses[name] = normalized_joint_loss(
            evaluation["Y"], prediction, reference
        )
    losses["predictive_gain_G"] = (
        losses["predictive_loss_without_G"] - losses["predictive_loss_with_G"]
    )
    return losses


def _policy_metrics(
    pair: InterferencePair,
    fitted_models: list[BaseEstimator],
    g_evaluation: np.ndarray,
    outcome_types: list[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    if pair.metadata["scenario"] != "policy_shift":
        return {
            "policy_value_rmse": np.nan,
            "policy_ordering_accuracy": np.nan,
            "policy_regret": np.nan,
        }
    evaluation = pair.evaluation
    rows = np.arange(len(g_evaluation))
    values = np.empty((2, len(outcome_types)))
    for policy_index, probability in enumerate([0.2, 0.8]):
        draws = []
        for _ in range(30):
            allocation = rng.binomial(1, probability, size=len(rows))
            exposure = compute_exposure(
                allocation,
                evaluation["W_observed"],
                pair.metadata["primary_mapping"],
            )
            draws.append(
                np.mean(
                    _predict_models(
                        fitted_models,
                        evaluation,
                        exposure,
                        rows,
                        outcome_types,
                        a_override=allocation,
                        g_override=exposure,
                        mode="full",
                    ),
                    axis=0,
                )
            )
        values[policy_index] = np.mean(draws, axis=0)
    estimated_effect = values[1] - values[0]
    true_effect = pair.truth["policy_OE"]
    estimated_choice = int(np.mean(values[1]) > np.mean(values[0]))
    true_choice = int(
        np.mean(pair.truth["policy_value"][1])
        > np.mean(pair.truth["policy_value"][0])
    )
    chosen_true = pair.truth["policy_value"][estimated_choice]
    best_true = pair.truth["policy_value"][true_choice]
    return {
        "policy_value_rmse": float(
            np.sqrt(np.mean((estimated_effect - true_effect) ** 2))
        ),
        "policy_ordering_accuracy": float(estimated_choice == true_choice),
        "policy_regret": float(np.mean(best_true - chosen_true)),
    }


def evaluate_pair(
    pair: InterferencePair,
    seed: int,
    n_bootstrap: int = 29,
) -> dict[str, Any]:
    timer = Timer()
    discovery, evaluation = pair.discovery, pair.evaluation
    outcome_types = pair.metadata["outcome_types"]
    primary_mapping = pair.metadata["primary_mapping"]
    g_discovery = compute_exposure(
        discovery["A"], discovery["W_observed"], primary_mapping
    )
    g_evaluation = compute_exposure(
        evaluation["A"], evaluation["W_observed"], primary_mapping
    )
    if not np.allclose(np.diag(discovery["W_observed"]), 0):
        raise ValueError("W_observed diagonal must be zero")
    if np.any(discovery["W_observed"] < 0):
        raise ValueError("W_observed must be nonnegative")
    rng = np.random.default_rng(seed + 59063)
    crossfit = _crossfit_outcome_regression(
        discovery, g_discovery, outcome_types, seed
    )
    bootstrap_ase, bootstrap_curve = _cluster_bootstrap(
        discovery, g_discovery, outcome_types, rng, n_bootstrap
    )
    null_ase, null_curve = _null_cluster_bootstrap(
        discovery,
        g_discovery,
        outcome_types,
        rng,
        n_bootstrap,
        seed,
    )
    support = {
        own_treatment: _support_diagnostics(
            discovery["A"], g_discovery, own_treatment
        )
        for own_treatment in [0, 1]
    }

    global_results: dict[int, tuple[float, float, np.ndarray, np.ndarray]] = {}
    outcome_rejections: dict[int, np.ndarray] = {}
    for own_treatment in [0, 1]:
        global_results[own_treatment] = _global_test(
            crossfit["ASE"][own_treatment],
            null_ase[:, own_treatment],
        )
        outcome_rejections[own_treatment] = _bh_rejections(
            global_results[own_treatment][3]
        )
        if not support[own_treatment]["supported"]:
            statistic, _, standard_error, _ = global_results[own_treatment]
            global_results[own_treatment] = (
                statistic,
                1.0,
                standard_error,
                np.ones_like(standard_error),
            )
            outcome_rejections[own_treatment][:] = False

    primary_a = 0
    estimate = crossfit["ASE"][primary_a]
    truth = pair.truth["ASE"][primary_a]
    uncertainty_se = np.std(
        bootstrap_ase[:, primary_a], axis=0, ddof=1
    )
    lower = estimate - 2.07 * uncertainty_se
    upper = estimate + 2.07 * uncertainty_se
    affected = pair.truth["affected_outcomes"][primary_a].astype(bool)
    selected = outcome_rejections[primary_a]
    outcome_scores = binary_scores(affected, selected)
    errors = estimate - truth
    nonzero = np.abs(truth) > EPS
    scale = pair.truth["outcome_scale"]
    sase_error = estimate / np.maximum(scale, EPS) - pair.truth["SASE"][primary_a]

    estimated_curve = crossfit["curve"][primary_a]
    true_curve = pair.truth["curve"][primary_a]
    curve_error = estimated_curve - true_curve
    curve_standard_error = np.std(
        bootstrap_curve[:, primary_a], axis=0, ddof=1
    )
    centered_curve = estimated_curve - np.mean(
        estimated_curve, axis=0, keepdims=True
    )
    functional_statistic = np.mean(centered_curve**2, axis=0)
    null_centered_curve = null_curve[:, primary_a] - np.mean(
        null_curve[:, primary_a], axis=1, keepdims=True
    )
    null_functional_statistic = np.mean(null_centered_curve**2, axis=1)
    functional_p = norm.sf(
        (
            functional_statistic
            - np.mean(null_functional_statistic, axis=0)
        )
        / np.maximum(
            np.std(null_functional_statistic, axis=0, ddof=1), EPS
        )
    )
    functional_selected = _bh_rejections(functional_p)
    null_functional_mean = np.mean(null_functional_statistic, axis=0)
    null_functional_sd = np.maximum(
        np.std(null_functional_statistic, axis=0, ddof=1), EPS
    )
    functional_global_statistic = float(
        np.max(
            (functional_statistic - null_functional_mean)
            / null_functional_sd
        )
    )
    null_functional_max = np.max(
        (null_functional_statistic - null_functional_mean[None, :])
        / null_functional_sd[None, :],
        axis=1,
    )
    functional_global_p = float(
        (1 + np.sum(null_functional_max >= functional_global_statistic))
        / (n_bootstrap + 1)
    )
    if functional_global_p > 0.05:
        functional_selected[:] = False
    if not support[primary_a]["supported"]:
        functional_selected[:] = False
        functional_global_p = 1.0
    functional_truth = pair.truth["functional_affected_outcomes"][
        primary_a
    ].astype(bool)
    functional_scores = binary_scores(functional_truth, functional_selected)
    curve_lower = estimated_curve - 2.07 * curve_standard_error
    curve_upper = estimated_curve + 2.07 * curve_standard_error
    pointwise_curve_coverage = np.mean(
        (true_curve >= curve_lower) & (true_curve <= curve_upper)
    )
    standardized_bootstrap_deviation = np.abs(
        bootstrap_curve[:, primary_a] - estimated_curve[None, :, :]
    ) / np.maximum(curve_standard_error[None, :, :], EPS)
    simultaneous_critical = np.quantile(
        np.max(standardized_bootstrap_deviation, axis=1), 0.95, axis=0
    )
    simultaneous_coverage = np.mean(
        np.all(
            np.abs(curve_error)
            <= simultaneous_critical[None, :] * np.maximum(curve_standard_error, EPS),
            axis=0,
        )
    )

    estimated_ade = crossfit["mu"][1, 1] - crossfit["mu"][0, 1]
    true_ade = pair.truth["ADE"][1]
    estimated_interaction = crossfit["ASE"][1] - crossfit["ASE"][0]
    true_interaction = pair.truth["INT"]
    bootstrap_interaction = bootstrap_ase[:, 1] - bootstrap_ase[:, 0]
    interaction_se = np.std(bootstrap_interaction, axis=0, ddof=1)
    interaction_lower = estimated_interaction - 2.07 * interaction_se
    interaction_upper = estimated_interaction + 2.07 * interaction_se

    aipw_ase, aipw_diagnostics = _discrete_aipw(
        discovery, g_discovery, outcome_types, seed
    )
    mapping_estimates = {
        mapping: _mapping_estimate(discovery, mapping, outcome_types)
        for mapping in pair.metadata["sensitivity_mappings"]
    }
    mapping_matrix = np.vstack(list(mapping_estimates.values()))
    true_sign = np.sign(truth)
    mapping_sign_stability = np.mean(
        np.sign(mapping_matrix[:, nonzero]) == true_sign[None, nonzero]
    ) if np.any(nonzero) else np.nan
    sensitivity_range = np.mean(
        np.max(mapping_matrix, axis=0) - np.min(mapping_matrix, axis=0)
    )
    primary_full_ase = _simple_ase(
        discovery, g_discovery, outcome_types, mode="full"
    )
    naive_ase = _simple_ase(
        discovery, g_discovery, outcome_types, mode="naive"
    )
    predictive = _predictive_diagnostics(
        discovery,
        evaluation,
        g_discovery,
        g_evaluation,
        outcome_types,
    )
    rows = np.arange(len(g_discovery))
    final_models = _fit_models(
        discovery, g_discovery, rows, outcome_types, mode="full"
    )
    policy = _policy_metrics(
        pair, final_models, g_evaluation, outcome_types, rng
    )

    true_w = pair.truth["W_true_discovery"]
    observed_w = discovery["W_observed"]
    true_edges = np.triu(true_w > 0, 1)
    observed_edges = np.triu(observed_w > 0, 1)
    edge_disagreement = safe_div(
        float(np.sum(true_edges != observed_edges)),
        float(np.sum(true_edges | observed_edges)),
        0.0,
    )
    exposure_correlation = float(
        np.corrcoef(g_discovery, pair.truth["G_true_discovery"])[0, 1]
    )
    degree = discovery["degree_unweighted"]
    cluster_sizes = np.bincount(discovery["cluster_id"])
    statistic, global_p, _, outcome_p = global_results[primary_a]
    scenario = pair.metadata["scenario"]
    false_interference = float(
        not np.any(affected) and global_p <= 0.05
    )
    result: dict[str, Any] = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "status": "ok",
        "n_rows": float(len(g_discovery)),
        "n_clusters": float(len(np.unique(discovery["cluster_id"]))),
        "treatment_prevalence": float(np.mean(discovery["A"])),
        "degree_mean": float(np.mean(degree)),
        "degree_q95": float(np.quantile(degree, 0.95)),
        "cluster_size_min": float(np.min(cluster_sizes)),
        "cluster_size_max": float(np.max(cluster_sizes)),
        "exposure_mean": float(np.mean(g_discovery)),
        "exposure_sd": float(np.std(g_discovery)),
        "target_supported_a0": support[0]["supported"],
        "target_supported_a1": support[1]["supported"],
        "ess_a0_g0": support[0]["ess_g0"],
        "ess_a0_g1": support[0]["ess_g1"],
        "ess_a1_g0": support[1]["ess_g0"],
        "ess_a1_g1": support[1]["ess_g1"],
        "raw_count_a0_g0": support[0]["raw_g0"],
        "raw_count_a0_g1": support[0]["raw_g1"],
        "global_statistic_a0": statistic,
        "global_p_a0": global_p,
        "global_reject_001_a0": float(global_p <= 0.01),
        "global_reject_005_a0": float(global_p <= 0.05),
        "global_reject_010_a0": float(global_p <= 0.10),
        "global_p_a1": global_results[1][1],
        "global_reject_005_a1": float(global_results[1][1] <= 0.05),
        "outcome_precision": outcome_scores["precision"],
        "outcome_recall": outcome_scores["recall"],
        "outcome_f1": outcome_scores["f1"],
        "outcome_fdr": outcome_scores["fdr"],
        "affected_set_exact_recovery": float(np.array_equal(affected, selected)),
        "familywise_false_outcome": float(np.any(selected & ~affected)),
        "ASE_bias_mean": float(np.mean(errors)),
        "ASE_mae_mean": float(np.mean(np.abs(errors))),
        "ASE_rmse": float(np.sqrt(np.mean(errors**2))),
        "ASE_normalized_rmse": float(
            np.sqrt(np.mean((errors / np.maximum(scale, EPS)) ** 2))
        ),
        "SASE_bias_mean": float(np.mean(sase_error)),
        "SASE_rmse": float(np.sqrt(np.mean(sase_error**2))),
        "ASE_sign_accuracy": float(
            np.mean(np.sign(estimate[nonzero]) == np.sign(truth[nonzero]))
        )
        if np.any(nonzero)
        else np.nan,
        "ASE_ci_coverage": float(np.mean((truth >= lower) & (truth <= upper))),
        "ASE_ci_width": float(np.mean(upper - lower)),
        "null_interval_coverage_zero": float(
            np.mean((lower[~nonzero] <= 0) & (upper[~nonzero] >= 0))
        )
        if np.any(~nonzero)
        else np.nan,
        "functional_precision": functional_scores["precision"],
        "functional_recall": functional_scores["recall"],
        "functional_f1": functional_scores["f1"],
        "functional_any_detection": float(np.any(functional_selected)),
        "functional_global_p": functional_global_p,
        "dose_response_ISE": float(np.mean(curve_error**2)),
        "dose_response_IAE": float(np.mean(np.abs(curve_error))),
        "dose_response_max_error": float(np.max(np.abs(curve_error))),
        "dose_response_pointwise_coverage": float(pointwise_curve_coverage),
        "dose_response_simultaneous_coverage": float(simultaneous_coverage),
        "ADE_bias_mean": float(np.mean(estimated_ade - true_ade)),
        "ADE_rmse": float(np.sqrt(np.mean((estimated_ade - true_ade) ** 2))),
        "INT_bias_mean": float(np.mean(estimated_interaction - true_interaction)),
        "INT_rmse": float(
            np.sqrt(np.mean((estimated_interaction - true_interaction) ** 2))
        ),
        "INT_ci_coverage": float(
            np.mean(
                (true_interaction >= interaction_lower)
                & (true_interaction <= interaction_upper)
            )
        ),
        "interaction_detection": float(
            np.any(
                (interaction_lower > 0) | (interaction_upper < 0)
            )
        ),
        "aipw_discrete_low_high_mean": float(np.mean(aipw_ase)),
        "outcome_regression_ASE_rmse": float(
            np.sqrt(np.mean((primary_full_ase - truth) ** 2))
        ),
        "naive_A_G_ASE_rmse": float(
            np.sqrt(np.mean((naive_ase - truth) ** 2))
        ),
        "oracle_ASE_rmse": 0.0,
        "mapping_sign_stability": float(mapping_sign_stability),
        "mapping_sensitivity_range": float(sensitivity_range),
        "exposure_mapping_correlation": exposure_correlation,
        "relation_edge_disagreement_rate": edge_disagreement,
        "false_interference": false_interference,
        "false_interference_complete_null": (
            false_interference if scenario == "complete_null" else np.nan
        ),
        "false_interference_direct_only": (
            false_interference if scenario == "direct_effect_null" else np.nan
        ),
        "false_interference_shared_outcome_shock": (
            false_interference
            if scenario == "shared_outcome_shock"
            else np.nan
        ),
        "false_interference_observed_homophily": (
            false_interference if scenario == "observed_homophily" else np.nan
        ),
        "bootstrap_ASE_sd": float(
            np.mean(np.std(bootstrap_ase[:, primary_a], axis=0, ddof=1))
        ),
        "failed_or_singular_fits": 0.0,
        "runtime_seconds": timer.seconds,
    }
    for outcome in range(len(outcome_types)):
        result[f"ASE_estimate_Y{outcome + 1}"] = float(estimate[outcome])
        result[f"ASE_truth_Y{outcome + 1}"] = float(truth[outcome])
        result[f"ASE_error_Y{outcome + 1}"] = float(errors[outcome])
        result[f"ASE_p_Y{outcome + 1}"] = float(outcome_p[outcome])
        result[f"ASE_detected_Y{outcome + 1}"] = float(selected[outcome])
        result[f"ASE_coverage_Y{outcome + 1}"] = float(
            lower[outcome] <= truth[outcome] <= upper[outcome]
        )
    for mapping, mapping_estimate in mapping_estimates.items():
        result[f"ASE_mean_{mapping}_mapping"] = float(np.mean(mapping_estimate))
        result[f"ASE_rmse_{mapping}_mapping"] = float(
            np.sqrt(np.mean((mapping_estimate - truth) ** 2))
        )
    result.update(aipw_diagnostics)
    result.update(predictive)
    result.update(policy)
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
