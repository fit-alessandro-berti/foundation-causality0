from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.decomposition import PCA

from .common import (
    EPS,
    Timer,
    correlation_matrix,
    graph_scores,
    match_components,
    safe_div,
    save_dataset,
    set_scores,
    standardize_apply,
    standardize_fit,
    summarize_records,
)
from .latent_variable import (
    fit_sparse_pls,
    hard_groups,
    sparse_transform,
    tune_sparse_pls,
)


METHOD = "latent_variable_determination2"
SCENARIOS = [
    "empty_graph",
    "chain",
    "hub",
    "disconnected_graph",
    "weak_edges",
    "dense_near_limit",
    "small_sample",
    "ill_conditioned_covariance",
    "weak_measurement",
    "unequal_group_quality",
    "cross_loadings",
    "grouping_error",
    "outcome_irrelevant_node",
    "hidden_confounding",
    "temporal_data",
]


@dataclass
class GraphPair:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _edge_set(scenario: str, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if scenario == "empty_graph":
        return []
    if scenario in {"chain", "weak_edges", "small_sample", "ill_conditioned_covariance", "weak_measurement", "unequal_group_quality", "cross_loadings", "grouping_error", "outcome_irrelevant_node", "hidden_confounding", "temporal_data"}:
        return [(i, i + 1) for i in range(k - 1)]
    if scenario == "hub":
        return [(0, i) for i in range(1, k)]
    if scenario == "disconnected_graph":
        return [(0, 1), (1, 2), (3, 4), (4, 5)]
    if scenario == "dense_near_limit":
        return [(i, j) for i in range(k) for j in range(i + 1, k) if rng.random() < 0.5]
    return [(i, i + 1) for i in range(k - 1)]


def _precision_from_edges(
    k: int, edges: list[tuple[int, int]], weight: float, delta: float, rng: np.random.Generator
) -> np.ndarray:
    theta = np.zeros((k, k))
    for i, j in edges:
        value = weight * rng.choice([-1.0, 1.0])
        theta[i, j] = theta[j, i] = value
    for i in range(k):
        theta[i, i] = delta + np.sum(np.abs(theta[i]))
    return theta


def _partial_from_precision(theta: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.outer(np.diag(theta), np.diag(theta)))
    partial = -theta / diagonal
    np.fill_diagonal(partial, 1.0)
    return partial


def generate_pair(
    scenario: str, seed: int, n_discovery: int | None = None, n_evaluation: int = 700
) -> GraphPair:
    rng = np.random.default_rng(seed + 200_003)
    k, n_outcomes = 6, 6
    n_discovery = n_discovery or (120 if scenario == "small_sample" else 600)
    group_sizes = [6] * k
    measurement_sd_by_group = np.full(k, 0.55)
    if scenario == "weak_measurement":
        measurement_sd_by_group[:] = 1.35
    if scenario == "unequal_group_quality":
        group_sizes[-1] = 3
        measurement_sd_by_group[-1] = 1.7
    n_noise = 12
    n_features = sum(group_sizes) + n_noise
    edges = _edge_set(scenario, k, rng)
    weight = 0.12 if scenario == "weak_edges" else (0.72 if scenario == "ill_conditioned_covariance" else 0.38)
    delta = 0.12 if scenario == "ill_conditioned_covariance" else 0.8
    theta = _precision_from_edges(k, edges, weight, delta, rng)
    covariance = np.linalg.inv(theta)
    scales = np.sqrt(np.diag(covariance))
    covariance = covariance / np.outer(scales, scales)
    theta = np.linalg.inv(covariance)
    partial = _partial_from_precision(theta)
    loading = np.zeros((n_features, k))
    true_group = np.full(n_features, -1, dtype=int)
    feature_sd = np.full(n_features, 0.55)
    start = 0
    for latent, size in enumerate(group_sizes):
        stop = start + size
        loading[start:stop, latent] = rng.choice([-1.0, 1.0], size=size) * rng.uniform(
            0.7, 1.0, size=size
        )
        true_group[start:stop] = latent
        feature_sd[start:stop] = measurement_sd_by_group[latent]
        start = stop
    cross_loading_features: list[int] = []
    if scenario == "cross_loadings":
        for feature in rng.choice(sum(group_sizes), size=8, replace=False):
            other = (true_group[feature] + rng.integers(1, k)) % k
            loading[feature, other] = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.55)
            cross_loading_features.append(int(feature))
    gamma = np.eye(k) * (0.38 if scenario == "grouping_error" else 1.0)
    gamma += np.roll(np.eye(k), 1, axis=0) * (0.35 if scenario == "grouping_error" else 0.18)
    relevant = np.arange(k - 1 if scenario == "outcome_irrelevant_node" else k, dtype=int)
    if scenario == "outcome_irrelevant_node":
        gamma[:, -1] = 0.0
    lag_matrix = np.zeros((k, k))
    if scenario == "temporal_data":
        lag_matrix = np.diag(np.linspace(0.25, 0.55, k))
        lag_matrix[1:, :-1] += np.eye(k - 1) * 0.12
    b_dag = np.zeros((k, k))
    hidden_loading = np.zeros(k)
    if scenario == "hidden_confounding":
        b_dag[1, 0] = 0.5
        b_dag[3, 2] = -0.45
        hidden_loading[[0, 2]] = [0.8, -0.7]

    def sample_z(n_rows: int) -> np.ndarray:
        if scenario == "temporal_data":
            innovations_cov = covariance * 0.55
            z = np.zeros((n_rows + 100, k))
            for t in range(1, len(z)):
                z[t] = lag_matrix @ z[t - 1] + rng.multivariate_normal(np.zeros(k), innovations_cov)
            return z[100:]
        z = rng.multivariate_normal(np.zeros(k), covariance, size=n_rows)
        if scenario == "hidden_confounding":
            hidden = rng.normal(size=(n_rows, 1))
            z = z + hidden @ hidden_loading[None, :]
        return z

    def sample(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = sample_z(n_rows)
        x = z @ loading.T + rng.normal(size=(n_rows, n_features)) * feature_sd
        y = z @ gamma.T + rng.normal(scale=0.7, size=(n_rows, n_outcomes))
        return x, y, z

    xd, yd, zd = sample(n_discovery)
    xe, ye, ze = sample(n_evaluation)
    metadata = {
        "method": METHOD,
        "scenario": scenario,
        "seed": seed,
        "feature_names": [f"X{j + 1}" for j in range(n_features)],
        "outcome_names": [f"Y{j + 1}" for j in range(n_outcomes)],
        "relevant_latents": relevant.tolist(),
        "graph_family": scenario,
        "cross_loading_features": cross_loading_features,
        "hidden_variables": [f"H->{[0, 2]}"] if scenario == "hidden_confounding" else [],
        "primary_target": "undirected_conditional_dependence",
        "extension_target": (
            "lagged_directed_network"
            if scenario == "temporal_data"
            else ("PAG_with_hidden_confounding" if scenario == "hidden_confounding" else None)
        ),
    }
    truth = {
        "Z_discovery": zd,
        "Z_evaluation": ze,
        "Lambda": loading,
        "Gamma": gamma,
        "true_group": true_group,
        "Theta": theta,
        "partial_correlation": partial,
        "edge_set": np.asarray(edges, dtype=int).reshape(-1, 2),
        "relevant_latents": relevant,
        "B_dag": b_dag,
        "hidden_loading": hidden_loading,
        "lag_matrix": lag_matrix,
    }
    return GraphPair({"X": xd, "Y": yd}, {"X": xe, "Y": ye}, truth, metadata)


def _gaussian_nll(scores: np.ndarray, precision: np.ndarray) -> float:
    covariance = np.cov(scores, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]])
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf
    return float(np.trace(covariance @ precision) - logdet)


def fit_network(train_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    if train_scores.shape[1] < 2:
        partial = np.eye(train_scores.shape[1])
        return {
            "partial": partial,
            "alpha": np.nan,
            "heldout_nll": np.nan,
            "train_scores": train_scores,
            "test_scores": test_scores,
        }
    mean, scale = standardize_fit(train_scores)
    train = standardize_apply(train_scores, mean, scale)
    test = standardize_apply(test_scores, mean, scale)
    split = max(train.shape[1] + 5, int(0.75 * len(train)))
    fit, validation = train[:split], train[split:]
    candidates: list[tuple[float, float]] = []
    for alpha in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GraphicalLasso(alpha=alpha, max_iter=300, tol=1e-4).fit(fit)
            candidates.append((_gaussian_nll(validation, model.precision_), alpha))
        except Exception:
            continue
    alpha = min(candidates)[1] if candidates else 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GraphicalLasso(alpha=alpha, max_iter=500, tol=1e-4).fit(train)
    partial = _partial_from_precision(model.precision_)
    return {
        "partial": partial,
        "alpha": alpha,
        "heldout_nll": _gaussian_nll(test, model.precision_),
        "train_scores": train,
        "test_scores": test,
    }


def _oracle_group_scores(
    x_train: np.ndarray, x_test: np.ndarray, true_group: np.ndarray, relevant: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    train_scores, test_scores, supports = [], [], []
    for latent in relevant:
        features = np.flatnonzero(true_group == latent)
        pca = PCA(n_components=1).fit(x_train[:, features])
        train_scores.append(pca.transform(x_train[:, features])[:, 0])
        test_scores.append(pca.transform(x_test[:, features])[:, 0])
        supports.append(set(features.tolist()))
    return np.column_stack(train_scores), np.column_stack(test_scores), supports


def _estimated_group_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[set[int]], np.ndarray]:
    groups = hard_groups(weights)
    train_scores, test_scores, supports = [], [], []
    for component in range(weights.shape[1]):
        features = np.flatnonzero(groups == component)
        if not len(features):
            continue
        pca = PCA(n_components=1).fit(x_train[:, features])
        train_scores.append(pca.transform(x_train[:, features])[:, 0])
        test_scores.append(pca.transform(x_test[:, features])[:, 0])
        supports.append(set(features.tolist()))
    if not train_scores:
        return np.empty((len(x_train), 0)), np.empty((len(x_test), 0)), [], groups
    return np.column_stack(train_scores), np.column_stack(test_scores), supports, groups


def _align_scores_to_truth(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    supports: list[set[int]],
    true_group: np.ndarray,
    relevant: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[float]]:
    if not supports or not len(relevant):
        return (
            np.empty((len(train_scores), 0)),
            np.empty((len(test_scores), 0)),
            [],
            [],
        )
    weights = np.zeros((len(supports), len(relevant)))
    for i, support in enumerate(supports):
        for j, latent in enumerate(relevant):
            truth = set(np.flatnonzero(true_group == latent).tolist())
            union = support | truth
            weights[i, j] = len(support & truth) / len(union) if union else 1.0
    matching = match_components(weights)
    matching = sorted(matching, key=lambda pair: pair[1])
    aligned_train, aligned_test, correlations = [], [], []
    for estimated, truth_position in matching:
        latent = int(relevant[truth_position])
        correlation = float(
            correlation_matrix(train_scores[:, [estimated]], z_train[:, [latent]])[0, 0]
        )
        sign = 1.0 if correlation >= 0 else -1.0
        aligned_train.append(train_scores[:, estimated] * sign)
        aligned_test.append(test_scores[:, estimated] * sign)
        correlations.append(abs(correlation))
    return (
        np.column_stack(aligned_train),
        np.column_stack(aligned_test),
        matching,
        correlations,
    )


def _embed_partial(
    estimated_partial: np.ndarray, matching: list[tuple[int, int]], k_true: int
) -> np.ndarray:
    embedded = np.zeros((k_true, k_true))
    np.fill_diagonal(embedded, 1.0)
    truth_positions = [truth_position for _, truth_position in matching]
    for i, truth_i in enumerate(truth_positions):
        for j, truth_j in enumerate(truth_positions):
            embedded[truth_i, truth_j] = estimated_partial[i, j]
    return embedded


def _topology_metrics(true_partial: np.ndarray, estimated_partial: np.ndarray, threshold: float) -> dict[str, float]:
    true_graph = nx.from_numpy_array((np.abs(true_partial) > EPS).astype(int))
    estimated_graph = nx.from_numpy_array((np.abs(estimated_partial) >= threshold).astype(int))
    true_graph.remove_edges_from(nx.selfloop_edges(true_graph))
    estimated_graph.remove_edges_from(nx.selfloop_edges(estimated_graph))
    true_degree = np.asarray([degree for _, degree in true_graph.degree()], dtype=float)
    estimated_degree = np.asarray([degree for _, degree in estimated_graph.degree()], dtype=float)
    degree_corr = (
        float(np.corrcoef(true_degree, estimated_degree)[0, 1])
        if np.std(true_degree) > EPS and np.std(estimated_degree) > EPS
        else np.nan
    )
    components = list(nx.connected_components(true_graph))
    spurious_bridges = 0
    for i, component in enumerate(components):
        for other in components[i + 1 :]:
            spurious_bridges += sum(
                estimated_graph.has_edge(a, b) for a in component for b in other
            )
    return {
        "degree_correlation": degree_corr,
        "connected_component_count_error": float(
            abs(nx.number_connected_components(true_graph) - nx.number_connected_components(estimated_graph))
        ),
        "spurious_bridges": float(spurious_bridges),
    }


def _lagged_extension_scores(z_train: np.ndarray, lag_truth: np.ndarray) -> dict[str, float]:
    if not np.any(lag_truth):
        return {"lag_edge_precision": np.nan, "lag_edge_recall": np.nan, "lag_edge_f1": np.nan}
    coef = np.linalg.lstsq(
        np.column_stack([np.ones(len(z_train) - 1), z_train[:-1]]),
        z_train[1:],
        rcond=None,
    )[0][1:].T
    threshold = 0.08
    truth = np.abs(lag_truth) > EPS
    estimated = np.abs(coef) >= threshold
    scores = set_scores(zip(*np.where(truth)), zip(*np.where(estimated)))
    return {
        "lag_edge_precision": scores["precision"],
        "lag_edge_recall": scores["recall"],
        "lag_edge_f1": scores["f1"],
    }


def _conditional_edge_stability(
    scores: np.ndarray, alpha: float, rng: np.random.Generator, n_bootstrap: int
) -> float:
    if scores.shape[1] < 2 or not np.isfinite(alpha):
        return np.nan
    edge_sets = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(scores), size=len(scores))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GraphicalLasso(alpha=alpha, max_iter=300).fit(scores[idx])
        partial = _partial_from_precision(model.precision_)
        edge_sets.append(
            set(
                (i, j)
                for i in range(partial.shape[0])
                for j in range(i + 1, partial.shape[0])
                if abs(partial[i, j]) >= 0.08
            )
        )
    similarities = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            union = edge_sets[i] | edge_sets[j]
            similarities.append(len(edge_sets[i] & edge_sets[j]) / len(union) if union else 1.0)
    return float(np.mean(similarities)) if similarities else np.nan


def evaluate_pair(pair: GraphPair, seed: int, n_bootstrap: int = 5) -> dict[str, Any]:
    timer = Timer()
    xd_raw, yd_raw = pair.discovery["X"], pair.discovery["Y"]
    xe_raw, ye_raw = pair.evaluation["X"], pair.evaluation["Y"]
    xm, xs = standardize_fit(xd_raw)
    ym, ys = standardize_fit(yd_raw)
    xd, xe = standardize_apply(xd_raw, xm, xs), standardize_apply(xe_raw, xm, xs)
    yd = standardize_apply(yd_raw, ym, ys)
    relevant = pair.truth["relevant_latents"]
    truth_partial = pair.truth["partial_correlation"][np.ix_(relevant, relevant)]
    threshold = 0.08

    level_a = fit_network(
        pair.truth["Z_discovery"][:, relevant], pair.truth["Z_evaluation"][:, relevant]
    )
    level_a_scores = graph_scores(truth_partial, level_a["partial"], threshold)

    b_train, b_test, b_supports = _oracle_group_scores(
        xd, xe, pair.truth["true_group"], relevant
    )
    b_train, b_test, b_matching, b_correlations = _align_scores_to_truth(
        b_train,
        b_test,
        b_supports,
        pair.truth["true_group"],
        relevant,
        pair.truth["Z_discovery"],
        pair.truth["Z_evaluation"],
    )
    level_b = fit_network(b_train, b_test)
    b_embedded = _embed_partial(level_b["partial"], b_matching, len(relevant))
    level_b_scores = graph_scores(truth_partial, b_embedded, threshold)

    components, keep, _ = tune_sparse_pls(xd_raw, yd_raw, max_components=7, seed=seed)
    grouping_model = fit_sparse_pls(xd, yd, components, keep)
    c_train_raw, c_test_raw, c_supports, c_groups = _estimated_group_scores(
        xd, xe, grouping_model["weights"]
    )
    c_train, c_test, c_matching, c_correlations = _align_scores_to_truth(
        c_train_raw,
        c_test_raw,
        c_supports,
        pair.truth["true_group"],
        relevant,
        pair.truth["Z_discovery"],
        pair.truth["Z_evaluation"],
    )
    level_c = fit_network(c_train, c_test)
    c_embedded = _embed_partial(level_c["partial"], c_matching, len(relevant))
    level_c_scores = graph_scores(truth_partial, c_embedded, threshold)
    topology = _topology_metrics(truth_partial, c_embedded, threshold)

    relevant_features = np.flatnonzero(np.isin(pair.truth["true_group"], relevant))
    selected_features = np.flatnonzero(c_groups >= 0)
    support = set_scores(relevant_features.tolist(), selected_features.tolist())
    true_groups_for_signal = pair.truth["true_group"][relevant_features]
    estimated_groups_for_signal = c_groups[relevant_features]
    group_ari = (
        float(__import__("sklearn.metrics").metrics.adjusted_rand_score(true_groups_for_signal, estimated_groups_for_signal))
        if len(relevant_features)
        else np.nan
    )
    stability = _conditional_edge_stability(
        level_c["train_scores"],
        level_c["alpha"],
        np.random.default_rng(seed + 31013),
        n_bootstrap,
    )
    result: dict[str, Any] = {
        "method": METHOD,
        "scenario": pair.metadata["scenario"],
        "seed": seed,
        "status": "ok",
        "K_true_relevant": float(len(relevant)),
        "K_selected": float(grouping_model["n_components"]),
        "node_count_error": float(abs(grouping_model["n_components"] - len(relevant))),
        "node_support_precision": support["precision"],
        "node_support_recall": support["recall"],
        "node_support_f1": support["f1"],
        "group_ari": group_ari,
        "mean_score_correlation": float(np.mean(c_correlations)) if c_correlations else np.nan,
        "measurement_reliability_level_B": float(np.mean(np.square(b_correlations)))
        if b_correlations
        else np.nan,
        "heldout_nll_level_A": level_a["heldout_nll"],
        "heldout_nll_level_B": level_b["heldout_nll"],
        "heldout_nll_level_C": level_c["heldout_nll"],
        "bootstrap_edge_jaccard": stability,
        "runtime_seconds": timer.seconds,
    }
    for prefix, scores in [
        ("level_A", level_a_scores),
        ("level_B", level_b_scores),
        ("level_C", level_c_scores),
    ]:
        for key, value in scores.items():
            result[f"{prefix}_{key}"] = value
    result.update(topology)
    result.update(_lagged_extension_scores(pair.truth["Z_discovery"], pair.truth["lag_matrix"]))
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
