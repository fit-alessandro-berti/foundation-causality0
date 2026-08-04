from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import (
    DESIGN_RANDOMIZED,
    Episode,
    ROLE_COVARIATE,
    ROLE_EXPOSURE,
    ROLE_OUTCOME,
    ROLE_PREDICTOR,
    ROLE_TREATMENT,
    TASK_ATE,
    TASK_INTERFERENCE,
)


EXPERT_NAMES = (
    "linear",
    "orthogonal",
    "spline",
    "interaction",
    "piecewise",
    "design_or_null",
)


@dataclass
class ExpertResult:
    estimates: np.ndarray
    standard_errors: np.ndarray
    mask: np.ndarray
    nuisance: dict[str, float]


def _columns(ep: Episode, role: int) -> np.ndarray:
    return np.flatnonzero(ep.roles == role)


def _covariates(ep: Episode) -> np.ndarray:
    return np.flatnonzero(
        (ep.roles == ROLE_COVARIATE) | (ep.roles == ROLE_PREDICTOR)
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float = 1e-3) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    gram = design.T @ design
    ridge = np.eye(gram.shape[0]) * penalty
    ridge[0, 0] = 0.0
    system = gram + ridge
    try:
        return np.linalg.solve(system, design.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, design.T @ y, rcond=None)[0]


def _predict(x: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficient


def _effect_se(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 1.0
    return float(max(np.std(scores, ddof=1) / np.sqrt(len(scores)), 1e-3))


def _impute(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    means = np.nanmean(values, axis=0)
    means = np.nan_to_num(means)
    rows, columns = np.nonzero(np.isnan(values))
    values[rows, columns] = means[columns]
    return values


def _folds(
    x: np.ndarray,
    groups: np.ndarray | None = None,
    row_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic two-fold assignment independent of presentation order.

    ``row_ids`` are preferred for independent observations.  Network episodes
    instead provide analysis-cluster identifiers through ``groups`` so that a
    connected cluster is never split across nuisance-training folds.  The
    canonicalized-covariate fallback is deliberately invariant to both row and
    covariate permutations; it is used only by legacy inputs without stable
    identifiers.
    """
    if groups is not None:
        unique = np.unique(groups)
        if len(unique) < 2:
            return _folds(x, row_ids=row_ids)
        mapping = {value: index % 2 for index, value in enumerate(unique)}
        return np.asarray([mapping[value] for value in groups], dtype=int)

    if row_ids is not None and len(row_ids) == len(x) and len(np.unique(row_ids)) == len(x):
        order = np.argsort(row_ids, kind="stable")
    elif x.shape[1]:
        canonical_rows = np.sort(np.nan_to_num(x), axis=1)
        order = np.lexsort(np.flipud(canonical_rows.T))
    else:
        order = np.arange(len(x))
    folds = np.empty(len(x), dtype=int)
    folds[order] = np.arange(len(x)) % 2
    return folds


def _connected_components(adjacency: np.ndarray) -> np.ndarray:
    labels = np.full(len(adjacency), -1, dtype=int)
    component = 0
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adjacency[node] > 0)
            unseen = neighbors[labels[neighbors] < 0]
            labels[unseen] = component
            stack.extend(unseen.tolist())
        component += 1
    return labels


def _ate_experts(ep: Episode) -> ExpertResult:
    values = _impute(ep.values)
    cov = _covariates(ep)
    treatment = int(_columns(ep, ROLE_TREATMENT)[0])
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    x, a, y = values[:, cov], values[:, treatment], values[:, outcome]
    n = len(y)

    linear_design = np.column_stack([x, a])
    linear_coef = _ridge_fit(linear_design, y)
    linear = float(linear_coef[-1])
    linear_score = np.full(n, linear)

    row_ids = ep.cluster_ids if len(ep.cluster_ids) == n else None
    folds = _folds(x, row_ids=row_ids)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    propensity = np.zeros(n)
    for fold in range(2):
        train = folds != fold
        test = ~train
        if ep.design == DESIGN_RANDOMIZED:
            propensity[test] = np.clip(a[train].mean(), 0.05, 0.95)
        else:
            propensity_coef = _ridge_fit(x[train], a[train], 0.05)
            propensity[test] = np.clip(_predict(x[test], propensity_coef), 0.05, 0.95)
        nuisance_x = np.column_stack([x[train], a[train], a[train, None] * x[train]])
        nuisance_coef = _ridge_fit(nuisance_x, y[train], 0.02)
        low = np.column_stack([x[test], np.zeros(test.sum()), np.zeros_like(x[test])])
        high = np.column_stack([x[test], np.ones(test.sum()), x[test]])
        mu0[test] = _predict(low, nuisance_coef)
        mu1[test] = _predict(high, nuisance_coef)
    dr_score = (
        mu1
        - mu0
        + a * (y - mu1) / propensity
        - (1.0 - a) * (y - mu0) / (1.0 - propensity)
    )
    orthogonal = float(dr_score.mean())

    spline_x = np.column_stack([x, x**2, np.tanh(x)])
    spline_design = np.column_stack([spline_x, a])
    spline_coef = _ridge_fit(spline_design, y, 0.03)
    spline = float(spline_coef[-1])

    interaction_x = np.column_stack([x, x**2, a, a[:, None] * x])
    interaction_coef = _ridge_fit(interaction_x, y, 0.03)
    low = np.column_stack([x, x**2, np.zeros(n), np.zeros_like(x)])
    high = np.column_stack([x, x**2, np.ones(n), x])
    interaction_scores = _predict(high, interaction_coef) - _predict(
        low, interaction_coef
    )
    interaction = float(interaction_scores.mean())

    # Use the same hinge candidates for every eligible covariate.  Restricting
    # the expansion to the first columns made the estimate depend on arbitrary
    # presentation order.
    hinges = np.maximum(x[:, :, None] - np.array([-0.5, 0.0, 0.5]), 0)
    hinges = hinges.reshape(n, -1)
    piece_design = np.column_stack([x, hinges, a, a[:, None] * hinges])
    piece_coef = _ridge_fit(piece_design, y, 0.05)
    low = np.column_stack([x, hinges, np.zeros(n), np.zeros_like(hinges)])
    high = np.column_stack([x, hinges, np.ones(n), hinges])
    piece_scores = _predict(high, piece_coef) - _predict(low, piece_coef)
    piecewise = float(piece_scores.mean())

    treated = a > 0.5
    if treated.any() and (~treated).any():
        design_scores = np.r_[
            y[treated] - y[treated].mean() + y[treated].mean() - y[~treated].mean(),
            y[~treated] - y[~treated].mean() + y[treated].mean() - y[~treated].mean(),
        ]
        design = float(y[treated].mean() - y[~treated].mean())
        design_se = _effect_se(design_scores)
    else:
        design, design_se = linear, 1.0

    estimates = np.asarray(
        [linear, orthogonal, spline, interaction, piecewise, design], dtype=np.float32
    )
    ses = np.asarray(
        [
            _effect_se(y - _predict(linear_design, linear_coef) + linear),
            _effect_se(dr_score),
            _effect_se(y - _predict(spline_design, spline_coef) + spline),
            _effect_se(interaction_scores),
            _effect_se(piece_scores),
            design_se,
        ],
        dtype=np.float32,
    )
    mask = np.ones(len(EXPERT_NAMES), dtype=bool)
    mask[-1] = ep.design == DESIGN_RANDOMIZED
    nuisance = {
        "mu0_oof_rmse": float(np.sqrt(np.mean((y[a < 0.5] - mu0[a < 0.5]) ** 2)))
        if (a < 0.5).any()
        else np.nan,
        "mu1_oof_rmse": float(np.sqrt(np.mean((y[a > 0.5] - mu1[a > 0.5]) ** 2)))
        if (a > 0.5).any()
        else np.nan,
        "propensity_brier": float(np.mean((a - propensity) ** 2)),
        "propensity_clip_fraction": float(np.mean((propensity <= 0.051) | (propensity >= 0.949))),
        "orthogonal_score_sd": float(np.std(dr_score)),
    }
    return ExpertResult(estimates, ses, mask, nuisance)


def _network_experts(ep: Episode) -> ExpertResult:
    values = _impute(ep.values)
    cov = _covariates(ep)
    treatment = int(_columns(ep, ROLE_TREATMENT)[0])
    exposure = int(_columns(ep, ROLE_EXPOSURE)[0])
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    x, a, g, y = (
        values[:, cov],
        values[:, treatment],
        values[:, exposure],
        values[:, outcome],
    )
    n = len(y)
    g0, g1 = ep.query_exposure

    linear_x = np.column_stack([x, a, g, a * g])
    linear_coef = _ridge_fit(linear_x, y, 0.01)
    low = np.column_stack([x, a, np.full(n, g0), a * g0])
    high = np.column_stack([x, a, np.full(n, g1), a * g1])
    linear_scores = _predict(high, linear_coef) - _predict(low, linear_coef)
    linear = float(linear_scores.mean())

    clusters = (
        ep.cluster_ids
        if len(ep.cluster_ids) == n
        else _connected_components(ep.adjacency)
    )
    folds = _folds(x, clusters)
    oof = np.zeros(n)
    fold_effects = []
    for fold in range(2):
        train = folds != fold
        test = ~train
        coef = _ridge_fit(linear_x[train], y[train], 0.02)
        oof[test] = _predict(linear_x[test], coef)
        fold_effects.append(
            float(
                (
                    _predict(high[test], coef) - _predict(low[test], coef)
                ).mean()
            )
        )
    orthogonal = float(np.mean(fold_effects))

    spline_basis = np.column_stack(
        [g, g**2, np.maximum(g - 0.25, 0), np.maximum(g - 0.5, 0), np.maximum(g - 0.75, 0)]
    )
    spline_x = np.column_stack([x, a, spline_basis, a[:, None] * spline_basis])
    spline_coef = _ridge_fit(spline_x, y, 0.03)

    def spline_at(level: float) -> np.ndarray:
        basis = np.tile(
            np.asarray(
                [
                    level,
                    level**2,
                    max(level - 0.25, 0),
                    max(level - 0.5, 0),
                    max(level - 0.75, 0),
                ]
            ),
            (n, 1),
        )
        return np.column_stack([x, a, basis, a[:, None] * basis])

    spline_scores = _predict(spline_at(g1), spline_coef) - _predict(
        spline_at(g0), spline_coef
    )
    spline = float(spline_scores.mean())

    interaction_x = np.column_stack([x, a, g, a * g, g**2, a * g**2])
    interaction_coef = _ridge_fit(interaction_x, y, 0.03)

    def interaction_at(level: float) -> np.ndarray:
        return np.column_stack(
            [x, a, np.full(n, level), a * level, np.full(n, level**2), a * level**2]
        )

    interaction_scores = _predict(interaction_at(g1), interaction_coef) - _predict(
        interaction_at(g0), interaction_coef
    )
    interaction = float(interaction_scores.mean())

    best_bic = np.inf
    best_threshold = 0.5
    best_coef: np.ndarray | None = None
    for threshold in np.unique(np.quantile(g, [0.2, 0.35, 0.5, 0.65, 0.8])):
        indicator = (g >= threshold).astype(float)
        candidate_x = np.column_stack([x, a, indicator, a * indicator, g])
        coefficient = _ridge_fit(candidate_x, y, 0.02)
        residual = y - _predict(candidate_x, coefficient)
        bic = n * np.log(max(np.mean(residual**2), 1e-8)) + len(coefficient) * np.log(n)
        if bic < best_bic:
            best_bic, best_threshold, best_coef = bic, float(threshold), coefficient
    assert best_coef is not None

    def piece_at(level: float) -> np.ndarray:
        indicator = float(level >= best_threshold)
        return np.column_stack(
            [x, a, np.full(n, indicator), a * indicator, np.full(n, level)]
        )

    piece_scores = _predict(piece_at(g1), best_coef) - _predict(piece_at(g0), best_coef)
    piecewise = float(piece_scores.mean())
    estimates = np.asarray(
        [linear, orthogonal, spline, interaction, piecewise, 0.0], dtype=np.float32
    )
    ses = np.asarray(
        [
            _effect_se(linear_scores),
            max(float(np.std(fold_effects)), 1e-3),
            _effect_se(spline_scores),
            _effect_se(interaction_scores),
            _effect_se(piece_scores),
            max(float(np.std(y) / np.sqrt(n)), 1e-3),
        ],
        dtype=np.float32,
    )
    nuisance = {
        "network_response_oof_rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
        "cluster_fold_effect_sd": float(np.std(fold_effects)),
        "exposure_support_p05": float(np.quantile(g, 0.05)),
        "exposure_support_p95": float(np.quantile(g, 0.95)),
    }
    return ExpertResult(estimates, ses, np.ones(len(EXPERT_NAMES), dtype=bool), nuisance)


def estimator_experts(ep: Episode) -> ExpertResult:
    if ep.task == TASK_ATE:
        return _ate_experts(ep)
    if ep.task == TASK_INTERFERENCE:
        return _network_experts(ep)
    return ExpertResult(
        np.zeros(len(EXPERT_NAMES), dtype=np.float32),
        np.ones(len(EXPERT_NAMES), dtype=np.float32),
        np.zeros(len(EXPERT_NAMES), dtype=bool),
        {},
    )
