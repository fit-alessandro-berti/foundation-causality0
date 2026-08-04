from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer

from .data import (
    Episode,
    ROLE_COVARIATE,
    ROLE_EXPOSURE,
    ROLE_OUTCOME,
    ROLE_PREDICTOR,
    ROLE_TREATMENT,
    TASK_ATE,
    TASK_INTERFERENCE,
    TASK_REGIME,
)
from .estimators import _folds, estimator_experts


def _columns(ep: Episode, role: int) -> np.ndarray:
    return np.flatnonzero(ep.roles == role)


def _covariates(ep: Episode) -> np.ndarray:
    return np.flatnonzero(
        (ep.roles == ROLE_COVARIATE) | (ep.roles == ROLE_PREDICTOR)
    )


def _complete(values: np.ndarray) -> np.ndarray:
    result = values.astype(float, copy=True)
    means = np.nan_to_num(np.nanmean(result, axis=0))
    rows, columns = np.nonzero(np.isnan(result))
    result[rows, columns] = means[columns]
    return result


def linear_effect(ep: Episode) -> float:
    values = _complete(ep.values)
    cov = _covariates(ep)
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    if ep.task == TASK_ATE:
        treatment = int(_columns(ep, ROLE_TREATMENT)[0])
        model = LinearRegression().fit(values[:, np.r_[cov, treatment]], values[:, outcome])
        return float(model.coef_[-1])
    if ep.task == TASK_INTERFERENCE:
        treatment = int(_columns(ep, ROLE_TREATMENT)[0])
        exposure = int(_columns(ep, ROLE_EXPOSURE)[0])
        a, g = values[:, treatment], values[:, exposure]
        design = np.column_stack([values[:, cov], a, g, a * g])
        model = LinearRegression().fit(design, values[:, outcome])
        coefficient_g = model.coef_[-2]
        coefficient_ag = model.coef_[-1]
        return float(0.5 * (coefficient_g + 0.5 * coefficient_ag))
    return 0.0


def forest_effect(ep: Episode, trees: int = 120) -> float:
    values = _complete(ep.values)
    cov = _covariates(ep)
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    if ep.task == TASK_ATE:
        treatment = int(_columns(ep, ROLE_TREATMENT)[0])
        features = values[:, np.r_[cov, treatment]]
        model = RandomForestRegressor(
            n_estimators=trees, min_samples_leaf=6, max_features=0.8, n_jobs=1, random_state=ep.seed
        ).fit(features, values[:, outcome])
        low, high = features.copy(), features.copy()
        low[:, -1], high[:, -1] = 0.0, 1.0
        return float(np.mean(model.predict(high) - model.predict(low)))
    if ep.task == TASK_INTERFERENCE:
        treatment = int(_columns(ep, ROLE_TREATMENT)[0])
        exposure = int(_columns(ep, ROLE_EXPOSURE)[0])
        features = values[:, np.r_[cov, treatment, exposure]]
        model = RandomForestRegressor(
            n_estimators=trees, min_samples_leaf=6, max_features=0.8, n_jobs=1, random_state=ep.seed
        ).fit(features, values[:, outcome])
        low, high = features.copy(), features.copy()
        low[:, -1], high[:, -1] = 0.25, 0.75
        return float(np.mean(model.predict(high) - model.predict(low)))
    return 0.0


def aipw_effect(ep: Episode, trees: int = 100) -> float:
    if ep.task != TASK_ATE:
        return forest_effect(ep, trees)
    # The online estimator library uses deterministic out-of-fold nuisance
    # predictions, so the reported orthogonal estimate is never in-sample.
    return float(estimator_experts(ep).estimates[1])


def spline_dr_effect(ep: Episode) -> float:
    """Cross-fitted AIPW with flexible spline nuisance models.

    This is a conventional flexible comparator, not part of the pretrained
    router's expert library.  Stable row identifiers keep its sample split
    invariant to row and covariate presentation order.
    """
    if ep.task != TASK_ATE:
        return float(estimator_experts(ep).estimates[1])
    values = _complete(ep.values)
    cov = _covariates(ep)
    treatment = int(_columns(ep, ROLE_TREATMENT)[0])
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    x, a, y = values[:, cov], values[:, treatment], values[:, outcome]
    row_ids = ep.cluster_ids if len(ep.cluster_ids) == len(y) else None
    folds = _folds(x, row_ids=row_ids)
    mu0 = np.zeros(len(y))
    mu1 = np.zeros(len(y))
    propensity = np.zeros(len(y))
    for fold in range(2):
        train = folds != fold
        test = ~train
        spline = SplineTransformer(
            n_knots=4,
            degree=2,
            include_bias=False,
            extrapolation="linear",
        ).fit(x[train])
        train_basis = spline.transform(x[train])
        test_basis = spline.transform(x[test])
        if np.unique(a[train]).size < 2:
            propensity[test] = np.clip(a[train].mean(), 0.05, 0.95)
        elif ep.design == 1:
            propensity[test] = np.clip(a[train].mean(), 0.05, 0.95)
        else:
            propensity_model = LogisticRegression(
                C=0.5,
                max_iter=500,
                random_state=ep.seed,
            ).fit(train_basis, a[train])
            propensity[test] = np.clip(
                propensity_model.predict_proba(test_basis)[:, 1], 0.05, 0.95
            )
        outcome_design = np.column_stack(
            [train_basis, a[train], a[train, None] * train_basis]
        )
        outcome_model = Ridge(alpha=1.0).fit(outcome_design, y[train])
        mu0[test] = outcome_model.predict(
            np.column_stack(
                [test_basis, np.zeros(test.sum()), np.zeros_like(test_basis)]
            )
        )
        mu1[test] = outcome_model.predict(
            np.column_stack([test_basis, np.ones(test.sum()), test_basis])
        )
    scores = (
        mu1
        - mu0
        + a * (y - mu1) / propensity
        - (1.0 - a) * (y - mu0) / (1.0 - propensity)
    )
    return float(scores.mean())


def classic_regime(ep: Episode) -> tuple[int, float]:
    """BIC-penalized one-split linear model, a compact MOB-like specialist."""
    values = _complete(ep.values)
    cov = _covariates(ep)
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    x, y = values[:, cov], values[:, outcome]
    base = LinearRegression().fit(x, y)
    base_sse = float(np.sum((y - base.predict(x)) ** 2))
    best_bic = len(y) * np.log(max(base_sse / len(y), 1e-8)) + (x.shape[1] + 1) * np.log(len(y))
    # The executable model reserves index 12 as its task-level "no split" token.
    best_feature = 12
    for local_j, feature in enumerate(cov):
        for threshold in np.quantile(x[:, local_j], [0.25, 0.4, 0.5, 0.6, 0.75]):
            side = x[:, local_j] > threshold
            if side.sum() < 18 or (~side).sum() < 18:
                continue
            design = np.column_stack([x, side[:, None] * x])
            model = LinearRegression().fit(design, y)
            sse = float(np.sum((y - model.predict(design)) ** 2))
            bic = len(y) * np.log(max(sse / len(y), 1e-8)) + (design.shape[1] + 1) * np.log(len(y))
            if bic < best_bic:
                best_bic = bic
                best_feature = int(feature)
    return best_feature, float(base_sse)


def all_baselines(ep: Episode) -> dict[str, float | int]:
    if ep.task == TASK_REGIME:
        feature, _ = classic_regime(ep)
        return {"MOB-like": feature}
    estimates = {
        "Linear g-computation": linear_effect(ep),
        "Random forest g-computation": forest_effect(ep),
    }
    experts = estimator_experts(ep)
    estimates["Spline g-computation"] = float(experts.estimates[2])
    estimates["Interaction g-computation"] = float(experts.estimates[3])
    if ep.task == TASK_ATE:
        estimates["AIPW"] = aipw_effect(ep)
        estimates["DR learner (spline nuisances)"] = spline_dr_effect(ep)
        if experts.mask[5]:
            estimates["Randomized difference-in-means"] = float(experts.estimates[5])
    else:
        estimates["Piecewise exposure-response"] = float(experts.estimates[4])
        estimates["Null shrinkage"] = 0.0
    return estimates
