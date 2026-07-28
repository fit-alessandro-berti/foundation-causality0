from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

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


def _columns(ep: Episode, role: int) -> np.ndarray:
    return np.flatnonzero(ep.roles == role)


def _covariates(ep: Episode) -> np.ndarray:
    return np.flatnonzero(
        (ep.roles == ROLE_COVARIATE) | (ep.roles == ROLE_PREDICTOR)
    )


def linear_effect(ep: Episode) -> float:
    values = ep.values
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
    values = ep.values
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
    values = ep.values
    cov = _covariates(ep)
    treatment = int(_columns(ep, ROLE_TREATMENT)[0])
    outcome = int(_columns(ep, ROLE_OUTCOME)[0])
    x, a, y = values[:, cov], values[:, treatment], values[:, outcome]
    propensity = LogisticRegression(C=1.0, max_iter=500).fit(x, a).predict_proba(x)[:, 1]
    propensity = np.clip(propensity, 0.05, 0.95)
    features = np.column_stack([x, a])
    model = RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=7, max_features=0.8, n_jobs=1, random_state=ep.seed
    ).fit(features, y)
    low, high = features.copy(), features.copy()
    low[:, -1], high[:, -1] = 0.0, 1.0
    mu0, mu1 = model.predict(low), model.predict(high)
    score = mu1 - mu0 + a * (y - mu1) / propensity - (1 - a) * (y - mu0) / (1 - propensity)
    return float(np.mean(score))


def classic_regime(ep: Episode) -> tuple[int, float]:
    """BIC-penalized one-split linear model, a compact MOB-like specialist."""
    values = ep.values
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
    if ep.task == TASK_ATE:
        estimates["AIPW"] = aipw_effect(ep)
    return estimates
