"""Adapter and discovery-only candidate screening for observed regimes."""

from __future__ import annotations

import numpy as np

from ...data import ROLE_COVARIATE, ROLE_OUTCOME, ROLE_PREDICTOR
from ..catalog import RepositoryCase
from ..contracts import ObservedCase, RegimeQuery, Task
from .base import column_indices, impute_for_screening


def _repository_case(case: RepositoryCase, outcome: str) -> ObservedCase:
    arrays = case.load_observed()
    metadata = case.safe_metadata()
    x = np.asarray(arrays["X"])
    y = np.asarray(arrays["Y"])
    features = tuple(metadata.get("feature_names", [f"X{i + 1}" for i in range(x.shape[1])]))
    outcomes = tuple(metadata.get("outcome_names", [f"Y{i + 1}" for i in range(y.shape[1])]))
    if outcome not in outcomes:
        raise ValueError(f"Unknown outcome {outcome}")
    outcome_values = y[:, outcomes.index(outcome)]
    names = features + (outcome,)
    roles = np.asarray([ROLE_COVARIATE] * len(features) + [ROLE_OUTCOME])
    return ObservedCase(
        np.column_stack([x, outcome_values]),
        names,
        roles,
        Task.OBSERVED_REGIME,
        source=case.source,
        metadata=metadata,
    )


def screen_candidates(
    observed: ObservedCase,
    query: RegimeQuery,
) -> list[dict[str, float | str | int]]:
    """Rank candidates by the best discovery-only split BIC gain."""

    ordinary_indices = column_indices(observed.variable_names, query.ordinary_predictors)
    outcome_index = column_indices(observed.variable_names, [query.outcome])[0]
    values = impute_for_screening(observed.values)
    y = values[:, outcome_index]
    base = np.column_stack([np.ones(len(y)), values[:, ordinary_indices]])
    base_coef = np.linalg.lstsq(base, y, rcond=None)[0]
    base_sse = max(float(np.square(y - base @ base_coef).sum()), 1e-10)
    records = []
    for candidate in query.regime_candidates:
        index = column_indices(observed.variable_names, [candidate])[0]
        best_gain = -np.inf
        best_quantile = query.threshold_quantiles[0]
        best_threshold = float(np.quantile(values[:, index], best_quantile))
        for quantile in query.threshold_quantiles:
            threshold = float(np.quantile(values[:, index], quantile))
            side = values[:, index] > threshold
            if min(int(side.sum()), int((~side).sum())) < 10:
                continue
            design = np.column_stack([base, side, side[:, None] * base[:, 1:]])
            coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
            sse = max(float(np.square(y - design @ coefficient).sum()), 1e-10)
            gain = (
                len(y) * np.log(base_sse / len(y))
                + base.shape[1] * np.log(len(y))
                - len(y) * np.log(sse / len(y))
                - design.shape[1] * np.log(len(y))
            ) / len(y)
            if gain > best_gain:
                best_gain, best_quantile, best_threshold = gain, quantile, threshold
        records.append(
            {
                "variable": candidate,
                "score": float(max(best_gain, 0.0)),
                "best_quantile": float(best_quantile),
                "best_threshold": best_threshold,
            }
        )
    ranked = sorted(records, key=lambda row: (-float(row["score"]), str(row["variable"])))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def adapt_regime(
    case: RepositoryCase | ObservedCase,
    query: RegimeQuery,
    max_variables: int = 12,
) -> tuple[ObservedCase, list[dict[str, object]]]:
    observed = _repository_case(case, query.outcome) if isinstance(case, RepositoryCase) else case
    budget = max_variables - len(query.ordinary_predictors) - 1
    candidates = query.regime_candidates
    screening: list[dict[str, object]] = []
    if query.candidate_policy == "screen-then-model":
        screening = list(screen_candidates(observed, query))
        candidates = tuple(str(row["variable"]) for row in screening[:budget])
    elif query.candidate_policy == "chunked":
        raise ValueError(
            "Chunked candidate aggregation is a research option and is not enabled until cross-chunk calibration is validated."
        )
    if len(candidates) > budget:
        raise ValueError(
            f"The selection uses {len(query.ordinary_predictors) + len(candidates) + 1} variables; select at most {budget} regime candidates."
        )
    selected = query.ordinary_predictors + candidates + (query.outcome,)
    indices = column_indices(observed.variable_names, selected)
    roles = np.asarray(
        [ROLE_PREDICTOR] * len(query.ordinary_predictors)
        + [ROLE_COVARIATE] * len(candidates)
        + [ROLE_OUTCOME]
    )
    adapted = ObservedCase(
        observed.values[:, indices],
        selected,
        roles,
        Task.OBSERVED_REGIME,
        source=observed.source,
        metadata={**observed.metadata, "candidate_policy": query.candidate_policy},
    )
    return adapted, screening

