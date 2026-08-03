"""Adapter for mapped network-exposure spillover contrasts."""

from __future__ import annotations

import numpy as np

from ...data import ROLE_COVARIATE, ROLE_EXPOSURE, ROLE_OUTCOME, ROLE_TREATMENT
from ..catalog import RepositoryCase
from ..contracts import InterferenceQuery, ObservedCase, Task
from .base import column_indices


def compute_exposure(adjacency: np.ndarray, treatment: np.ndarray, mapping: str) -> np.ndarray:
    weights = np.asarray(adjacency, dtype=float).copy()
    np.fill_diagonal(weights, 0.0)
    if mapping == "unweighted":
        weights = (weights != 0).astype(float)
    elif mapping == "distance_two":
        direct = (weights != 0).astype(float)
        distance_two = (direct @ direct > 0).astype(float)
        distance_two[direct > 0] = 0.0
        np.fill_diagonal(distance_two, 0.0)
        weights = distance_two
    elif mapping != "weighted":
        raise ValueError(f"Unsupported exposure mapping: {mapping}")
    denominator = weights.sum(1)
    return (weights @ treatment) / np.maximum(denominator, 1e-12)


def _repository_case(case: RepositoryCase, query: InterferenceQuery) -> ObservedCase:
    arrays = case.load_observed()
    metadata = case.safe_metadata()
    x = np.asarray(arrays["X"])
    a = np.asarray(arrays["A"]).reshape(-1)
    y = np.asarray(arrays["Y"])
    adjacency = np.asarray(arrays["W_observed"])
    clusters = np.asarray(arrays["cluster_id"]).reshape(-1)
    features = tuple(metadata.get("feature_names", [f"X{i + 1}" for i in range(x.shape[1])]))
    outcomes = tuple(metadata.get("outcome_names", [f"Y{i + 1}" for i in range(y.shape[1])]))
    if query.outcome not in outcomes:
        raise ValueError(f"Unknown outcome {query.outcome}")
    exposure = compute_exposure(adjacency, a, query.exposure_mapping)
    columns = [x]
    names: tuple[str, ...] = features
    roles = [ROLE_COVARIATE] * len(features)
    if query.include_degree:
        degree = (adjacency != 0).sum(1).astype(float)
        degree = (degree - degree.mean()) / max(float(degree.std()), 1.0)
        columns.append(degree[:, None])
        names += ("degree",)
        roles.append(ROLE_COVARIATE)
    columns.extend([a[:, None], exposure[:, None], y[:, outcomes.index(query.outcome), None]])
    names += (query.treatment, query.exposure, query.outcome)
    roles.extend([ROLE_TREATMENT, ROLE_EXPOSURE, ROLE_OUTCOME])
    return ObservedCase(
        np.column_stack(columns),
        names,
        np.asarray(roles),
        Task.NETWORK_INTERFERENCE,
        adjacency=adjacency,
        cluster_ids=clusters,
        source=case.source,
        metadata={**metadata, "exposure_mapping": query.exposure_mapping},
    )


def adapt_interference(
    case: RepositoryCase | ObservedCase,
    query: InterferenceQuery,
) -> ObservedCase:
    if not (
        np.isfinite(query.exposure_low)
        and np.isfinite(query.exposure_high)
        and query.exposure_low < query.exposure_high
    ):
        raise ValueError("Exposure targets must be finite and satisfy g0 < g1.")
    observed = _repository_case(case, query) if isinstance(case, RepositoryCase) else case
    base_covariates = query.covariates or tuple(
        name
        for name, role in zip(observed.variable_names, observed.roles)
        if role == ROLE_COVARIATE and name != "degree"
    )
    extra_covariates = ("degree",) if query.include_degree and "degree" in observed.variable_names else ()
    selected_covariates = tuple(dict.fromkeys(base_covariates + extra_covariates))
    selected = selected_covariates + (query.treatment, query.exposure, query.outcome)
    indices = column_indices(observed.variable_names, selected)
    roles = np.asarray(
        [ROLE_COVARIATE] * len(selected_covariates)
        + [ROLE_TREATMENT, ROLE_EXPOSURE, ROLE_OUTCOME]
    )
    return ObservedCase(
        observed.values[:, indices],
        selected,
        roles,
        Task.NETWORK_INTERFERENCE,
        adjacency=observed.adjacency,
        cluster_ids=observed.cluster_ids,
        source=observed.source,
        metadata=observed.metadata,
    )
