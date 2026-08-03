"""Adapter for binary static average-treatment-effect queries."""

from __future__ import annotations

import numpy as np

from ...data import ROLE_COVARIATE, ROLE_OUTCOME, ROLE_TREATMENT
from ..catalog import RepositoryCase
from ..contracts import ATEQuery, ObservedCase, Task
from .base import column_indices


def _repository_case(case: RepositoryCase) -> ObservedCase:
    arrays = case.load_observed()
    metadata = case.safe_metadata()
    x = np.asarray(arrays["X"])
    a = np.asarray(arrays["A"]).reshape(-1)
    y = np.asarray(arrays["Y"])
    if y.ndim == 2:
        y = y[:, 0]
    features = tuple(metadata.get("feature_names", [f"X{i + 1}" for i in range(x.shape[1])]))
    names = features + ("A", "Y")
    roles = np.asarray([ROLE_COVARIATE] * x.shape[1] + [ROLE_TREATMENT, ROLE_OUTCOME])
    return ObservedCase(
        np.column_stack([x, a, y]),
        names,
        roles,
        Task.STATIC_ATE,
        source=case.source,
        metadata=metadata,
    )


def adapt_ate(case: RepositoryCase | ObservedCase, query: ATEQuery) -> ObservedCase:
    if not np.isclose(query.treatment_low, 0.0) or not np.isclose(
        query.treatment_high, 1.0
    ):
        raise ValueError(
            "The current static ATE adapter supports only the binary contrast 0 → 1."
        )
    observed = _repository_case(case) if isinstance(case, RepositoryCase) else case
    selected = query.covariates + (query.treatment, query.outcome)
    indices = column_indices(observed.variable_names, selected)
    roles = np.asarray(
        [ROLE_COVARIATE] * len(query.covariates) + [ROLE_TREATMENT, ROLE_OUTCOME]
    )
    return ObservedCase(
        observed.values[:, indices],
        selected,
        roles,
        Task.STATIC_ATE,
        source=observed.source,
        metadata=observed.metadata,
    )
