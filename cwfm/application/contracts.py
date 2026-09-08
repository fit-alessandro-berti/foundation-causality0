"""Inference-only contracts for the public CWFM application API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np


class Task(str, Enum):
    STATIC_ATE = "static_ate"
    OBSERVED_REGIME = "observed_regime"
    NETWORK_INTERFERENCE = "network_interference"


class AssignmentDesign(str, Enum):
    RANDOMIZED = "randomized"
    OBSERVATIONAL = "observational"
    UNKNOWN = "unknown"


class AssertionState(str, Enum):
    ASSERTED = "asserted"
    NOT_ASSERTED = "not_asserted"
    UNKNOWN = "unknown"


class SupportState(str, Enum):
    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CaseSource:
    """Truth-free description of where an observed case came from."""

    method: str
    scenario: str
    seed: int
    discovery_path: Path | None = None
    metadata_path: Path | None = None
    backend: str = "CWFM checkpoint"


@dataclass(frozen=True)
class ObservedCase:
    """Observed arrays and roles; deliberately contains no training target."""

    values: np.ndarray
    variable_names: tuple[str, ...]
    roles: np.ndarray
    task: Task
    adjacency: np.ndarray | None = None
    cluster_ids: np.ndarray | None = None
    source: CaseSource | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        roles = np.asarray(self.roles)
        if values.ndim != 2:
            raise ValueError("ObservedCase.values must be a two-dimensional array")
        if values.shape[1] != len(self.variable_names):
            raise ValueError("variable_names must have one entry per value column")
        if roles.shape != (values.shape[1],):
            raise ValueError("roles must have one entry per value column")
        if len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError("variable names must be unique")
        object.__setattr__(self, "values", values.astype(np.float32, copy=False))
        object.__setattr__(self, "roles", roles.astype(np.int64, copy=False))
        if self.adjacency is not None:
            object.__setattr__(
                self, "adjacency", np.asarray(self.adjacency, dtype=np.float32)
            )
        if self.cluster_ids is not None:
            object.__setattr__(
                self, "cluster_ids", np.asarray(self.cluster_ids, dtype=np.int64)
            )


@dataclass(frozen=True)
class ATEQuery:
    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    treatment_low: float = 0.0
    treatment_high: float = 1.0

    @property
    def task(self) -> Task:
        return Task.STATIC_ATE


@dataclass(frozen=True)
class RegimeQuery:
    outcome: str
    ordinary_predictors: tuple[str, ...]
    regime_candidates: tuple[str, ...]
    threshold_quantiles: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80)
    candidate_policy: Literal["explicit", "screen-then-model", "chunked"] = "explicit"

    def __post_init__(self) -> None:
        if len(self.threshold_quantiles) != 5:
            raise ValueError("The current regime decoder requires exactly five threshold quantiles.")
        if any(not 0.0 < value < 1.0 for value in self.threshold_quantiles):
            raise ValueError("Threshold quantiles must lie strictly between zero and one.")

    @property
    def task(self) -> Task:
        return Task.OBSERVED_REGIME


@dataclass(frozen=True)
class InterferenceQuery:
    treatment: str = "A"
    exposure: str = "G"
    outcome: str = "Y"
    covariates: tuple[str, ...] = ()
    exposure_low: float = 0.25
    exposure_high: float = 0.75
    exposure_mapping: Literal["weighted", "unweighted", "distance_two"] = "weighted"
    include_degree: bool = True

    @property
    def task(self) -> Task:
        return Task.NETWORK_INTERFERENCE


AnalysisQuery = ATEQuery | RegimeQuery | InterferenceQuery


@dataclass(frozen=True)
class AssumptionLedger:
    """User declarations used by the formal answer/abstention gate.

    These fields are assumptions, not facts established from the observed data.
    An answered status certifies compliance with the implemented rules only;
    incomplete or incorrect declarations can still produce invalid estimates.
    """

    assignment_design: AssignmentDesign = AssignmentDesign.UNKNOWN
    no_unmeasured_confounding: AssertionState = AssertionState.UNKNOWN
    consistency: AssertionState = AssertionState.UNKNOWN
    treatment_support: SupportState = SupportState.UNKNOWN
    exposure_mapping_predeclared: bool | None = None
    interference_restricted_to_observed_graph: bool | None = None
