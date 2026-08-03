"""Deterministic pre-inference validation and empirical support checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..data import ROLE_EXPOSURE, ROLE_OUTCOME, ROLE_TREATMENT
from .contracts import ObservedCase, Task


class IssueLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    level: IssueLevel
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    support_adequate: bool = True

    @property
    def valid(self) -> bool:
        return not any(issue.level == IssueLevel.ERROR for issue in self.issues)

    @property
    def errors(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.level == IssueLevel.ERROR]

    @property
    def warnings(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.level == IssueLevel.WARNING]


def validate_case(case: ObservedCase, max_variables: int = 12) -> ValidationReport:
    report = ValidationReport()
    values = case.values
    n, p = values.shape
    report.diagnostics.update(
        {
            "sample_size": n,
            "variable_count": p,
            "maximum_variables": max_variables,
            "missing_count": int(np.isnan(values).sum()),
            "missing_fraction": float(np.isnan(values).mean()),
        }
    )
    if p > max_variables:
        report.issues.append(
            ValidationIssue(
                IssueLevel.ERROR,
                "variable_limit",
                f"Selected input has {p} variables; CWFM accepts at most {max_variables}. Select fewer variables.",
            )
        )
    if n < 30:
        report.issues.append(
            ValidationIssue(
                IssueLevel.ERROR,
                "sample_size",
                "At least 30 observations are required for the current expert estimators.",
            )
        )
    if not np.issubdtype(values.dtype, np.number):
        report.issues.append(
            ValidationIssue(IssueLevel.ERROR, "numeric", "All selected columns must be numeric.")
        )
    all_missing = np.isnan(values).all(axis=0)
    if all_missing.any():
        names = [case.variable_names[index] for index in np.flatnonzero(all_missing)]
        report.issues.append(
            ValidationIssue(
                IssueLevel.ERROR,
                "all_missing",
                f"Columns contain no observed values: {', '.join(names)}",
            )
        )
    constant = np.nanstd(values, axis=0) < 1e-10
    if constant.any():
        names = [case.variable_names[index] for index in np.flatnonzero(constant)]
        report.issues.append(
            ValidationIssue(
                IssueLevel.WARNING,
                "constant_columns",
                f"Constant columns provide no information: {', '.join(names)}",
            )
        )

    outcomes = np.flatnonzero(case.roles == ROLE_OUTCOME)
    treatments = np.flatnonzero(case.roles == ROLE_TREATMENT)
    exposures = np.flatnonzero(case.roles == ROLE_EXPOSURE)
    if len(outcomes) != 1:
        report.issues.append(
            ValidationIssue(IssueLevel.ERROR, "outcome_role", "Exactly one outcome must be selected.")
        )
    if case.task in {Task.STATIC_ATE, Task.NETWORK_INTERFERENCE} and len(treatments) != 1:
        report.issues.append(
            ValidationIssue(IssueLevel.ERROR, "treatment_role", "Exactly one treatment must be selected.")
        )
    if case.task == Task.NETWORK_INTERFERENCE and len(exposures) != 1:
        report.issues.append(
            ValidationIssue(IssueLevel.ERROR, "exposure_role", "Exactly one exposure must be selected.")
        )

    if len(treatments) == 1:
        treatment = values[:, treatments[0]]
        observed = treatment[np.isfinite(treatment)]
        levels, counts = np.unique(observed, return_counts=True)
        report.diagnostics["treatment_levels"] = levels.tolist()
        report.diagnostics["treatment_counts"] = {
            str(float(level)): int(count) for level, count in zip(levels, counts)
        }
        if len(levels) != 2 or not np.allclose(levels, [0.0, 1.0]):
            report.issues.append(
                ValidationIssue(
                    IssueLevel.ERROR,
                    "binary_treatment",
                    "The current effect queries require a binary treatment encoded as 0 and 1.",
                )
            )
        elif counts.min() < 10:
            report.support_adequate = False
            report.issues.append(
                ValidationIssue(
                    IssueLevel.WARNING,
                    "treatment_support",
                    "One treatment group has fewer than 10 observations.",
                )
            )

    if case.task == Task.NETWORK_INTERFERENCE:
        if case.adjacency is None or case.adjacency.shape != (n, n):
            report.issues.append(
                ValidationIssue(
                    IssueLevel.ERROR,
                    "adjacency_shape",
                    f"Network adjacency must have shape ({n}, {n}).",
                )
            )
        elif not np.isfinite(case.adjacency).all():
            report.issues.append(
                ValidationIssue(IssueLevel.ERROR, "adjacency_finite", "Network adjacency contains non-finite values.")
            )
        elif (case.adjacency < 0).any():
            report.issues.append(
                ValidationIssue(
                    IssueLevel.ERROR,
                    "adjacency_nonnegative",
                    "Mapped exposure currently requires non-negative network weights.",
                )
            )
        else:
            symmetric = bool(np.allclose(case.adjacency, case.adjacency.T, atol=1e-6))
            diagonal_zero = bool(np.allclose(np.diag(case.adjacency), 0.0, atol=1e-6))
            degrees = (case.adjacency != 0).sum(1)
            report.diagnostics.update(
                {
                    "graph_symmetric": symmetric,
                    "graph_diagonal_zero": diagonal_zero,
                    "isolated_units": int((degrees == 0).sum()),
                    "graph_density": float(np.count_nonzero(case.adjacency) / max(n * (n - 1), 1)),
                    "degree_min": int(degrees.min()),
                    "degree_median": float(np.median(degrees)),
                    "degree_max": int(degrees.max()),
                }
            )
            if not symmetric:
                report.issues.append(
                    ValidationIssue(IssueLevel.WARNING, "graph_symmetry", "The observed network is directed or asymmetric.")
                )
            if not diagonal_zero:
                report.issues.append(
                    ValidationIssue(IssueLevel.ERROR, "graph_diagonal", "Network self-edges are not supported.")
                )
        if case.cluster_ids is None or len(case.cluster_ids) != n:
            report.issues.append(
                ValidationIssue(
                    IssueLevel.ERROR,
                    "cluster_length",
                    "cluster_ids must have one value per observation.",
                )
            )
        elif len(np.unique(case.cluster_ids)) < 2:
            report.issues.append(
                ValidationIssue(IssueLevel.WARNING, "cluster_count", "Only one cluster is present.")
            )
        if len(exposures) == 1:
            exposure = values[:, exposures[0]]
            finite = exposure[np.isfinite(exposure)]
            report.diagnostics.update(
                {
                    "exposure_min": float(np.min(finite)),
                    "exposure_p05": float(np.quantile(finite, 0.05)),
                    "exposure_median": float(np.median(finite)),
                    "exposure_p95": float(np.quantile(finite, 0.95)),
                    "exposure_max": float(np.max(finite)),
                }
            )
    return report


def evaluate_exposure_support(
    case: ObservedCase,
    low: float,
    high: float,
    minimum_nearby: int = 10,
) -> tuple[bool, dict[str, Any], list[str]]:
    index = int(np.flatnonzero(case.roles == ROLE_EXPOSURE)[0])
    treatment_index = int(np.flatnonzero(case.roles == ROLE_TREATMENT)[0])
    exposure = case.values[:, index]
    treatment = case.values[:, treatment_index]
    finite = exposure[np.isfinite(exposure)]
    tolerance = max(0.05, 0.10 * max(float(np.ptp(finite)), 1e-6))
    near_low = int((np.abs(exposure - low) <= tolerance).sum())
    near_high = int((np.abs(exposure - high) <= tolerance).sum())
    p05, p95 = np.quantile(finite, [0.05, 0.95])
    by_treatment = {}
    adequate_by_treatment = True
    for level in (0.0, 1.0):
        group = exposure[np.isclose(treatment, level) & np.isfinite(exposure)]
        if len(group):
            q05, q95 = np.quantile(group, [0.05, 0.95])
            group_ok = bool(q05 <= low <= q95 and q05 <= high <= q95)
            adequate_by_treatment &= group_ok
            by_treatment[str(int(level))] = {
                "count": int(len(group)),
                "p05": float(q05),
                "p95": float(q95),
                "targets_inside": group_ok,
            }
    adequate = bool(
        p05 <= low < high <= p95
        and near_low >= minimum_nearby
        and near_high >= minimum_nearby
        and adequate_by_treatment
    )
    reasons = []
    if not adequate:
        reasons.append(
            "Requested exposure targets do not have adequate observed support in both own-treatment groups."
        )
    diagnostics = {
        "g0": low,
        "g1": high,
        "tolerance": tolerance,
        "near_g0": near_low,
        "near_g1": near_high,
        "p05": float(p05),
        "p95": float(p95),
        "by_treatment": by_treatment,
        "adequate": adequate,
    }
    return adequate, diagnostics, reasons


def evaluate_ate_support(
    case: ObservedCase,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Approximate overlap check using a deterministic ridge propensity fit."""

    treatment_index = int(np.flatnonzero(case.roles == ROLE_TREATMENT)[0])
    excluded = (case.roles == ROLE_TREATMENT) | (case.roles == ROLE_OUTCOME)
    x = case.values[:, ~excluded].astype(float, copy=True)
    a = case.values[:, treatment_index].astype(float)
    means = np.nanmean(x, axis=0)
    rows, columns = np.nonzero(np.isnan(x))
    x[rows, columns] = np.nan_to_num(means)[columns]
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - np.mean(x, axis=0)) / scale
    design = np.column_stack([np.ones(len(x)), x])
    ridge = np.eye(design.shape[1]) * 0.05
    ridge[0, 0] = 0.0
    coefficient = np.linalg.lstsq(design.T @ design + ridge, design.T @ a, rcond=None)[0]
    propensity = np.clip(design @ coefficient, 0.0, 1.0)
    outside = (propensity < 0.05) | (propensity > 0.95)
    p01, p05, p50, p95, p99 = np.quantile(propensity, [0.01, 0.05, 0.5, 0.95, 0.99])
    adequate = bool(outside.mean() <= 0.10 and p05 > 0.01 and p95 < 0.99)
    diagnostics = {
        "propensity_p01": float(p01),
        "propensity_p05": float(p05),
        "propensity_median": float(p50),
        "propensity_p95": float(p95),
        "propensity_p99": float(p99),
        "outside_0.05_0.95_fraction": float(outside.mean()),
        "adequate": adequate,
    }
    reasons = [] if adequate else [
        "The discovery-only propensity diagnostic indicates inadequate treatment overlap."
    ]
    return adequate, diagnostics, reasons
