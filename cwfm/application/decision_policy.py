"""Formal, assumption-driven answer and abstention policy."""

from __future__ import annotations

from .contracts import (
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    SupportState,
    Task,
)
from .results import AnalysisStatus


def decide(
    task: Task,
    assumptions: AssumptionLedger,
    empirical_support_adequate: bool,
) -> tuple[AnalysisStatus, list[str]]:
    if task == Task.OBSERVED_REGIME:
        return AnalysisStatus.ANSWERED, [
            "Regime determination is a calibrated structural decision, not a causal identification claim."
        ]

    declared_support = assumptions.treatment_support
    if declared_support == SupportState.INADEQUATE or not empirical_support_adequate:
        return AnalysisStatus.ABSTAINED_INADEQUATE_SUPPORT, [
            "The requested contrast does not meet the declared or empirical support rule."
        ]

    consistency = assumptions.consistency == AssertionState.ASSERTED
    assignment_identifies = assumptions.assignment_design == AssignmentDesign.RANDOMIZED
    observational_identifies = (
        assumptions.assignment_design == AssignmentDesign.OBSERVATIONAL
        and assumptions.no_unmeasured_confounding == AssertionState.ASSERTED
    )
    identified = consistency and (assignment_identifies or observational_identifies)
    if task == Task.NETWORK_INTERFERENCE:
        identified = bool(
            identified
            and assumptions.exposure_mapping_predeclared is True
            and assumptions.interference_restricted_to_observed_graph is True
        )
    if not identified:
        return AnalysisStatus.ABSTAINED_IDENTIFICATION_NOT_ESTABLISHED, [
            "Identification is not established by the declarations in the assumption ledger."
        ]

    if task == Task.STATIC_ATE and assignment_identifies:
        return AnalysisStatus.ANSWERED, [
            "Identified by the declared randomized assignment and consistency assumption."
        ]
    if task == Task.NETWORK_INTERFERENCE:
        return AnalysisStatus.ANSWERED_CONDITIONAL_ON_ASSUMPTIONS, [
            "Identified conditional on the assignment, consistency, predeclared exposure mapping, and observed-network interference restriction."
        ]
    return AnalysisStatus.ANSWERED_CONDITIONAL_ON_ASSUMPTIONS, [
        "Identified conditional on the declared no-unmeasured-confounding and consistency assumptions."
    ]

