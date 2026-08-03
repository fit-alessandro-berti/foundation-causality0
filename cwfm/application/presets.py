"""Repository-aware default queries used by batch and diagnostic clients."""

from __future__ import annotations

from .catalog import RepositoryCase
from .contracts import (
    ATEQuery,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    InterferenceQuery,
    RegimeQuery,
)


TASK_METHOD = {
    "static_ate": "static_ate",
    "observed_regime": "causal_model_determination",
    "network_interference": "causal_interference_detection",
}


def build_repository_request(
    case: RepositoryCase,
    task: str,
    outcome: str,
):
    metadata = case.safe_metadata()
    features = tuple(metadata.get("feature_names", []))
    if task == "static_ate":
        return (
            ATEQuery("A", "Y", features),
            AssumptionLedger(
                assignment_design=(
                    AssignmentDesign.RANDOMIZED
                    if metadata.get("assignment_design") == "randomized"
                    else AssignmentDesign.OBSERVATIONAL
                ),
                no_unmeasured_confounding=AssertionState.ASSERTED,
                consistency=AssertionState.ASSERTED,
            ),
        )
    if task == "observed_regime":
        ordinary = tuple(
            features[index]
            for index in metadata.get("ordinary_predictors", [0, 1, 2, 3])
        )
        candidates = tuple(name for name in features if name not in ordinary)
        return (
            RegimeQuery(
                outcome,
                ordinary,
                candidates,
                candidate_policy="screen-then-model",
            ),
            AssumptionLedger(),
        )
    if task == "network_interference":
        return (
            InterferenceQuery("A", "G", outcome, features),
            AssumptionLedger(
                assignment_design=AssignmentDesign.RANDOMIZED,
                no_unmeasured_confounding=AssertionState.ASSERTED,
                consistency=AssertionState.ASSERTED,
                exposure_mapping_predeclared=True,
                interference_restricted_to_observed_graph=True,
            ),
        )
    raise ValueError(f"Unsupported task: {task}")
