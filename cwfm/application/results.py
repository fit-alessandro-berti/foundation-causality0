"""Serializable result types shared by the CLI examples and Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    SupportState,
    Task,
)


class AnalysisStatus(str, Enum):
    ANSWERED = "ANSWERED"
    ANSWERED_CONDITIONAL_ON_ASSUMPTIONS = "ANSWERED_CONDITIONAL_ON_ASSUMPTIONS"
    ABSTAINED_IDENTIFICATION_NOT_ESTABLISHED = (
        "ABSTAINED_IDENTIFICATION_NOT_ESTABLISHED"
    )
    ABSTAINED_INADEQUATE_SUPPORT = "ABSTAINED_INADEQUATE_SUPPORT"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"

    @property
    def answered(self) -> bool:
        return self in {
            AnalysisStatus.ANSWERED,
            AnalysisStatus.ANSWERED_CONDITIONAL_ON_ASSUMPTIONS,
        }


@dataclass
class ExpertResult:
    name: str
    estimate: float | None
    standard_error: float | None
    weight: float
    available: bool = True


@dataclass
class RegimeResult:
    split_detected: bool
    evidence_score: float
    calibrated_threshold: float
    selected_variable: str | None = None
    selected_threshold: float | None = None
    change_type: str | None = None
    left_count: int | None = None
    right_count: int | None = None
    evidence_table: list[dict[str, Any]] = field(default_factory=list)
    candidate_screening: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InterferenceResult:
    exposure_mapping: str
    exposure_low: float
    exposure_high: float
    estimand_description: str
    support: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDiagnostics:
    expert_estimates: list[float | None] = field(default_factory=list)
    expert_standard_errors: list[float | None] = field(default_factory=list)
    expert_weights: list[float] = field(default_factory=list)
    compiled_estimate: float | None = None
    final_estimate: float | None = None
    residual_correction: float | None = None
    residual_gate: float | None = None
    identification_gate: str | None = None
    support_probability: float | None = None
    prior_mismatch_probability: float | None = None
    flexible_route_probability: float | None = None
    mechanism_routing: list[float] = field(default_factory=list)
    mechanism_expert_routing: list[list[float]] = field(default_factory=list)
    world_weights: list[float] = field(default_factory=list)
    world_particle_estimates: list[float] = field(default_factory=list)
    world_entropy: float | None = None
    world_effective_count: float | None = None
    graph_logits: list[list[float]] = field(default_factory=list)
    nuisance: dict[str, float | None] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    checkpoint_hash: str | None = None
    calibration_hash: str | None = None
    model_configuration: dict[str, Any] = field(default_factory=dict)
    repository_inputs: list[str] = field(default_factory=list)
    selected_variables: list[str] = field(default_factory=list)
    query: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    software_versions: dict[str, str] = field(default_factory=dict)
    random_seed: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class AnalysisResult:
    analysis_id: str
    task: Task
    status: AnalysisStatus
    question: str
    estimand: str
    estimate: float | None
    interval: tuple[float, float] | None
    interval_level: float | None
    assumptions: AssumptionLedger
    decision_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expert_results: list[ExpertResult] = field(default_factory=list)
    selected_expert: str | None = None
    compiled_estimate: float | None = None
    final_estimate: float | None = None
    regime_result: RegimeResult | None = None
    interference_result: InterferenceResult | None = None
    diagnostics: ModelDiagnostics = field(default_factory=ModelDiagnostics)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AnalysisResult":
        assumption_values = raw["assumptions"]
        assumptions = AssumptionLedger(
            assignment_design=AssignmentDesign(assumption_values["assignment_design"]),
            no_unmeasured_confounding=AssertionState(
                assumption_values["no_unmeasured_confounding"]
            ),
            consistency=AssertionState(assumption_values["consistency"]),
            treatment_support=SupportState(assumption_values["treatment_support"]),
            exposure_mapping_predeclared=assumption_values.get(
                "exposure_mapping_predeclared"
            ),
            interference_restricted_to_observed_graph=assumption_values.get(
                "interference_restricted_to_observed_graph"
            ),
        )
        return cls(
            analysis_id=str(raw["analysis_id"]),
            task=Task(raw["task"]),
            status=AnalysisStatus(raw["status"]),
            question=str(raw["question"]),
            estimand=str(raw["estimand"]),
            estimate=raw.get("estimate"),
            interval=tuple(raw["interval"]) if raw.get("interval") is not None else None,
            interval_level=raw.get("interval_level"),
            assumptions=assumptions,
            decision_reasons=list(raw.get("decision_reasons", [])),
            warnings=list(raw.get("warnings", [])),
            expert_results=[ExpertResult(**item) for item in raw.get("expert_results", [])],
            selected_expert=raw.get("selected_expert"),
            compiled_estimate=raw.get("compiled_estimate"),
            final_estimate=raw.get("final_estimate"),
            regime_result=(
                RegimeResult(**raw["regime_result"])
                if raw.get("regime_result") is not None
                else None
            ),
            interference_result=(
                InterferenceResult(**raw["interference_result"])
                if raw.get("interference_result") is not None
                else None
            ),
            diagnostics=ModelDiagnostics(**raw.get("diagnostics", {})),
            provenance=Provenance(**raw.get("provenance", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> "AnalysisResult":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> "AnalysisResult":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output
