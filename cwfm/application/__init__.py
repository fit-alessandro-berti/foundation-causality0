"""Stable, truth-safe application layer for CWFM inference."""

from .catalog import RepositoryCase, RepositoryCatalog
from .contracts import (
    ATEQuery,
    AnalysisQuery,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    CaseSource,
    InterferenceQuery,
    ObservedCase,
    RegimeQuery,
    SupportState,
    Task,
)
from .model_bundle import ModelBundle, ModelBundleStatus
from .results import AnalysisResult, AnalysisStatus
from .runner import CWFMRunner
from .exporting import analysis_zip, export_analysis
from .adapters.reference_benchmarks import (
    ReferenceBenchmarkResult,
    load_reference_result,
)
from .presets import TASK_METHOD, build_repository_request

__all__ = [
    "ATEQuery",
    "AnalysisQuery",
    "AnalysisResult",
    "AnalysisStatus",
    "AssertionState",
    "AssignmentDesign",
    "AssumptionLedger",
    "CWFMRunner",
    "CaseSource",
    "InterferenceQuery",
    "ModelBundle",
    "ModelBundleStatus",
    "ObservedCase",
    "RegimeQuery",
    "ReferenceBenchmarkResult",
    "RepositoryCase",
    "RepositoryCatalog",
    "SupportState",
    "Task",
    "TASK_METHOD",
    "analysis_zip",
    "export_analysis",
    "load_reference_result",
    "build_repository_request",
]
