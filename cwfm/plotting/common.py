"""Small validation helpers for plotting structured application results."""

from __future__ import annotations

from cwfm.application.results import AnalysisResult


def require_result(result: AnalysisResult) -> AnalysisResult:
    if not isinstance(result, AnalysisResult):
        raise TypeError("plotting helpers require an AnalysisResult")
    return result

