"""Observed-regime plotting data."""

from __future__ import annotations

import pandas as pd

from cwfm.application.results import AnalysisResult

from .common import require_result


def regime_evidence_frame(result: AnalysisResult) -> pd.DataFrame:
    result = require_result(result)
    rows = result.regime_result.evidence_table if result.regime_result else []
    return pd.DataFrame(rows, columns=["variable", "quantile", "threshold", "evidence_score"])

