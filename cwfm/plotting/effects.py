"""Effect and routing plotting data."""

from __future__ import annotations

import pandas as pd

from cwfm.application.results import AnalysisResult

from .common import require_result


def expert_frame(result: AnalysisResult) -> pd.DataFrame:
    result = require_result(result)
    rows = [
        {
            "expert": item.name,
            "estimate": item.estimate,
            "standard_error": item.standard_error,
            "weight": item.weight,
            "available": item.available,
        }
        for item in result.expert_results
    ]
    if result.compiled_estimate is not None:
        rows.append(
            {
                "expert": "compiled_cwfm",
                "estimate": result.compiled_estimate,
                "standard_error": None,
                "weight": None,
                "available": True,
            }
        )
    if result.final_estimate is not None:
        rows.append(
            {
                "expert": "final_cwfm",
                "estimate": result.final_estimate,
                "standard_error": None,
                "weight": None,
                "available": True,
            }
        )
    return pd.DataFrame(rows)

