"""Network support plotting data."""

from __future__ import annotations

import pandas as pd

from cwfm.application.results import AnalysisResult

from .common import require_result


def exposure_support_frame(result: AnalysisResult) -> pd.DataFrame:
    result = require_result(result)
    support = result.interference_result.support if result.interference_result else {}
    rows = [
        {"own_treatment": level, **values}
        for level, values in support.get("by_treatment", {}).items()
    ]
    return pd.DataFrame(rows)

