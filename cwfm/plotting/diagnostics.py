"""Research-diagnostic plotting data with interpretation guards."""

from __future__ import annotations

import pandas as pd

from cwfm.application.results import AnalysisResult

from .common import require_result


def world_particle_frame(result: AnalysisResult) -> pd.DataFrame:
    result = require_result(result)
    return pd.DataFrame(
        {
            "particle": list(range(len(result.diagnostics.world_weights))),
            "weight": result.diagnostics.world_weights,
            "estimate": result.diagnostics.world_particle_estimates,
        }
    )
