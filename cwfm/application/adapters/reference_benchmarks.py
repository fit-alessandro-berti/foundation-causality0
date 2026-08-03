"""Public access to stored classical benchmark-evaluator results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceBenchmarkResult:
    method: str
    scenario: str
    seed: int
    backend: str
    cwfm_checkpoint_used: bool
    metrics: dict[str, Any]


def load_reference_result(
    project_root: str | Path,
    method: str,
    scenario: str,
    seed: int,
) -> ReferenceBenchmarkResult:
    """Load an already-evaluated classical result without opening case truth."""

    path = Path(project_root).resolve() / "docs" / "data" / "metrics" / method / "per_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stored reference metrics not found: {path}")
    frame = pd.read_csv(path)
    subset = frame[(frame["scenario"] == scenario) & (frame["seed"] == seed)]
    if subset.empty:
        raise KeyError(f"Stored reference result not found: {method}/{scenario}/seed_{seed:04d}")
    row = subset.iloc[0].to_dict()
    metrics = {
        str(key): (
            None
            if isinstance(value, float) and not np.isfinite(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in row.items()
    }
    return ReferenceBenchmarkResult(
        method=method,
        scenario=scenario,
        seed=seed,
        backend="classical reference evaluator",
        cwfm_checkpoint_used=False,
        metrics=metrics,
    )
