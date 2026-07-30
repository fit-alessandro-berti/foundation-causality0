\
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _metrics_frame(project_root: Path, method: str) -> pd.DataFrame:
    path = project_root / "docs" / "data" / "metrics" / method / "per_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return pd.read_csv(path, keep_default_na=False)


def select_median_seed(
    project_root: Path,
    method: str,
    scenario: str,
    metric: str,
    *,
    tie_break: str = "min",
) -> int:
    frame = _metrics_frame(project_root, method)
    subset = frame.loc[frame["scenario"] == scenario, ["seed", metric]].copy()
    if subset.empty:
        raise ValueError(f"No rows for {method}/{scenario}")
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset[np.isfinite(subset[metric])]
    if subset.empty:
        raise ValueError(f"No finite values for {method}/{scenario}/{metric}")
    median = float(subset[metric].median())
    distance = np.abs(subset[metric].to_numpy(dtype=float) - median)
    minimum = float(np.min(distance))
    candidates = subset.loc[np.isclose(distance, minimum), "seed"].astype(int)
    if tie_break == "max":
        return int(candidates.max())
    if tie_break != "min":
        raise ValueError("tie_break must be 'min' or 'max'")
    return int(candidates.min())


def select_flagged_seed(
    project_root: Path,
    method: str,
    scenario: str,
    flag_metric: str,
    *,
    target: float = 1.0,
    tie_break: str = "min",
) -> int:
    frame = _metrics_frame(project_root, method)
    values = pd.to_numeric(frame[flag_metric], errors="coerce")
    subset = frame.loc[
        (frame["scenario"] == scenario) & np.isclose(values, target),
        ["seed"],
    ]
    if subset.empty:
        raise ValueError(
            f"No seed for {method}/{scenario} with {flag_metric}={target}"
        )
    seeds = subset["seed"].astype(int)
    return int(seeds.max() if tie_break == "max" else seeds.min())
