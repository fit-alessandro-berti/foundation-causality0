from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DESIGN_OBSERVED,
    ROLE_COVARIATE,
    ROLE_OUTCOME,
    ROLE_TREATMENT,
    TASK_ATE,
    Episode,
)


ENERGY_COLUMNS = (
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T_out",
    "RH_out",
)


@lru_cache(maxsize=2)
def load_appliances_energy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the fixed real-measurement covariates and prognostic response."""
    frame = pd.read_csv(Path(path), compression="zip")
    missing = set(ENERGY_COLUMNS + ("Appliances",)) - set(frame.columns)
    if missing:
        raise ValueError(f"External energy archive is missing columns: {missing}")
    covariates = frame.loc[:, ENERGY_COLUMNS].to_numpy(dtype=float)
    covariates = (covariates - covariates.mean(0)) / covariates.std(0)
    prognostic = np.log1p(frame["Appliances"].to_numpy(dtype=float))
    prognostic = (prognostic - prognostic.mean()) / prognostic.std()
    return covariates, prognostic


def generate_appliances_energy_episode(
    seed: int,
    path: Path = Path("data/external/appliances-energy.zip"),
    stream: str = "test",
    n: int = 128,
) -> Episode:
    """Create a semi-synthetic ATE episode over measured building data.

    Even source rows form the calibration pool and odd source rows form the
    evaluation pool. Treatment assignment, heterogeneity, and noise are fixed
    before evaluation and depend only on the selected real covariates and the
    episode seed.
    """
    if stream not in {"calibration", "test"}:
        raise ValueError("stream must be 'calibration' or 'test'")
    covariates, prognostic = load_appliances_energy(str(path))
    pool = np.arange(len(covariates))[0 if stream == "calibration" else 1 :: 2]
    rng = np.random.default_rng(seed)
    source_rows = rng.choice(pool, size=n, replace=False)
    x = covariates[source_rows]
    base = prognostic[source_rows]
    propensity = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                0.45 * x[:, 0]
                - 0.35 * x[:, 1]
                + 0.25 * x[:, 6]
                - 0.15 * x[:, 7],
                -4.0,
                4.0,
            )
        )
    )
    treatment = rng.binomial(1, propensity).astype(float)
    effect = 0.40 + 0.18 * np.tanh(x[:, 2]) - 0.12 * np.tanh(x[:, 7])
    outcome = base + effect * treatment + rng.normal(0.0, 0.35, n)
    values = np.column_stack([x, treatment, outcome]).astype(np.float32)
    roles = np.asarray(
        [ROLE_COVARIATE] * len(ENERGY_COLUMNS)
        + [ROLE_TREATMENT, ROLE_OUTCOME],
        dtype=np.int64,
    )
    graph = np.zeros((len(roles), len(roles)), dtype=np.float32)
    graph[: len(ENERGY_COLUMNS), -2:] = 1
    graph[-2, -1] = 1

    row_order = rng.permutation(n)
    values = values[row_order]
    source_rows = source_rows[row_order]
    covariate_order = rng.permutation(len(ENERGY_COLUMNS))
    column_order = np.r_[covariate_order, len(ENERGY_COLUMNS), len(roles) - 1]
    values = values[:, column_order]
    roles = roles[column_order]
    graph = graph[np.ix_(column_order, column_order)]

    return Episode(
        values=values,
        roles=roles,
        task=TASK_ATE,
        design=DESIGN_OBSERVED,
        target=float(effect.mean()),
        identified=1,
        supported=1,
        structure_target=len(roles),
        mechanism=1,
        route_flexible=1.0,
        graph=graph,
        adjacency=np.zeros((n, n), dtype=np.float32),
        scenario="external_appliances_energy",
        seed=seed,
        cluster_ids=source_rows.astype(np.int64),
        metadata={
            "generator_version": 1,
            "template_id": "external:appliances_energy",
            "mechanism_family": "semi_synthetic_energy_ate",
            "graph_family": "iid",
            "sample_size": n,
            "variable_count": len(roles),
            "signal_strength": "regular",
            "overlap_bucket": "regular",
            "source_stream": stream,
            "source": "UCI Appliances Energy Prediction, DOI 10.24432/C5VC8G",
        },
    )

