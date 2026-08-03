"""Shared adapter utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def column_indices(names: tuple[str, ...], selected: Iterable[str]) -> list[int]:
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(f"Unknown selected variables: {', '.join(missing)}")
    return [lookup[name] for name in selected]


def impute_for_screening(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    means = np.nanmean(result, axis=0)
    rows, columns = np.nonzero(np.isnan(result))
    result[rows, columns] = np.nan_to_num(means)[columns]
    return result

