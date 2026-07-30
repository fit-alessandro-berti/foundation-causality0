\
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 220) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def symmetric_limit(*arrays: np.ndarray, floor: float = 0.1) -> float:
    finite = [
        np.nanmax(np.abs(array))
        for array in arrays
        if np.size(array) and np.any(np.isfinite(array))
    ]
    return max(finite, default=floor)


def annotate_matrix(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    row_labels: Iterable[str],
    col_labels: Iterable[str],
    title: str,
    limit: float | None = None,
    fmt: str = ".2f",
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if limit is None:
        limit = symmetric_limit(matrix)
    image = axis.imshow(matrix, aspect="auto", vmin=-limit, vmax=limit, cmap="coolwarm")
    axis.set_title(title)
    axis.set_xticks(range(matrix.shape[1]), labels=list(col_labels))
    axis.set_yticks(range(matrix.shape[0]), labels=list(row_labels))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axis.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def flatten_tree_lines(
    node: dict,
    *,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    prefix: str = "",
) -> list[str]:
    if node.get("leaf", False):
        return [f"{prefix}leaf (n={int(node['n'])})"]
    feature = int(node["feature"])
    raw_threshold = float(node["threshold"] * x_scale[feature] + x_mean[feature])
    lines = [f"{prefix}X{feature + 1} <= {raw_threshold:.3f}"]
    lines.extend(
        flatten_tree_lines(
            node["left"], x_mean=x_mean, x_scale=x_scale, prefix=prefix + "  L: "
        )
    )
    lines.extend(
        flatten_tree_lines(
            node["right"], x_mean=x_mean, x_scale=x_scale, prefix=prefix + "  R: "
        )
    )
    return lines
