\
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CaseData:
    discovery: dict[str, np.ndarray]
    evaluation: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    metadata: dict[str, Any]
    directory: Path


def add_project_to_path(project_root: Path) -> None:
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark artifact: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_case(project_root: Path, method: str, scenario: str, seed: int) -> CaseData:
    directory = (
        project_root
        / "docs"
        / "data"
        / "generated"
        / method
        / scenario
        / f"seed_{seed:04d}"
    )
    if not directory.exists():
        raise FileNotFoundError(
            f"Case directory not found: {directory}. "
            "Run the benchmark first or choose an available seed."
        )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return CaseData(
        discovery=load_npz(directory / "discovery.npz"),
        evaluation=load_npz(directory / "evaluation.npz"),
        truth=load_npz(directory / "truth.npz"),
        metadata=metadata,
        directory=directory,
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_table(
    frame: pd.DataFrame,
    stem: Path,
    *,
    index: bool = False,
    float_format: str = "%.3f",
) -> dict[str, str]:
    """Save one table as CSV, Markdown and LaTeX."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    tex_path = stem.with_suffix(".tex")
    frame.to_csv(csv_path, index=index)
    md_path.write_text(frame.to_markdown(index=index) + "\n", encoding="utf-8")
    tex_path.write_text(
        frame.to_latex(index=index, float_format=float_format),
        encoding="utf-8",
    )
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
    }


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
