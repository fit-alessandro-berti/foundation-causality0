"""Checkpoint-bound calibration loading and interval rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Task


@dataclass(frozen=True)
class Calibration:
    normalized_radius: dict[str, float]
    absolute_radius: dict[str, dict[str, float]]
    counts: dict[str, int]
    regime: dict[str, float | int]
    interval_level: float = 0.90

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Calibration":
        return cls(
            normalized_radius={str(k): float(v) for k, v in raw["normalized_radius"].items()},
            absolute_radius={
                str(task): {str(name): float(value) for name, value in values.items()}
                for task, values in raw.get("absolute_radius", {}).items()
            },
            counts={str(k): int(v) for k, v in raw.get("counts", {}).items()},
            regime={str(k): v for k, v in raw["regime"].items()},
            interval_level=float(raw.get("interval_level", 0.90)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "calibration" in raw:
            raw = raw["calibration"]
        return cls.from_dict(raw)

    @property
    def regime_evidence_threshold(self) -> float:
        return float(self.regime["evidence_threshold"])

    def radius(
        self,
        task: Task,
        selected_expert: str,
        expert_estimates: list[float],
    ) -> float:
        disagreement = float(np.std(expert_estimates))
        bucket = "high_disagreement" if disagreement > 0.25 else "low_disagreement"
        key = f"{task.value}|{selected_expert}|{bucket}"
        if key in self.normalized_radius:
            return self.normalized_radius[key]
        return self.normalized_radius[task.value]

    def interval(
        self,
        estimate: float,
        predictive_scale: float,
        task: Task,
        selected_expert: str,
        expert_estimates: list[float],
    ) -> tuple[float, float]:
        half_width = (
            self.radius(task, selected_expert, expert_estimates) * predictive_scale
        )
        return estimate - half_width, estimate + half_width
