"""Helpers for reproducible analysis provenance."""

from __future__ import annotations

import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import AnalysisQuery, AssumptionLedger, ObservedCase
from .model_bundle import ModelBundle
from .results import Provenance, _jsonable


def build_provenance(
    case: ObservedCase,
    query: AnalysisQuery,
    assumptions: AssumptionLedger,
    bundle: ModelBundle,
    random_seed: int,
) -> Provenance:
    inputs: list[str] = []
    if case.source:
        for path in (case.source.discovery_path, case.source.metadata_path):
            if path is not None:
                try:
                    inputs.append(str(path.relative_to(bundle.project_root)))
                except ValueError:
                    inputs.append(str(path))
    return Provenance(
        checkpoint_hash=bundle.checkpoint_hash,
        calibration_hash=bundle.calibration_hash,
        model_configuration=bundle.config.to_dict(),
        repository_inputs=inputs,
        selected_variables=list(case.variable_names),
        query=_jsonable(asdict(query)),
        assumptions=_jsonable(asdict(assumptions)),
        software_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        random_seed=random_seed,
    )
