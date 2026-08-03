"""Controlled repository catalog with a metadata allowlist and truth firewall."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .contracts import CaseSource, Task


METHOD_TASKS: dict[str, Task | None] = {
    "static_ate": Task.STATIC_ATE,
    "causal_model_determination": Task.OBSERVED_REGIME,
    "causal_interference_detection": Task.NETWORK_INTERFERENCE,
    "latent_variable_determination": None,
    "latent_variable_determination2": None,
    "temporal_split_detection": None,
    "temporal_split_detection_method": None,
}

SAFE_METADATA: dict[str, frozenset[str]] = {
    "static_ate": frozenset(
        {"feature_names", "outcome_names", "assignment_design", "method", "seed"}
    ),
    "causal_model_determination": frozenset(
        {"feature_names", "outcome_names", "ordinary_predictors", "method", "seed"}
    ),
    "causal_interference_detection": frozenset(
        {
            "feature_names",
            "outcome_names",
            "outcome_types",
            "assignment_design",
            "assignment_metadata",
            "cluster_size",
            "n_clusters",
            "observed_confounders",
            "primary_mapping",
            "sensitivity_mappings",
            "target_levels",
            "method",
            "seed",
        }
    ),
    "latent_variable_determination": frozenset(
        {"feature_names", "outcome_names", "method", "seed", "target"}
    ),
    "latent_variable_determination2": frozenset(
        {"feature_names", "outcome_names", "method", "seed", "primary_target"}
    ),
    "temporal_split_detection": frozenset(
        {"feature_names", "outcome_names", "method", "seed", "transition_window"}
    ),
    "temporal_split_detection_method": frozenset(
        {"feature_names", "outcome_names", "method", "seed", "lag_order_fitted"}
    ),
}


@dataclass(frozen=True)
class RepositoryCase:
    project_root: Path
    method: str
    scenario: str
    seed: int
    directory: Path
    task: Task | None
    backend: str

    @property
    def discovery_path(self) -> Path:
        return self.directory / "discovery.npz"

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def source(self) -> CaseSource:
        return CaseSource(
            method=self.method,
            scenario=self.scenario,
            seed=self.seed,
            discovery_path=self.discovery_path if self.discovery_path.exists() else None,
            metadata_path=self.metadata_path if self.metadata_path.exists() else None,
            backend=self.backend,
        )

    def safe_metadata(self) -> dict[str, Any]:
        """Read only public descriptive fields; never open truth.npz."""

        if not self.metadata_path.exists():
            return {}
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        allowed = SAFE_METADATA.get(self.method, frozenset())
        return {key: raw[key] for key in allowed if key in raw}

    def load_observed(self) -> dict[str, np.ndarray]:
        """Load discovery arrays or a deterministic model-native ATE fixture."""

        if self.discovery_path.exists():
            with np.load(self.discovery_path, allow_pickle=False) as archive:
                return {name: np.asarray(archive[name]) for name in archive.files}
        generator_path = self.directory / "generator.json"
        if self.method == "static_ate" and generator_path.exists():
            from ..data import generate_ate

            spec = json.loads(generator_path.read_text(encoding="utf-8"))
            episode = generate_ate(
                int(spec.get("seed", self.seed)),
                str(spec["scenario"]),
                n=int(spec.get("sample_size", 96)),
                p=int(spec.get("covariates", 6)),
            )
            p = episode.values.shape[1] - 2
            return {
                "X": episode.values[:, :p].copy(),
                "A": episode.values[:, p].copy(),
                "Y": episode.values[:, p + 1].copy(),
            }
        raise FileNotFoundError(f"No discovery input found for {self.directory}")

    def preview(self) -> dict[str, Any]:
        arrays = self.load_observed()
        metadata = self.safe_metadata()
        return {
            "method": self.method,
            "scenario": self.scenario,
            "seed": self.seed,
            "backend": self.backend,
            "task": self.task.value if self.task else None,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "missing": int(np.isnan(value).sum())
                    if np.issubdtype(value.dtype, np.number)
                    else None,
                }
                for name, value in arrays.items()
            },
            "metadata": metadata,
        }


class RepositoryCatalog:
    """Index controlled repository cases without accepting arbitrary paths."""

    def __init__(self, project_root: Path, cases: list[RepositoryCase]) -> None:
        self.project_root = project_root
        self._cases = tuple(cases)

    @classmethod
    def from_project_root(cls, project_root: str | Path = ".") -> "RepositoryCatalog":
        root = Path(project_root).resolve()
        cases: list[RepositoryCase] = []
        locations = [root / "docs" / "data" / "generated", root / "examples" / "data"]
        for location in locations:
            if not location.exists():
                continue
            for method_dir in sorted(path for path in location.iterdir() if path.is_dir()):
                method = method_dir.name
                if method not in METHOD_TASKS:
                    continue
                for scenario_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
                    for seed_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir()):
                        if not seed_dir.name.startswith("seed_"):
                            continue
                        if not (
                            (seed_dir / "discovery.npz").exists()
                            or (seed_dir / "generator.json").exists()
                        ):
                            continue
                        try:
                            seed = int(seed_dir.name.removeprefix("seed_"))
                        except ValueError:
                            continue
                        task = METHOD_TASKS[method]
                        backend = (
                            "CWFM checkpoint" if task is not None else "Classical reference evaluator"
                        )
                        cases.append(
                            RepositoryCase(
                                root,
                                method,
                                scenario_dir.name,
                                seed,
                                seed_dir,
                                task,
                                backend,
                            )
                        )
        return cls(root, cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[RepositoryCase]:
        return iter(self._cases)

    def methods(self) -> list[str]:
        return sorted({case.method for case in self._cases})

    def scenarios(self, method: str) -> list[str]:
        return sorted({case.scenario for case in self._cases if case.method == method})

    def seeds(self, method: str, scenario: str) -> list[int]:
        return sorted(
            case.seed
            for case in self._cases
            if case.method == method and case.scenario == scenario
        )

    def resolve(self, method: str, scenario: str, seed: int = 0) -> RepositoryCase:
        for case in self._cases:
            if (case.method, case.scenario, case.seed) == (method, scenario, seed):
                return case
        raise KeyError(f"Repository case not found: {method}/{scenario}/seed_{seed:04d}")

    get = resolve

    def for_task(self, task: Task) -> list[RepositoryCase]:
        return [case for case in self._cases if case.task == task]

