"""Versioned, integrity-checked CWFM model bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion

from ..model import CWFM, CWFMConfig
from .calibration import Calibration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelBundleStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass
class ModelBundle:
    project_root: Path
    bundle_dir: Path
    model_path: Path
    config_path: Path
    calibration_path: Path
    manifest_path: Path
    status: ModelBundleStatus
    reason: str
    config: CWFMConfig
    calibration: Calibration | None
    manifest: dict[str, Any]
    checkpoint_hash: str | None = None
    calibration_hash: str | None = None

    @classmethod
    def from_project_root(cls, project_root: str | Path = ".") -> "ModelBundle":
        root = Path(project_root).resolve()
        bundle_dir = root / "artifacts" / "cwfm" / "release"
        model_path = bundle_dir / "model_state.pt"
        config_path = bundle_dir / "model_config.json"
        calibration_path = bundle_dir / "calibration.json"
        manifest_path = bundle_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        if config_path.exists():
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            checkpoint_manifest = root / "artifacts" / "cwfm" / "checkpoint_manifest.json"
            raw = json.loads(checkpoint_manifest.read_text(encoding="utf-8")) if checkpoint_manifest.exists() else {}
            raw_config = raw.get("configuration", {})
        config_fields = CWFMConfig.__dataclass_fields__
        config = CWFMConfig(**{key: raw_config[key] for key in config_fields if key in raw_config})

        calibration = None
        calibration_hash = None
        if calibration_path.exists():
            calibration_hash = sha256_file(calibration_path)
            calibration = Calibration.load(calibration_path)
        else:
            results_manifest = root / "artifacts" / "cwfm" / "results" / "manifest.json"
            if results_manifest.exists():
                calibration = Calibration.load(results_manifest)
                calibration_hash = sha256_file(results_manifest)

        expected_calibration_hash = manifest.get("calibration_sha256")
        calibration_problems = []
        if calibration is None:
            calibration_problems.append("calibration.json is missing")
        elif expected_calibration_hash and calibration_hash != expected_calibration_hash:
            calibration_problems.append("calibration SHA-256 does not match manifest")
        if calibration_problems:
            return cls(
                root,
                bundle_dir,
                model_path,
                config_path,
                calibration_path,
                manifest_path,
                ModelBundleStatus.INVALID,
                "; ".join(calibration_problems),
                config,
                calibration,
                manifest,
                calibration_hash=calibration_hash,
            )
        if not model_path.exists():
            return cls(
                root,
                bundle_dir,
                model_path,
                config_path,
                calibration_path,
                manifest_path,
                ModelBundleStatus.UNAVAILABLE,
                f"model_state.pt was not found at {model_path}",
                config,
                calibration,
                manifest,
                calibration_hash=calibration_hash,
            )
        checkpoint_hash = sha256_file(model_path)
        expected_model_hash = manifest.get("checkpoint_sha256")
        problems = []
        if expected_model_hash and checkpoint_hash != expected_model_hash:
            problems.append("checkpoint SHA-256 does not match manifest")
        status = ModelBundleStatus.INVALID if problems else ModelBundleStatus.AVAILABLE
        return cls(
            root,
            bundle_dir,
            model_path,
            config_path,
            calibration_path,
            manifest_path,
            status,
            "; ".join(problems) if problems else "model and calibration are available",
            config,
            calibration,
            manifest,
            checkpoint_hash=checkpoint_hash,
            calibration_hash=calibration_hash,
        )

    def load_model(self, device: torch.device) -> CWFM:
        if self.status != ModelBundleStatus.AVAILABLE:
            raise RuntimeError(self.reason)
        with torch.serialization.safe_globals([TorchVersion]):
            saved = torch.load(self.model_path, map_location=device, weights_only=True)
        state_dict = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
        model = CWFM(self.config).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def describe(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "expected_model_path": str(self.model_path),
            "checkpoint_hash": self.checkpoint_hash,
            "calibration_hash": self.calibration_hash,
            "configuration": self.config.to_dict(),
            "supported_tasks": self.manifest.get(
                "supported_tasks",
                ["static_ate", "observed_regime", "network_interference"],
            ),
            "maximum_variables": self.config.max_variables,
        }
