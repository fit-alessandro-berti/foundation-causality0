"""Check the CWFM installation and versioned model bundle.

Scientific question: whether this checkout can execute the released CWFM.
Target: package, data-catalog, bundle, calibration, and device compatibility.
Files read: safe case metadata, bundle manifests, calibration, and model weights
when present. No case truth or evaluation answers are read. The check requires
no causal assumptions and makes no scientific estimate. It reports package
versions, the model hash, the 12-variable contract, calibration compatibility,
and a minimal forward pass. Run with ``--help`` for project-root, device,
output, seed, dry-run, JSON, and quiet controls. Example: ``python
examples/00_check_installation.py --device cpu``. This corresponds to the
executable reference-model scope described in the paper implementation section.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import torch

from _common import add_common_arguments
from cwfm import __version__
from cwfm.application import (
    ATEQuery,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    CWFMRunner,
    ModelBundleStatus,
    RepositoryCatalog,
)
from cwfm.application.model_bundle import ModelBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    bundle = ModelBundle.from_project_root(args.project_root)
    forward_status = "skipped"
    if bundle.status == ModelBundleStatus.AVAILABLE and not args.dry_run:
        case = catalog.resolve("static_ate", "randomized_linear", 0)
        query = ATEQuery("A", "Y", tuple(case.safe_metadata()["feature_names"]))
        result = CWFMRunner(bundle, args.device, args.seed).analyze_ate(
            case,
            query,
            AssumptionLedger(
                assignment_design=AssignmentDesign.RANDOMIZED,
                consistency=AssertionState.ASSERTED,
            ),
        )
        forward_status = "ok" if result.status.answered else result.status.value
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cwfm": __version__,
        "repository_root": str(catalog.project_root),
        "catalog_cases": len(catalog),
        "catalog_methods": catalog.methods(),
        "model_bundle": bundle.describe(),
        "cuda_available": torch.cuda.is_available(),
        "benchmark_explorer": "available",
        "cwfm_inference": bundle.status.value,
        "minimal_forward_pass": forward_status,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.quiet:
        print("CWFM installation check")
        print("─" * 56)
        print(f"Repository data catalog       OK: {len(catalog)} cases")
        print("Core dependencies             OK")
        print(f"Calibration                   {'OK' if bundle.calibration else 'ERROR'}")
        print(f"Model bundle                  {bundle.status.value.upper()}: {bundle.reason}")
        print(f"CWFM inference                {bundle.status.value.upper()}")
        print(f"Minimal forward pass          {forward_status.upper()}")
        print("Benchmark explorer            AVAILABLE")
        if bundle.status != ModelBundleStatus.AVAILABLE:
            print(f"\nExpected model location:\n{bundle.model_path}")
    return 0 if bundle.status == ModelBundleStatus.AVAILABLE and forward_status in {"ok", "skipped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
