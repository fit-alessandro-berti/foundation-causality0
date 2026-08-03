"""Inspect stored classical reference-evaluator benchmark results.

Scientific target: the method-specific latent grouping, latent graph, temporal
break, temporal VARX, regime, or interference benchmark metric recorded for one
scenario and seed. This script is explicitly not a CWFM prediction. It reads
safe discovery metadata and the stored ``per_seed.csv`` evaluator result; it
does not open truth.npz. Interpretations are limited to the classical backend's
documented metrics. Outputs include ``reference_result.json`` and the safe case
preview. Example: ``python examples/07_reference_benchmark_atlas.py --method
temporal_split_detection --scenario one_coefficient_break --seed 0``. Common
``--help`` controls include root, device, output, seed, dry-run, JSON, and quiet.
This exposes the broader benchmark environment while preserving the executable
CWFM scope distinction in the paper.
"""

from __future__ import annotations

import argparse
import json

from _common import add_common_arguments, print_dry_run
from cwfm.application import RepositoryCatalog, load_reference_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--method",
        choices=(
            "causal_model_determination",
            "causal_interference_detection",
            "latent_variable_determination",
            "latent_variable_determination2",
            "temporal_split_detection",
            "temporal_split_detection_method",
        ),
        default="temporal_split_detection",
    )
    parser.add_argument("--scenario", default="one_coefficient_break")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    case = catalog.resolve(args.method, args.scenario, args.seed)
    if args.dry_run:
        print_dry_run(case.preview(), args.as_json, args.quiet)
        return 0
    result = load_reference_result(args.project_root, args.method, args.scenario, args.seed)
    payload = {
        "backend": result.backend,
        "cwfm_checkpoint_used": result.cwfm_checkpoint_used,
        "method": result.method,
        "scenario": result.scenario,
        "seed": result.seed,
        "metrics": result.metrics,
        "safe_preview": case.preview(),
    }
    output = args.output_dir or args.project_root / "outputs" / "07_reference_benchmark_atlas"
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Backend: classical reference evaluator")
            print("CWFM checkpoint used: no")
            print(json.dumps(result.metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
