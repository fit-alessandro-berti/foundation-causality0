"""Batch comparable repository cases through the public CWFM runner.

Scientific target: compare answer rates, estimates, interval widths, support
warnings, selected experts, and compiled/final differences across scenarios and
seeds for one supported task. The script reads discovery arrays and allowlisted
metadata only by default. Each outcome is one model call; no multi-output model
is implied. ``--audit-with-truth`` inventories truth arrays only after fitting,
in a separate audit directory, and never changes predictions. Unsupported
cross-task pooling is not performed. Outputs include ``comparison.csv``, one
result bundle per case, and optional audit inventories. Example: ``python
examples/05_compare_repository_cases.py --task network_interference --scenarios
linear_spillover threshold_saturation complete_null --seeds 0 1``. Use
``--help`` for common controls. This corresponds to the held-out comparison and
reproducibility subsection of the paper implementation.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import add_common_arguments, print_dry_run, write_rows
from cwfm.application import (
    CWFMRunner,
    RepositoryCatalog,
    TASK_METHOD,
    build_repository_request,
    export_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--task", choices=tuple(TASK_METHOD), default="network_interference")
    parser.add_argument("--scenarios", nargs="+", default=["linear_spillover", "threshold_saturation", "complete_null"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--outcome", default="Y1")
    parser.add_argument("--audit-with-truth", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    runner = CWFMRunner.from_project_root(args.project_root, args.device, args.seed)
    method = TASK_METHOD[args.task]
    requested = [(scenario, seed) for scenario in args.scenarios for seed in args.seeds]
    if args.dry_run:
        print_dry_run({"task": args.task, "cases": requested}, args.as_json, args.quiet)
        return 0
    output = args.output_dir or args.project_root / "outputs" / "05_compare_repository_cases"
    rows = []
    audit_rows = []
    for scenario, seed in requested:
        case = catalog.resolve(method, scenario, seed)
        query, assumptions = build_repository_request(case, args.task, args.outcome)
        result = runner.analyze(case, query, assumptions)
        export_analysis(result, output / "analyses")
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "status": result.status.value,
                "estimate": result.estimate,
                "interval_width": result.interval[1] - result.interval[0] if result.interval else None,
                "selected_expert": result.selected_expert,
                "compiled_final_difference": (
                    result.final_estimate - result.compiled_estimate
                    if result.final_estimate is not None and result.compiled_estimate is not None
                    else None
                ),
                "warning_count": len(result.warnings),
            }
        )
        if args.audit_with_truth:
            truth_path = case.directory / "truth.npz"
            with np.load(truth_path, allow_pickle=False) as truth:
                for name in truth.files:
                    audit_rows.append(
                        {"scenario": scenario, "seed": seed, "array": name, "shape": list(truth[name].shape)}
                    )
    write_rows(output / "comparison.csv", rows)
    if audit_rows:
        (output / "audit").mkdir(parents=True, exist_ok=True)
        (output / "audit" / "truth_inventory.json").write_text(json.dumps(audit_rows, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(rows, indent=2) if args.as_json else f"Saved {len(rows)} comparisons to {output / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
