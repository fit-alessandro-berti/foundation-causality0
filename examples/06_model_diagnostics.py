"""Expose guarded research diagnostics for one supported CWFM analysis.

Scientific target: inspect six estimator weights and estimates, compiled versus
final output, residual gate, mechanism routing, world particles, mismatch, and
exploratory graph logits. It reads the selected discovery case and safe metadata
through the normal task adapter. Identification and support policies remain in
force, and an abstained causal estimate is not restored by diagnostics. World
weights are not interpreted as separated causal worlds and graph logits are not
a validated causal graph. Outputs use the ordinary structured result bundle.
Example: ``python examples/06_model_diagnostics.py --task
network_interference --scenario threshold_saturation``. Use ``--help`` for root,
device, output, seed, dry-run, JSON, and quiet. No truth is read. This implements
the architecture-diagnostics disclosure discussed in the experimental report.
"""

from __future__ import annotations

import argparse

from _common import add_common_arguments, finish, print_dry_run
from cwfm.application import (
    CWFMRunner,
    RepositoryCatalog,
    TASK_METHOD,
    build_repository_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--task", choices=("static_ate", "observed_regime", "network_interference"), default="network_interference")
    parser.add_argument("--scenario", default="threshold_saturation")
    parser.add_argument("--outcome", default="Y1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    method = TASK_METHOD[args.task]
    case = catalog.resolve(method, args.scenario, args.seed)
    query, assumptions = build_repository_request(case, args.task, args.outcome)
    if args.dry_run:
        print_dry_run({"case": case.preview(), "query": query.__dict__}, args.as_json, args.quiet)
        return 0
    result = CWFMRunner.from_project_root(args.project_root, args.device, args.seed).analyze(case, query, assumptions)
    if not args.quiet and not args.as_json and result.expert_results:
        print("RESEARCH DIAGNOSTICS")
        for expert in result.expert_results:
            print(f"{expert.name:16s} estimate={expert.estimate!s:>12} weight={expert.weight:.4f}")
        print(f"Compiled: {result.compiled_estimate}")
        print(f"Correction: {result.diagnostics.residual_correction}")
        print(f"Residual gate: {result.diagnostics.residual_gate}")
        print(f"Final: {result.final_estimate}\n")
    return finish(
        result,
        args,
        args.project_root / "outputs" / "06_model_diagnostics",
        f"python examples/06_model_diagnostics.py --task {args.task} --scenario {args.scenario} --seed {args.seed} --outcome {args.outcome}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
