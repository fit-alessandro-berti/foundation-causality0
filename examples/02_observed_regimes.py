"""Determine an observed regime using the calibrated CWFM split rule.

Scientific question: whether coefficients relating ordinary predictors to one
outcome change across an observed candidate variable and threshold. The target
is an outcome-specific structural split, not a causal ATE. Only discovery.npz
and allowlisted metadata are read; truth structures and truth.npz are isolated.
Four ordinary predictors, up to seven screened or explicit candidates, and one
outcome fit the 12-variable model limit. Screening ranks candidates on discovery
data and is recorded in provenance and CSV. Unsupported interpretations include
joint multi-outcome modeling and validated cross-chunk aggregation. Outputs add
candidate screening and threshold-evidence tables. Examples: ``python
examples/02_observed_regimes.py`` and ``python examples/02_observed_regimes.py
--scenario correlated_proxy --candidate-policy screen-then-model``. Use
``--help`` for device, output, seed, dry-run, JSON, and quiet controls. This
implements the observed-regime subsection of the executable CWFM scope.
"""

from __future__ import annotations

import argparse
import shlex

from _common import add_common_arguments, finish, print_dry_run
from cwfm.application import (
    AnalysisStatus,
    AssumptionLedger,
    CWFMRunner,
    RegimeQuery,
    RepositoryCatalog,
    export_analysis,
)


PRESETS = {
    "proxy-ambiguity": "correlated_proxy",
    "nonlinear-misspecification": "global_nonlinearity_misspecified",
    "null-control": "null",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--scenario", default="clean_one_split")
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--outcome", default="Y1")
    parser.add_argument("--ordinary-predictors", nargs="+")
    parser.add_argument("--regime-candidates", nargs="+")
    parser.add_argument(
        "--candidate-policy",
        choices=("explicit", "screen-then-model", "chunked"),
        default="screen-then-model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = PRESETS.get(args.preset, args.scenario)
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    case = catalog.resolve("causal_model_determination", scenario, args.seed)
    metadata = case.safe_metadata()
    features = tuple(metadata["feature_names"])
    default_ordinary = tuple(features[index] for index in metadata.get("ordinary_predictors", [0, 1, 2, 3]))
    ordinary = tuple(args.ordinary_predictors or default_ordinary)
    candidates = tuple(args.regime_candidates or [name for name in features if name not in ordinary])
    selected_outcomes = tuple(metadata["outcome_names"]) if args.outcome == "all" else (args.outcome,)
    queries = [
        RegimeQuery(
            outcome=outcome,
            ordinary_predictors=ordinary,
            regime_candidates=candidates,
            candidate_policy=args.candidate_policy,
        )
        for outcome in selected_outcomes
    ]
    assumptions = AssumptionLedger()
    if args.dry_run:
        print_dry_run(
            {"case": case.preview(), "queries": [query.__dict__ for query in queries], "variable_budget": 12},
            args.as_json,
            args.quiet,
        )
        return 0
    runner = CWFMRunner.from_project_root(args.project_root, args.device, args.seed)
    results = runner.analyze_batch(case, queries, assumptions)
    command = " ".join(shlex.quote(item) for item in [
        "python", "examples/02_observed_regimes.py", "--scenario", scenario,
        "--seed", str(args.seed), "--outcome", args.outcome,
        "--candidate-policy", args.candidate_policy,
    ])
    if len(results) == 1:
        return finish(results[0], args, args.project_root / "outputs" / "02_observed_regimes", command)
    output = args.output_dir or args.project_root / "outputs" / "02_observed_regimes"
    for result in results:
        export_analysis(result, output, command)
    if not args.quiet:
        if args.as_json:
            import json

            print(json.dumps([result.to_dict() for result in results], indent=2))
        else:
            print("Executed separate outcome-specific analyses:")
            for outcome, result in zip(selected_outcomes, results):
                print(f"- {outcome}: {result.status.value}")
    unavailable = {
        AnalysisStatus.MODEL_UNAVAILABLE,
        AnalysisStatus.INVALID_INPUT,
        AnalysisStatus.UNSUPPORTED_QUERY,
    }
    return 2 if any(result.status in unavailable for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
