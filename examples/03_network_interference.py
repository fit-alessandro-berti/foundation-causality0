"""Analyze a mapped peer-exposure contrast on an observed network.

Scientific question: the average change in one outcome when mapped peer
exposure moves from g0 to g1 while each unit's observed own treatment is held in
the averaging distribution. Files read are discovery.npz and allowlisted
network metadata; truth mappings and truth.npz are never read. One treatment,
one computed exposure, one outcome, six covariates, and optional degree fit the
12-variable limit. Identification additionally requires a predeclared exposure
mapping and interference restricted to the observed graph; empirical support is
checked within own-treatment groups. This is not a direct-treatment or joint
policy effect. Outputs include network support, expert routing, diagnostics,
and provenance. Examples: ``python examples/03_network_interference.py`` and
``python examples/03_network_interference.py --scenario misspecified_exposure
--compare-mappings``. Use ``--help`` for all common controls. This implements
the network-interference subsection of the executable CWFM scope.
"""

from __future__ import annotations

import argparse
import csv
import shlex

from _common import add_common_arguments, finish, print_dry_run
from cwfm.application import (
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    CWFMRunner,
    InterferenceQuery,
    RepositoryCatalog,
    SupportState,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--scenario", default="linear_spillover")
    parser.add_argument("--outcome", default="Y1")
    parser.add_argument(
        "--mapping", choices=("weighted", "unweighted", "distance_two"), default="weighted"
    )
    parser.add_argument("--g0", type=float, default=0.25)
    parser.add_argument("--g1", type=float, default=0.75)
    parser.add_argument("--compare-mappings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    case = catalog.resolve("causal_interference_detection", args.scenario, args.seed)
    metadata = case.safe_metadata()
    covariates = tuple(metadata["feature_names"])
    query = InterferenceQuery(
        treatment="A",
        exposure="G",
        outcome=args.outcome,
        covariates=covariates,
        exposure_low=args.g0,
        exposure_high=args.g1,
        exposure_mapping=args.mapping,
    )
    assumptions = AssumptionLedger(
        assignment_design=(
            AssignmentDesign.RANDOMIZED
            if str(metadata.get("assignment_design", "")).startswith("randomized")
            else AssignmentDesign.OBSERVATIONAL
        ),
        no_unmeasured_confounding=AssertionState.ASSERTED,
        consistency=AssertionState.ASSERTED,
        treatment_support=SupportState.UNKNOWN,
        exposure_mapping_predeclared=True,
        interference_restricted_to_observed_graph=True,
    )
    if args.dry_run:
        print_dry_run(
            {"case": case.preview(), "query": query.__dict__, "assumptions": str(assumptions)},
            args.as_json,
            args.quiet,
        )
        return 0
    runner = CWFMRunner.from_project_root(args.project_root, args.device, args.seed)
    result = runner.analyze_interference(case, query, assumptions)
    command = " ".join(shlex.quote(item) for item in [
        "python", "examples/03_network_interference.py", "--scenario", args.scenario,
        "--seed", str(args.seed), "--outcome", args.outcome, "--mapping", args.mapping,
        "--g0", str(args.g0), "--g1", str(args.g1),
    ])
    code = finish(
        result, args, args.project_root / "outputs" / "03_network_interference", command
    )
    if args.compare_mappings:
        rows = []
        for mapping in ("weighted", "unweighted", "distance_two"):
            sensitivity_query = InterferenceQuery(
                "A", "G", args.outcome, covariates, args.g0, args.g1, mapping
            )
            sensitivity = runner.analyze_interference(case, sensitivity_query, assumptions)
            rows.append(
                {
                    "mapping": mapping,
                    "status": sensitivity.status.value,
                    "estimate": sensitivity.estimate,
                    "lower": sensitivity.interval[0] if sensitivity.interval else None,
                    "upper": sensitivity.interval[1] if sensitivity.interval else None,
                }
            )
        output = (args.output_dir or args.project_root / "outputs" / "03_network_interference") / result.analysis_id
        output.mkdir(parents=True, exist_ok=True)
        with (output / "mapping_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

