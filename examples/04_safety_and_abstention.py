"""Demonstrate assumption- and support-driven CWFM abstention.

Scientific question: which requested causal contrasts are reportable under the
declared assumptions and observed support. This is a policy demonstration, not
a single estimand. It reads only the same truth-safe ATE and interference cases
used by the public runner. Cases cover randomized ATE, observational ATE without
identification, adequate network support, poor exposure support, and an unknown
mapping. Unsupported principal estimates are suppressed. Outputs include one
structured result per case and ``safety_matrix.csv``; if weights are absent,
otherwise-answerable rows explicitly say MODEL_UNAVAILABLE. Example: ``python
examples/04_safety_and_abstention.py``. Common ``--help`` options include root,
device, output, seed, dry-run, JSON, and quiet. No truth file is read. This
implements the formal identification-gate and abstention policy described in
the executable-model safety subsection.
"""

from __future__ import annotations

import argparse
import json

from _common import add_common_arguments, print_dry_run, write_rows
from cwfm.application import (
    ATEQuery,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    CWFMRunner,
    InterferenceQuery,
    RepositoryCatalog,
    SupportState,
    export_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    runner = CWFMRunner.from_project_root(args.project_root, args.device, args.seed)
    ate_random = catalog.resolve("static_ate", "randomized_linear", 0)
    ate_observed = catalog.resolve("static_ate", "observational_nonlinear", 0)
    network_good = catalog.resolve("causal_interference_detection", "linear_spillover", args.seed)
    network_poor = catalog.resolve("causal_interference_detection", "poor_exposure_overlap", args.seed)
    ate_features = tuple(ate_random.safe_metadata()["feature_names"])
    network_features = tuple(network_good.safe_metadata()["feature_names"])
    randomized = AssumptionLedger(
        assignment_design=AssignmentDesign.RANDOMIZED,
        consistency=AssertionState.ASSERTED,
        treatment_support=SupportState.UNKNOWN,
    )
    network_assumptions = AssumptionLedger(
        assignment_design=AssignmentDesign.RANDOMIZED,
        consistency=AssertionState.ASSERTED,
        treatment_support=SupportState.UNKNOWN,
        exposure_mapping_predeclared=True,
        interference_restricted_to_observed_graph=True,
    )
    cases = [
        ("Randomized ATE", ate_random, ATEQuery("A", "Y", ate_features), randomized),
        (
            "Observational ATE, assumptions unknown",
            ate_observed,
            ATEQuery("A", "Y", ate_features),
            AssumptionLedger(
                assignment_design=AssignmentDesign.OBSERVATIONAL,
                consistency=AssertionState.ASSERTED,
            ),
        ),
        (
            "Network, adequate exposure support",
            network_good,
            InterferenceQuery("A", "G", "Y1", network_features),
            network_assumptions,
        ),
        (
            "Network, poor exposure support",
            network_poor,
            InterferenceQuery("A", "G", "Y1", network_features),
            network_assumptions,
        ),
        (
            "Network, unknown exposure mapping",
            network_good,
            InterferenceQuery("A", "G", "Y1", network_features),
            AssumptionLedger(
                assignment_design=AssignmentDesign.RANDOMIZED,
                consistency=AssertionState.ASSERTED,
                exposure_mapping_predeclared=False,
                interference_restricted_to_observed_graph=True,
            ),
        ),
    ]
    if args.dry_run:
        print_dry_run({"cases": [name for name, *_ in cases]}, args.as_json, args.quiet)
        return 0
    output = args.output_dir or args.project_root / "outputs" / "04_safety_and_abstention"
    rows = []
    for name, case, query, assumptions in cases:
        result = runner.analyze(case, query, assumptions)
        export_analysis(result, output / "analyses")
        rows.append(
            {
                "case": name,
                "identification": "established" if result.status.answered else "not established or unavailable",
                "support": "inadequate" if "INADEQUATE_SUPPORT" in result.status.value else "adequate or unchecked",
                "result": result.status.value,
                "estimate_reported": result.estimate is not None,
            }
        )
    write_rows(output / "safety_matrix.csv", rows)
    if not args.quiet:
        if args.as_json:
            print(json.dumps(rows, indent=2))
        else:
            print("SAFETY AND ABSTENTION MATRIX")
            for row in rows:
                print(f"{row['case']}: {row['result']}")
            print(f"\nSaved: {output / 'safety_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

