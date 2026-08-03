"""Run a truth-safe static average-treatment-effect analysis.

Scientific question: the average effect of binary A=1 versus A=0 on Y. The
estimand is an ATE when randomization or no-unmeasured-confounding plus
consistency is declared. Files read are the selected model-native fixture
specification and safe metadata; generator truth is discarded and ``truth.npz``
is never opened. Variables are up to ten covariates, A, and one Y. Without
identification declarations the runner abstains; poor overlap also causes
abstention. Unsupported interpretations include hidden-confounder discovery,
multiple treatments, and multiple outcomes in one call. Outputs include JSON,
CSV expert routing, diagnostics, assumptions, and provenance. Examples:
``python examples/01_static_ate.py`` and ``python examples/01_static_ate.py
--preset observational_nonlinear --assume-no-unmeasured-confounding``. Use
``--help`` for all common controls. This implements the static-effect subsection
of the executable CWFM scope.
"""

from __future__ import annotations

import argparse
import shlex

from _common import add_common_arguments, finish, print_dry_run
from cwfm.application import (
    ATEQuery,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    CWFMRunner,
    RepositoryCatalog,
    SupportState,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--preset",
        choices=(
            "randomized_linear",
            "observational_nonlinear",
            "poor_overlap",
            "observational_unknown_confounding",
        ),
        default="randomized_linear",
    )
    parser.add_argument("--assume-no-unmeasured-confounding", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = RepositoryCatalog.from_project_root(args.project_root)
    case = catalog.resolve("static_ate", args.preset, args.seed)
    metadata = case.safe_metadata()
    covariates = tuple(metadata.get("feature_names", [f"X{i + 1}" for i in range(6)]))
    query = ATEQuery("A", "Y", covariates)
    randomized = metadata.get("assignment_design") == "randomized"
    assumptions = AssumptionLedger(
        assignment_design=(
            AssignmentDesign.RANDOMIZED if randomized else AssignmentDesign.OBSERVATIONAL
        ),
        no_unmeasured_confounding=(
            AssertionState.ASSERTED
            if args.assume_no_unmeasured_confounding
            else AssertionState.UNKNOWN
        ),
        consistency=AssertionState.ASSERTED,
        treatment_support=SupportState.UNKNOWN,
    )
    if args.dry_run:
        print_dry_run(
            {"case": case.preview(), "query": query.__dict__, "assumptions": str(assumptions)},
            args.as_json,
            args.quiet,
        )
        return 0
    runner = CWFMRunner.from_project_root(args.project_root, args.device, args.seed)
    result = runner.analyze_ate(case, query, assumptions)
    command = " ".join(shlex.quote(item) for item in [
        "python", "examples/01_static_ate.py", "--preset", args.preset, "--seed", str(args.seed)
    ] + (["--assume-no-unmeasured-confounding"] if args.assume_no_unmeasured_confounding else []))
    return finish(result, args, args.project_root / "outputs" / "01_static_ate", command)


if __name__ == "__main__":
    raise SystemExit(main())

