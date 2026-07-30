\
from __future__ import annotations

import argparse
from pathlib import Path

from .cases import latent_graph, latent_groups, regime, temporal, varx
from .io_utils import add_project_to_path, write_json
from .report import write_reports
from .selection import select_flagged_seed, select_median_seed


def _select_seeds(project_root: Path) -> dict[str, int]:
    clean_seed = select_median_seed(
        project_root,
        "causal_model_determination",
        "clean_one_split",
        "regime_ari",
    )
    latent_seed = select_median_seed(
        project_root,
        "latent_variable_determination",
        "outcome_irrelevant_latent_block",
        "ari_signal",
    )
    graph_seed = select_median_seed(
        project_root,
        "latent_variable_determination2",
        "cross_loadings",
        "level_C_f1",
        tie_break="max",
    )
    temporal_seed = select_median_seed(
        project_root,
        "temporal_split_detection",
        "one_coefficient_break",
        "relative_coefficient_error",
    )
    varx_seed = select_median_seed(
        project_root,
        "temporal_split_detection_method",
        "one_B_break",
        "relative_B_error",
    )
    failure_seed = select_flagged_seed(
        project_root,
        "temporal_split_detection_method",
        "intercept_only_break",
        "false_B_attribution",
        target=1.0,
    )
    return {
        "clean_regime": clean_seed,
        "correlated_proxy": clean_seed,
        "nonlinearity_specification": clean_seed,
        "latent_groups": latent_seed,
        "latent_graph": graph_seed,
        "temporal": temporal_seed,
        "varx_positive": varx_seed,
        "varx_control": varx_seed,
        "varx_failure": failure_seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready qualitative case studies from the persisted "
            "Foundation Causality benchmark artifacts."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root of foundation-causality-main.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<project-root>/qualitative_evaluation_outputs"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "qualitative_evaluation_outputs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    add_project_to_path(project_root)

    seeds = _select_seeds(project_root)
    write_json(output_root / "seed_manifest.json", seeds)

    results = [
        regime.run_clean_regime(
            project_root, output_root, seed=seeds["clean_regime"]
        ),
        regime.run_correlated_proxy(
            project_root, output_root, seed=seeds["correlated_proxy"]
        ),
        regime.run_nonlinearity_specification(
            project_root,
            output_root,
            seed=seeds["nonlinearity_specification"],
        ),
        latent_groups.run(
            project_root, output_root, seed=seeds["latent_groups"]
        ),
        latent_graph.run(
            project_root, output_root, seed=seeds["latent_graph"]
        ),
        temporal.run(
            project_root, output_root, seed=seeds["temporal"]
        ),
        varx.run(
            project_root,
            output_root,
            positive_seed=seeds["varx_positive"],
            control_seed=seeds["varx_control"],
            failure_seed=seeds["varx_failure"],
        ),
    ]
    reports = write_reports(output_root, results, seeds)
    print(f"Generated qualitative evaluation at: {output_root}")
    for name, path in reports.items():
        print(f"  {name}: {path}")
    return 0
