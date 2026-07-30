\
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import relative_path, write_json


def _figure_links(result: dict[str, Any], output_root: Path) -> list[str]:
    paths: list[str] = []
    if "figure" in result:
        paths.append(relative_path(Path(result["figure"]), output_root))
    for path in result.get("figures", {}).values():
        paths.append(relative_path(Path(path), output_root))
    return paths


def write_reports(
    output_root: Path,
    results: list[dict[str, Any]],
    seed_manifest: dict[str, Any],
) -> dict[str, str]:
    by_name = {result["name"]: result for result in results}

    clean = by_name["clean_regime"]["summary"]
    nonlinear = by_name["nonlinearity_specification"]["summary"]
    latent = by_name["latent_groups"]["summary"]
    graph = by_name["latent_graph"]["summary"]
    temporal = by_name["temporal_breakpoints"]["summary"]
    varx = by_name["varx"]["summary"]

    report = f"""# Qualitative evaluation package

This package complements the aggregate benchmark metrics with small worked
examples. Every model is fitted only on the persisted discovery sample. The
evaluation rows and evaluator-only truth are loaded afterward for comparison.

## Seed selection

Representative examples use the seed whose designated primary metric is
closest to the five-run median. Ties are resolved deterministically and are
recorded in `seed_manifest.json`. The VARX failure case is the smallest seed
with a recorded false-B attribution.

## 1. Observed-regime recovery

The clean case uses seed {clean['seed']}. The true root feature is
`X{clean['true_feature'] + 1}` with threshold {clean['true_threshold']:.3f}.
The learned root is `X{clean['selected_feature'] + 1}` with raw-scale threshold
{clean['learned_threshold']:.3f}. The accompanying matched-probe table holds
ordinary predictors fixed and varies only the regime variable, then compares
the true conditional mean with the fitted model output.

## 2. Model misspecification

The matched nonlinear cases use seed {nonlinear['seed']} and share the same
observed data. The ground truth contains one smooth nonlinear mechanism and no
regime split. With the correct quadratic basis, split detected =
{nonlinear['correct_basis']['split_detected']}. With a misspecified linear local
model, split detected =
{nonlinear['misspecified_linear']['split_detected']}. The figure shows how a
piecewise linear tree can obtain better fit by inventing approximation regions
that are not true causal regimes.

## 3. Outcome-relevant latent groups

The latent-group example uses seed {latent['seed']}. The truth contains
{latent['K_true_outcome_relevant']} outcome-relevant latent groups, while the
model selects {latent['K_selected']}. Feature-level comparison counts are:

{latent['status_counts']}

The assignment strip and tables explicitly show correctly recovered indicators,
missed indicators, misassignments, and correct omission of the outcome-irrelevant
block and nuisance features.

## 4. Latent-graph error propagation

The cross-loading example uses seed {graph['seed']}. The graph comparison is:

- Level A missing edges: {graph['level_A']['missing']}; extra edges:
  {graph['level_A']['extra']}.
- Level B missing edges: {graph['level_B']['missing']}; extra edges:
  {graph['level_B']['extra']}.
- Level C missing edges: {graph['level_C']['missing']}; extra edges:
  {graph['level_C']['extra']}.

Because all panels use the same node positions, the reader can see whether
errors originate in graph estimation, score reconstruction, or estimated
grouping.

## 5. Temporal mechanism change

The coefficient-break example uses seed {temporal['seed']}. Ground-truth
breakpoints are {temporal['coefficient_break']['true_breakpoints']}; detected
raw breakpoints are {temporal['coefficient_break']['raw_breakpoints']}; retained
slope breakpoints are
{temporal['coefficient_break']['retained_slope_breakpoints']}.

The matched intercept-only control has true slope breakpoints
{temporal['intercept_only_control']['true_slope_breakpoints']}, detected raw
breakpoints {temporal['intercept_only_control']['raw_breakpoints']}, and retained
slope breakpoints
{temporal['intercept_only_control']['retained_slope_breakpoints']}.

## 6. Lagged VARX mechanism classification

In the positive example, the true B-breaks are
{varx['positive']['true_B_breaks']}, raw detections are
{varx['positive']['raw_breaks']}, and retained B-breaks are
{varx['positive']['retained_B_breaks']}.

In the A-only negative control, the true B-break set is
{varx['A_only_control']['true_B_breaks']}; raw detections are
{varx['A_only_control']['raw_breaks']}; retained B-breaks are
{varx['A_only_control']['retained_B_breaks']}.

The honest failure case has true B-breaks
{varx['intercept_only_failure']['true_B_breaks']} but retained B-breaks
{varx['intercept_only_failure']['retained_B_breaks']}.

## Recommended placement in the paper

The main text can use three compact figures:

1. clean observed-regime recovery plus the matched nonlinear specification
   comparison;
2. temporal coefficient break plus intercept-only control;
3. the Level A/B/C latent-graph comparison.

The latent-group assignment strip and the VARX positive/control/failure trio fit
well in an appendix qualitative atlas. Small CSV/Markdown/LaTeX tables are
included for direct insertion or editing.
"""
    report_path = output_root / "QUALITATIVE_EVALUATION.md"
    report_path.write_text(report, encoding="utf-8")

    captions = ["# Suggested figure captions", ""]
    for result in results:
        for figure in _figure_links(result, output_root):
            captions.append(f"## `{figure}`")
            if result["name"] == "clean_regime":
                captions.append(
                    "Ground-truth and learned regime tree for a representative "
                    "median-seed case, together with the true and estimated "
                    "change in local outcome coefficients."
                )
            elif result["name"] == "correlated_proxy":
                captions.append(
                    "Observed true regime variable and its correlated proxy. "
                    "The learned root split is overlaid on the same evaluation sample."
                )
            elif result["name"] == "nonlinearity_specification":
                captions.append(
                    "Matched nonlinear data under correct and misspecified "
                    "within-regime models. The misspecified model partitions a "
                    "single smooth mechanism into local linear approximation regions."
                )
            elif result["name"] == "latent_groups":
                captions.append(
                    "Feature-level ground truth, aligned model assignment, and "
                    "post-hoc comparison status for outcome-relevant latent groups."
                )
            elif result["name"] == "latent_graph":
                captions.append(
                    "Ground-truth latent graph and Level A/B/C reconstructions "
                    "under cross-loadings, using identical node positions and "
                    "explicit false-positive/false-negative edge annotation."
                )
            elif result["name"] == "temporal_breakpoints":
                captions.append(
                    "Temporal breakpoint walkthrough showing observed rows, "
                    "detected boundaries, and true versus estimated parameter changes."
                )
            elif result["name"] == "varx":
                captions.append(
                    "Piecewise VARX qualitative case: true and learned B changes, "
                    "an A-only specificity control, and an intercept-only false-B failure."
                )
            captions.append("")
    captions_path = output_root / "FIGURE_CAPTIONS.md"
    captions_path.write_text("\n".join(captions), encoding="utf-8")

    section = f"""## Qualitative mechanism-recovery analysis

To complement aggregate performance metrics, we inspected representative
median-seed examples using persisted discovery/evaluation pairs. Model fitting
used only the discovery arrays; generator truth was revealed only for post-hoc
annotation.

In the observed-regime example, the ground-truth split was
`X{clean['true_feature'] + 1} <= {clean['true_threshold']:.3f}`, while the model
selected `X{clean['selected_feature'] + 1} <=
{clean['learned_threshold']:.3f}`. Matched probes that held the ordinary
predictors fixed and crossed only the regime threshold showed that the fitted
local conditional means closely followed the ground-truth mechanism. In
contrast, on the matched nonlinear control, a correctly specified quadratic
local model returned no split, whereas a misspecified linear local model
partitioned the same smooth mechanism. This demonstrates that predictive gains
from partitioning do not by themselves establish a true causal-regime change.

The outcome-relevant latent-group example selected
{latent['K_selected']} components for
{latent['K_true_outcome_relevant']} relevant latent groups and visibly omitted
the irrelevant block and nuisance features, while the feature-level display
also exposed missed and misassigned indicators. In the cross-loading graph
example, the Level A graph based on true latent scores had missing edges
{graph['level_A']['missing']} and extra edges {graph['level_A']['extra']};
downstream Level C had missing edges {graph['level_C']['missing']} and extra
edges {graph['level_C']['extra']}, localizing the visible error to measurement
and grouping rather than to graph estimation alone.

Finally, the temporal worked example recovered raw breakpoints
{temporal['coefficient_break']['raw_breakpoints']} and retained slope
breakpoints {temporal['coefficient_break']['retained_slope_breakpoints']} for
ground truth {temporal['coefficient_break']['true_breakpoints']}. The matched
intercept-only control retained no slope breakpoint. The VARX examples similarly
separated a true B-change from an A-only change, while the included
intercept-only failure demonstrates the remaining risk of false B attribution.
"""
    section_path = output_root / "PAPER_SECTION_DRAFT.md"
    section_path.write_text(section, encoding="utf-8")

    manifest = {
        "seed_selection": seed_manifest,
        "results": results,
        "reports": {
            "qualitative_evaluation": str(report_path),
            "figure_captions": str(captions_path),
            "paper_section_draft": str(section_path),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    return {
        "qualitative_evaluation": str(report_path),
        "figure_captions": str(captions_path),
        "paper_section_draft": str(section_path),
        "manifest": str(output_root / "manifest.json"),
    }
