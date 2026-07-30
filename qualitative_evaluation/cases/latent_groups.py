\
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from ..io_utils import add_project_to_path, load_case, save_table, write_json
from ..plot_utils import save_figure


METHOD = "latent_variable_determination"
SCENARIO = "outcome_irrelevant_latent_block"


def run(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    add_project_to_path(project_root)
    from docs.data.benchmark.common import (
        correlation_matrix,
        match_components,
        standardize_apply,
        standardize_fit,
    )
    from docs.data.benchmark.latent_variable import (
        _impute_apply,
        _impute_fit,
        fit_sparse_pls,
        hard_groups,
        sparse_transform,
        tune_sparse_pls,
    )

    case = load_case(project_root, METHOD, SCENARIO, seed)
    out = output_root / "04_outcome_relevant_latent_groups"
    out.mkdir(parents=True, exist_ok=True)

    xd_raw, yd_raw = case.discovery["X"], case.discovery["Y"]
    xe_raw = case.evaluation["X"]
    impute = _impute_fit(xd_raw)
    xd_imputed = _impute_apply(xd_raw, impute)
    xe_imputed = _impute_apply(xe_raw, impute)
    x_mean, x_scale = standardize_fit(xd_imputed)
    y_mean, y_scale = standardize_fit(yd_raw)
    xd = standardize_apply(xd_imputed, x_mean, x_scale)
    xe = standardize_apply(xe_imputed, x_mean, x_scale)
    yd = standardize_apply(yd_raw, y_mean, y_scale)

    components, keep, search = tune_sparse_pls(
        xd_imputed, yd_raw, seed=seed
    )
    model = fit_sparse_pls(xd, yd, components, keep)
    groups = hard_groups(model["weights"])
    scores = sparse_transform(model, xe)
    relevant = case.truth["relevant_latents"].astype(int)
    signed_correlation = correlation_matrix(
        scores, case.truth["Z_evaluation"][:, relevant]
    )
    matching = match_components(np.abs(signed_correlation))
    estimated_to_truth = {
        int(estimated): int(relevant[truth_position])
        for estimated, truth_position in matching
    }

    aligned_scores = np.full(
        (len(xe), len(case.truth["Sigma_Z"])), np.nan, dtype=float
    )
    score_correlations: dict[int, float] = {}
    for estimated, truth_position in matching:
        truth_latent = int(relevant[truth_position])
        corr = float(signed_correlation[estimated, truth_position])
        aligned_scores[:, truth_latent] = scores[:, estimated] * (
            1.0 if corr >= 0 else -1.0
        )
        score_correlations[truth_latent] = abs(corr)

    rows = []
    for feature in range(len(groups)):
        true_group = int(case.truth["true_group"][feature])
        estimated_component = int(groups[feature])
        aligned_group = (
            estimated_to_truth.get(estimated_component, -2)
            if estimated_component >= 0
            else -1
        )
        selected = estimated_component >= 0
        true_relevant = true_group in set(relevant.tolist())
        if true_relevant and not selected:
            status = "missed relevant indicator"
        elif true_relevant and aligned_group == true_group:
            status = "correct assignment"
        elif true_relevant and selected:
            status = "misassigned relevant indicator"
        elif not true_relevant and not selected:
            status = "correct omission"
        else:
            status = "false inclusion"
        if selected:
            weight = float(model["weights"][feature, estimated_component])
        else:
            weight = 0.0
        rows.append(
            {
                "feature": f"X{feature + 1}",
                "feature_index": feature + 1,
                "true_group": (
                    f"Z{true_group + 1}" if true_group >= 0 else "noise"
                ),
                "truth_is_outcome_relevant": true_relevant,
                "estimated_component": (
                    f"C{estimated_component + 1}" if selected else "omitted"
                ),
                "estimated_component_aligned_to": (
                    f"Z{aligned_group + 1}"
                    if aligned_group >= 0
                    else "omitted/unmatched"
                ),
                "weight": weight,
                "status": status,
            }
        )
    assignment = pd.DataFrame(rows)
    assignment_paths = save_table(
        assignment, out / "feature_assignment_comparison"
    )

    top_rows = []
    for estimated, truth_position in matching:
        truth_latent = int(relevant[truth_position])
        features = np.flatnonzero(groups == estimated)
        order = features[
            np.argsort(-np.abs(model["weights"][features, estimated]))
        ]
        for rank, feature in enumerate(order[:5], start=1):
            top_rows.append(
                {
                    "estimated_component": f"C{estimated + 1}",
                    "aligned_truth": f"Z{truth_latent + 1}",
                    "rank": rank,
                    "feature": f"X{feature + 1}",
                    "estimated_weight": float(
                        model["weights"][feature, estimated]
                    ),
                    "true_group": (
                        f"Z{int(case.truth['true_group'][feature]) + 1}"
                        if int(case.truth["true_group"][feature]) >= 0
                        else "noise"
                    ),
                    "true_loading_on_aligned_latent": float(
                        case.truth["Lambda"][feature, truth_latent]
                    ),
                }
            )
    top_features = pd.DataFrame(top_rows)
    top_paths = save_table(top_features, out / "top_features_per_component")

    representative_features = [0, 8, 16, 24, 32]
    excerpt_rows = np.linspace(
        0, len(case.evaluation["X"]) - 1, 5, dtype=int
    )
    excerpt: dict[str, Any] = {"row": excerpt_rows}
    for feature in representative_features:
        excerpt[f"X{feature + 1}"] = case.evaluation["X"][
            excerpt_rows, feature
        ]
    for outcome in range(case.evaluation["Y"].shape[1]):
        excerpt[f"Y{outcome + 1}"] = case.evaluation["Y"][
            excerpt_rows, outcome
        ]
    for latent in range(case.truth["Z_evaluation"].shape[1]):
        excerpt[f"true_Z{latent + 1}"] = case.truth["Z_evaluation"][
            excerpt_rows, latent
        ]
        excerpt[f"estimated_Z{latent + 1}"] = aligned_scores[
            excerpt_rows, latent
        ]
    excerpt_frame = pd.DataFrame(excerpt)
    excerpt_paths = save_table(excerpt_frame, out / "small_data_example")

    true_codes = np.where(
        case.truth["true_group"] >= 0,
        case.truth["true_group"],
        4,
    )
    estimated_codes = np.asarray(
        [
            estimated_to_truth.get(int(component), -1)
            if component >= 0
            else -1
            for component in groups
        ]
    )
    estimated_display = np.where(estimated_codes >= 0, estimated_codes, 4)
    status_order = [
        "correct assignment",
        "misassigned relevant indicator",
        "missed relevant indicator",
        "correct omission",
        "false inclusion",
    ]
    status_codes = np.asarray(
        [status_order.index(value) for value in assignment["status"]]
    )

    group_cmap = ListedColormap(
        ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B8B8B8"]
    )
    status_cmap = ListedColormap(
        ["#54A24B", "#E45756", "#F2CF5B", "#B8B8B8", "#B279A2"]
    )
    group_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), group_cmap.N)
    status_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), status_cmap.N)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.2]},
    )
    axes[0].imshow(
        true_codes[None, :],
        aspect="auto",
        cmap=group_cmap,
        norm=group_norm,
    )
    axes[0].set_yticks([0], labels=["ground truth"])
    axes[0].set_title(
        f"Outcome-relevant latent grouping (seed {seed}); "
        f"selected K={model['n_components']}"
    )
    axes[1].imshow(
        estimated_display[None, :],
        aspect="auto",
        cmap=group_cmap,
        norm=group_norm,
    )
    axes[1].set_yticks([0], labels=["model output"])
    axes[2].imshow(
        status_codes[None, :],
        aspect="auto",
        cmap=status_cmap,
        norm=status_norm,
    )
    axes[2].set_yticks([0], labels=["comparison"])
    axes[2].set_xticks(
        np.arange(len(groups)),
        labels=[f"X{i + 1}" for i in range(len(groups))],
        rotation=90,
        fontsize=7,
    )
    group_legend = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=group_cmap(i), label=label)
        for i, label in enumerate(["Z1", "Z2", "Z3", "Z4", "omitted/noise"])
    ]
    status_legend = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=status_cmap(i), label=label)
        for i, label in enumerate(status_order)
    ]
    axes[0].legend(
        handles=group_legend,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.75),
        fontsize=8,
    )
    axes[2].legend(
        handles=status_legend,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.65),
        fontsize=8,
    )
    figure_path = save_figure(
        figure, out / "latent_group_assignment_strip.png"
    )

    summary = {
        "seed": seed,
        "scenario": SCENARIO,
        "K_true_outcome_relevant": len(relevant),
        "K_selected": int(model["n_components"]),
        "keep_per_component": int(keep),
        "matching_estimated_component_to_true_latent": estimated_to_truth,
        "score_correlations": score_correlations,
        "status_counts": assignment["status"].value_counts().to_dict(),
        "irrelevant_block_features": list(range(25, 33)),
        "noise_features": list(range(33, 45)),
        "cross_validation_search": search,
    }
    write_json(out / "summary.json", summary)
    return {
        "name": "latent_groups",
        "seed": seed,
        "directory": str(out),
        "figure": figure_path,
        "tables": {
            "assignment": assignment_paths,
            "top_features": top_paths,
            "data_excerpt": excerpt_paths,
        },
        "summary": summary,
    }
