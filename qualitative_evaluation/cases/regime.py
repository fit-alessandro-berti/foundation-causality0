\
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io_utils import add_project_to_path, load_case, save_table, write_json
from ..plot_utils import annotate_matrix, flatten_tree_lines, save_figure, symmetric_limit


METHOD = "causal_model_determination"


def _fit(project_root: Path, scenario: str, seed: int) -> dict[str, Any]:
    add_project_to_path(project_root)
    from docs.data.benchmark.causal_model import (
        _apply_tree,
        _fit_detect,
        _leaf_coefs,
    )
    from docs.data.benchmark.common import standardize_apply, standardize_fit

    case = load_case(project_root, METHOD, scenario, seed)
    xd, yd = case.discovery["X"], case.discovery["Y"]
    xe, ye = case.evaluation["X"], case.evaluation["Y"]
    x_mean, x_scale = standardize_fit(xd)
    y_mean, y_scale = standardize_fit(yd)
    xd_s = standardize_apply(xd, x_mean, x_scale)
    xe_s = standardize_apply(xe, x_mean, x_scale)
    yd_s = standardize_apply(yd, y_mean, y_scale)
    nonlinear_basis = bool(case.metadata["correct_nonlinear_basis"])
    tree, p_value, root = _fit_detect(
        xd_s,
        yd_s,
        nonlinear_basis,
        np.random.default_rng(seed + 71191),
        19,
    )
    pred_s, estimated_regime = _apply_tree(tree, xe_s, nonlinear_basis)
    prediction = pred_s * y_scale + y_mean
    return {
        "case": case,
        "tree": tree,
        "root": root,
        "p_value": p_value,
        "leaf_coefs": _leaf_coefs(tree),
        "estimated_regime": estimated_regime,
        "prediction": prediction,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "xd_s": xd_s,
        "yd_s": yd_s,
        "xe_s": xe_s,
        "nonlinear_basis": nonlinear_basis,
    }


def _raw_leaf_slopes(fit: dict[str, Any], n_predictors: int = 4) -> np.ndarray:
    slopes = []
    for coefficient in fit["leaf_coefs"]:
        standardized = coefficient[1 : 1 + n_predictors]
        raw = (
            standardized
            * fit["y_scale"][None, :]
            / fit["x_scale"][:n_predictors, None]
        )
        slopes.append(raw)
    return np.stack(slopes)


def run_clean_regime(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    fit = _fit(project_root, "clean_one_split", seed)
    case = fit["case"]
    out = output_root / "01_regime_clean"
    out.mkdir(parents=True, exist_ok=True)

    true_threshold = float(case.truth["thresholds"][0])
    root_feature = int(fit["root"]["feature"])
    learned_threshold = float(
        fit["root"]["threshold"] * fit["x_scale"][root_feature]
        + fit["x_mean"][root_feature]
    )

    true_slopes = case.truth["coefficients"]
    estimated_slopes = _raw_leaf_slopes(fit)
    true_delta = true_slopes[1] - true_slopes[0]
    estimated_delta = estimated_slopes[1] - estimated_slopes[0]

    comparison_rows = []
    for feature in range(true_delta.shape[0]):
        for outcome in range(true_delta.shape[1]):
            comparison_rows.append(
                {
                    "relationship": f"X{feature + 1} -> Y{outcome + 1}",
                    "feature": f"X{feature + 1}",
                    "outcome": f"Y{outcome + 1}",
                    "true_delta": true_delta[feature, outcome],
                    "estimated_delta": estimated_delta[feature, outcome],
                    "absolute_error": abs(
                        estimated_delta[feature, outcome]
                        - true_delta[feature, outcome]
                    ),
                }
            )
    delta_frame = pd.DataFrame(comparison_rows).sort_values(
        "true_delta", key=lambda values: np.abs(values), ascending=False
    )
    delta_paths = save_table(delta_frame, out / "coefficient_change_comparison")

    x_eval = case.evaluation["X"]
    y_eval = case.evaluation["Y"]
    regime_feature = int(case.truth["true_regime_features"][0])
    distance = x_eval[:, regime_feature] - true_threshold
    below = np.flatnonzero(distance <= 0)
    above = np.flatnonzero(distance > 0)
    below = below[np.argsort(np.abs(distance[below]))[:5]]
    above = above[np.argsort(np.abs(distance[above]))[:5]]
    selected = np.concatenate([below, above])
    selected = selected[np.argsort(x_eval[selected, regime_feature])]
    boundary = pd.DataFrame(
        {
            "row": selected,
            "X1": x_eval[selected, 0],
            "X2": x_eval[selected, 1],
            "X3": x_eval[selected, 2],
            "X4": x_eval[selected, 3],
            "X5_regime_variable": x_eval[selected, regime_feature],
            "Y1_observed": y_eval[selected, 0],
            "Y1_predicted": fit["prediction"][selected, 0],
            "true_regime": case.truth["regime_evaluation"][selected],
            "predicted_regime": fit["estimated_regime"][selected],
        }
    )
    boundary_paths = save_table(boundary, out / "boundary_rows")

    patterns = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, -1.0, 0.5, 0.0],
        ]
    )
    probes = []
    for pattern_id, pattern in enumerate(patterns, start=1):
        for regime_value in (-0.5, 0.5):
            row = np.zeros(case.discovery["X"].shape[1])
            row[:4] = pattern
            row[regime_feature] = regime_value
            probes.append((pattern_id, row))
    probe_x = np.stack([row for _, row in probes])

    add_project_to_path(project_root)
    from docs.data.benchmark.causal_model import _apply_tree
    from docs.data.benchmark.common import standardize_apply

    probe_s = standardize_apply(probe_x, fit["x_mean"], fit["x_scale"])
    probe_pred_s, probe_regime = _apply_tree(
        fit["tree"], probe_s, fit["nonlinear_basis"]
    )
    probe_pred = probe_pred_s * fit["y_scale"] + fit["y_mean"]
    true_mean = np.empty_like(probe_pred)
    true_regime = (probe_x[:, regime_feature] > true_threshold).astype(int)
    for index, row in enumerate(probe_x):
        regime = int(true_regime[index])
        true_mean[index] = (
            case.truth["intercepts"][regime]
            + row[:4] @ case.truth["coefficients"][regime]
        )

    probe_frame = pd.DataFrame(
        {
            "pattern": [pattern for pattern, _ in probes],
            "X1": probe_x[:, 0],
            "X2": probe_x[:, 1],
            "X3": probe_x[:, 2],
            "X4": probe_x[:, 3],
            "X5": probe_x[:, regime_feature],
            "true_regime": true_regime,
            "predicted_regime": probe_regime,
            "true_mean_Y1": true_mean[:, 0],
            "predicted_mean_Y1": probe_pred[:, 0],
            "true_mean_Y2": true_mean[:, 1],
            "predicted_mean_Y2": probe_pred[:, 1],
            "true_mean_Y3": true_mean[:, 2],
            "predicted_mean_Y3": probe_pred[:, 2],
        }
    )
    probe_paths = save_table(probe_frame, out / "matched_probes")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Ground-truth regime tree")
    axes[0, 0].text(
        0.03,
        0.88,
        f"X{regime_feature + 1} <= {true_threshold:.3f}\n"
        "  L: regime A\n"
        "  R: regime B",
        va="top",
        family="monospace",
        fontsize=12,
    )
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Learned regime tree")
    axes[0, 1].text(
        0.03,
        0.88,
        "\n".join(
            flatten_tree_lines(
                fit["tree"],
                x_mean=fit["x_mean"],
                x_scale=fit["x_scale"],
            )
        ),
        va="top",
        family="monospace",
        fontsize=11,
    )
    limit = symmetric_limit(true_delta, estimated_delta)
    annotate_matrix(
        axes[1, 0],
        true_delta,
        row_labels=[f"X{i + 1}" for i in range(true_delta.shape[0])],
        col_labels=[f"Y{i + 1}" for i in range(true_delta.shape[1])],
        title="Ground-truth coefficient change",
        limit=limit,
    )
    annotate_matrix(
        axes[1, 1],
        estimated_delta,
        row_labels=[f"X{i + 1}" for i in range(estimated_delta.shape[0])],
        col_labels=[f"Y{i + 1}" for i in range(estimated_delta.shape[1])],
        title="Estimated coefficient change",
        limit=limit,
    )
    figure.suptitle(
        f"Observed-regime worked example (seed {seed})\n"
        f"true threshold={true_threshold:.3f}; learned threshold={learned_threshold:.3f}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure_path = save_figure(figure, out / "regime_clean_overview.png")

    learned = {
        "seed": seed,
        "true_feature": regime_feature,
        "selected_feature": root_feature,
        "true_threshold": true_threshold,
        "learned_threshold": learned_threshold,
        "permutation_p": fit["p_value"],
        "tree": fit["tree"],
    }
    write_json(out / "learned_structure.json", learned)
    return {
        "name": "clean_regime",
        "seed": seed,
        "directory": str(out),
        "figure": figure_path,
        "tables": {
            "boundary": boundary_paths,
            "probes": probe_paths,
            "coefficient_change": delta_paths,
        },
        "summary": learned,
    }


def run_correlated_proxy(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    fit = _fit(project_root, "correlated_proxy", seed)
    case = fit["case"]
    out = output_root / "02_correlated_proxy"
    out.mkdir(parents=True, exist_ok=True)
    x = case.evaluation["X"]
    true_regime = case.truth["regime_evaluation"]
    true_feature = int(case.truth["true_regime_features"][0])
    proxy_feature = int(next(iter(case.metadata["proxy_groups"].values()))[0])
    root_feature = int(fit["root"]["feature"])
    learned_threshold = float(
        fit["root"]["threshold"] * fit["x_scale"][root_feature]
        + fit["x_mean"][root_feature]
    )
    frame = pd.DataFrame(
        {
            "row": np.arange(len(x)),
            f"X{true_feature + 1}_true_regime_feature": x[:, true_feature],
            f"X{proxy_feature + 1}_correlated_proxy": x[:, proxy_feature],
            "true_regime": true_regime,
            "predicted_regime": fit["estimated_regime"],
        }
    )
    table_paths = save_table(frame.head(20), out / "data_excerpt")

    figure, axis = plt.subplots(figsize=(7, 6))
    sample = np.linspace(0, len(x) - 1, min(350, len(x)), dtype=int)
    scatter = axis.scatter(
        x[sample, true_feature],
        x[sample, proxy_feature],
        c=true_regime[sample],
        alpha=0.65,
        s=20,
        cmap="viridis",
    )
    axis.axvline(0.0, linestyle="-", linewidth=2, label="true threshold")
    if root_feature == true_feature:
        axis.axvline(
            learned_threshold,
            linestyle="--",
            linewidth=2,
            label="learned threshold",
        )
    axis.set_xlabel(f"X{true_feature + 1}: true regime feature")
    axis.set_ylabel(f"X{proxy_feature + 1}: correlated proxy")
    axis.set_title(
        f"Correlated-proxy case (seed {seed})\n"
        f"selected root feature: X{root_feature + 1}"
    )
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="true regime")
    figure_path = save_figure(figure, out / "correlated_proxy.png")
    summary = {
        "seed": seed,
        "true_feature": true_feature,
        "proxy_feature": proxy_feature,
        "selected_feature": root_feature,
        "learned_threshold": learned_threshold,
        "permutation_p": fit["p_value"],
    }
    write_json(out / "summary.json", summary)
    return {
        "name": "correlated_proxy",
        "seed": seed,
        "directory": str(out),
        "figure": figure_path,
        "tables": {"excerpt": table_paths},
        "summary": summary,
    }


def run_nonlinearity_specification(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    correct = _fit(project_root, "global_nonlinearity_correct", seed)
    misspecified = _fit(
        project_root, "global_nonlinearity_misspecified", seed
    )
    out = output_root / "03_nonlinearity_specification"
    out.mkdir(parents=True, exist_ok=True)

    x_min, x_max = np.quantile(correct["case"].evaluation["X"][:, 0], [0.01, 0.99])
    grid = np.linspace(x_min, x_max, 180)
    probe = np.zeros((len(grid), correct["case"].evaluation["X"].shape[1]))
    probe[:, 0] = grid

    add_project_to_path(project_root)
    from docs.data.benchmark.causal_model import _apply_tree
    from docs.data.benchmark.common import standardize_apply

    rows = {"X1": grid, "true_mean_Y1": grid**2}
    for label, fit in [("correct_basis", correct), ("linear_misspecified", misspecified)]:
        standardized = standardize_apply(probe, fit["x_mean"], fit["x_scale"])
        pred_s, regimes = _apply_tree(
            fit["tree"], standardized, fit["nonlinear_basis"]
        )
        pred = pred_s * fit["y_scale"] + fit["y_mean"]
        rows[f"{label}_predicted_Y1"] = pred[:, 0]
        rows[f"{label}_predicted_regime"] = regimes
    probe_frame = pd.DataFrame(rows)
    probe_paths = save_table(probe_frame.iloc[::20].reset_index(drop=True), out / "probe_grid_excerpt")

    x_eval = correct["case"].evaluation["X"][:, 0]
    y_eval = correct["case"].evaluation["Y"][:, 0]
    sample = np.linspace(0, len(x_eval) - 1, min(260, len(x_eval)), dtype=int)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, label, fit, title in [
        (axes[0], "correct_basis", correct, "Correct quadratic basis"),
        (
            axes[1],
            "linear_misspecified",
            misspecified,
            "Misspecified linear local model",
        ),
    ]:
        axis.scatter(x_eval[sample], y_eval[sample], alpha=0.28, s=15)
        axis.plot(grid, grid**2, linewidth=2.5, label="ground-truth mean")
        axis.plot(
            grid,
            probe_frame[f"{label}_predicted_Y1"],
            linewidth=2.0,
            label="model output",
        )
        if not fit["tree"].get("leaf", False):
            split_features = []

            def collect(node: dict) -> None:
                if node.get("leaf", False):
                    return
                feature = int(node["feature"])
                raw = float(
                    node["threshold"] * fit["x_scale"][feature]
                    + fit["x_mean"][feature]
                )
                if feature == 0:
                    split_features.append(raw)
                collect(node["left"])
                collect(node["right"])

            collect(fit["tree"])
            for value in sorted(set(split_features)):
                axis.axvline(value, linestyle="--", alpha=0.8)
        axis.set_title(
            f"{title}\n"
            + ("no split" if fit["tree"].get("leaf", False) else "learned partition")
        )
        axis.set_xlabel("X1")
        axis.legend()
    axes[0].set_ylabel("Y1")
    figure.suptitle(
        "Same nonlinear data, different within-regime specification",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure_path = save_figure(
        figure, out / "nonlinearity_correct_vs_misspecified.png"
    )
    summary = {
        "seed": seed,
        "ground_truth_regimes": 1,
        "correct_basis": {
            "split_detected": not correct["tree"].get("leaf", False),
            "permutation_p": correct["p_value"],
            "tree": correct["tree"],
        },
        "misspecified_linear": {
            "split_detected": not misspecified["tree"].get("leaf", False),
            "permutation_p": misspecified["p_value"],
            "tree": misspecified["tree"],
        },
    }
    write_json(out / "summary.json", summary)
    return {
        "name": "nonlinearity_specification",
        "seed": seed,
        "directory": str(out),
        "figure": figure_path,
        "tables": {"probe_grid": probe_paths},
        "summary": summary,
    }
