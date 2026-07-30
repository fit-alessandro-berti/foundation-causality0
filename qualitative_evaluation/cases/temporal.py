\
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io_utils import add_project_to_path, load_case, save_table, write_json
from ..plot_utils import annotate_matrix, save_figure, symmetric_limit


METHOD = "temporal_split_detection"


def _fit(project_root: Path, scenario: str, seed: int) -> dict[str, Any]:
    add_project_to_path(project_root)
    from docs.data.benchmark.common import (
        segment_labels,
        standardize_apply,
        standardize_fit,
    )
    from docs.data.benchmark.temporal_split import (
        _binary_segmentation,
        _calibrate_full_search,
        _classify_slope_breaks,
        _design,
        _fit_piecewise,
        _predict_piecewise,
        _segment_fit,
    )

    case = load_case(project_root, METHOD, scenario, seed)
    xd_raw, yd_raw = case.discovery["X"], case.discovery["Y"]
    xe_raw, ye_raw = case.evaluation["X"], case.evaluation["Y"]
    x_mean, x_scale = standardize_fit(xd_raw)
    y_mean, y_scale = standardize_fit(yd_raw)
    xd = standardize_apply(xd_raw, x_mean, x_scale)
    xe = standardize_apply(xe_raw, x_mean, x_scale)
    yd = standardize_apply(yd_raw, y_mean, y_scale)
    nonlinear_basis = bool(case.metadata["correct_nonlinear_basis"])
    xd_design = _design(xd, nonlinear_basis)
    xe_design = _design(xe, nonlinear_basis)
    min_segment = 80
    scan_step = 5
    block_size = 12 if case.metadata["x_process"]["phi"] >= 0.8 else 1
    rng = np.random.default_rng(seed + 39019)
    critical, p_value, bootstrap_statistics = _calibrate_full_search(
        xd_design,
        yd,
        min_segment,
        scan_step,
        rng,
        19,
        block_size,
    )
    raw_breaks = (
        _binary_segmentation(
            xd_design,
            yd,
            critical,
            min_segment,
            scan_step,
        )
        if p_value <= 0.05
        else []
    )
    slope_breaks, slope_p_values = _classify_slope_breaks(
        xd_design, yd, raw_breaks
    )
    coefficients, discovery_fitted_s = _fit_piecewise(
        xd_design, yd, raw_breaks
    )
    evaluation_pred_s = _predict_piecewise(
        xe_design, coefficients, raw_breaks
    )
    evaluation_pred = evaluation_pred_s * y_scale + y_mean
    estimated_segment = segment_labels(len(xd_raw), raw_breaks)
    return {
        "case": case,
        "raw_breaks": raw_breaks,
        "slope_breaks": slope_breaks,
        "slope_p_values": slope_p_values,
        "coefficients": coefficients,
        "discovery_fitted_s": discovery_fitted_s,
        "evaluation_prediction": evaluation_pred,
        "estimated_segment": estimated_segment,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "critical": critical,
        "p_value": p_value,
        "bootstrap_statistics": bootstrap_statistics,
        "min_segment": min_segment,
        "scan_step": scan_step,
        "block_size": block_size,
    }


def _raw_slopes(fit: dict[str, Any], n_features: int) -> np.ndarray:
    output = []
    for coefficient in fit["coefficients"]:
        standardized = coefficient[1 : 1 + n_features]
        raw = (
            standardized
            * fit["y_scale"][None, :]
            / fit["x_scale"][:n_features, None]
        )
        output.append(raw)
    return np.stack(output)


def _window_table(
    fit: dict[str, Any],
    *,
    half_width: int = 4,
) -> pd.DataFrame:
    case = fit["case"]
    true_breaks = case.truth["true_breakpoints_all"].astype(int).tolist()
    center = (
        int(true_breaks[0])
        if true_breaks
        else len(case.discovery["X"]) // 2
    )
    start = max(0, center - half_width)
    end = min(len(case.discovery["X"]), center + half_width + 1)
    rows = np.arange(start, end)
    frame = pd.DataFrame(
        {
            "t": rows + 1,
            "X1": case.discovery["X"][rows, 0],
            "X3": case.discovery["X"][rows, 2],
            "X5": case.discovery["X"][rows, 4],
            "X6": case.discovery["X"][rows, 5],
            "Y1": case.discovery["Y"][rows, 0],
            "Y2": case.discovery["Y"][rows, 1],
            "true_segment": case.truth["segment_discovery"][rows],
            "estimated_segment": fit["estimated_segment"][rows],
        }
    )
    return frame


def _coefficient_table(
    true_delta: np.ndarray,
    estimated_delta: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for feature in range(true_delta.shape[0]):
        for outcome in range(true_delta.shape[1]):
            rows.append(
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
    return pd.DataFrame(rows).sort_values(
        "true_delta", key=lambda values: np.abs(values), ascending=False
    )


def run(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    positive = _fit(project_root, "one_coefficient_break", seed)
    control = _fit(project_root, "intercept_only_break", seed)
    out = output_root / "06_temporal_breakpoint_walkthrough"
    out.mkdir(parents=True, exist_ok=True)

    positive_window = _window_table(positive)
    positive_window_paths = save_table(
        positive_window, out / "coefficient_break_data_excerpt"
    )
    control_window = _window_table(control)
    control_window_paths = save_table(
        control_window, out / "intercept_break_data_excerpt"
    )

    true_positive_delta = (
        positive["case"].truth["coefficients"][1]
        - positive["case"].truth["coefficients"][0]
    )
    estimated_positive_slopes = _raw_slopes(
        positive, positive["case"].discovery["X"].shape[1]
    )
    estimated_positive_delta = (
        estimated_positive_slopes[1] - estimated_positive_slopes[0]
    )
    positive_coefficient_frame = _coefficient_table(
        true_positive_delta, estimated_positive_delta
    )
    positive_coefficient_paths = save_table(
        positive_coefficient_frame,
        out / "coefficient_break_parameter_comparison",
    )

    true_control_delta = (
        control["case"].truth["coefficients"][1]
        - control["case"].truth["coefficients"][0]
    )
    estimated_control_slopes = _raw_slopes(
        control, control["case"].discovery["X"].shape[1]
    )
    estimated_control_delta = (
        estimated_control_slopes[1] - estimated_control_slopes[0]
    )
    control_coefficient_frame = _coefficient_table(
        true_control_delta, estimated_control_delta
    )
    control_coefficient_paths = save_table(
        control_coefficient_frame,
        out / "intercept_break_slope_comparison",
    )

    true_break = int(
        positive["case"].truth["true_breakpoints_all"][0]
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    t = positive["case"].discovery["timestamps"]
    axes[0, 0].plot(t, positive["case"].discovery["Y"][:, 0], linewidth=0.9)
    axes[0, 0].axvline(
        true_break, linewidth=2.0, label="ground-truth breakpoint"
    )
    for index, value in enumerate(positive["raw_breaks"]):
        axes[0, 0].axvline(
            value,
            linestyle="--",
            linewidth=2.0,
            label="detected raw breakpoint" if index == 0 else None,
        )
    axes[0, 0].set_title("Full Y1 trajectory")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("Y1")
    axes[0, 0].legend(fontsize=8)

    zoom_start, zoom_end = true_break - 20, true_break + 20
    mask = (t >= zoom_start) & (t <= zoom_end)
    axes[0, 1].plot(
        t[mask],
        positive["case"].discovery["Y"][mask, 0],
        marker="o",
        markersize=3,
        linewidth=1.0,
    )
    axes[0, 1].axvline(true_break, linewidth=2.0)
    for value in positive["raw_breaks"]:
        axes[0, 1].axvline(value, linestyle="--", linewidth=2.0)
    axes[0, 1].set_title("Rows around the breakpoint")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("Y1")

    limit = symmetric_limit(
        true_positive_delta, estimated_positive_delta
    )
    annotate_matrix(
        axes[1, 0],
        true_positive_delta,
        row_labels=[
            f"X{i + 1}" for i in range(true_positive_delta.shape[0])
        ],
        col_labels=[
            f"Y{i + 1}" for i in range(true_positive_delta.shape[1])
        ],
        title="Ground-truth slope change",
        limit=limit,
    )
    annotate_matrix(
        axes[1, 1],
        estimated_positive_delta,
        row_labels=[
            f"X{i + 1}" for i in range(estimated_positive_delta.shape[0])
        ],
        col_labels=[
            f"Y{i + 1}" for i in range(estimated_positive_delta.shape[1])
        ],
        title="Estimated slope change",
        limit=limit,
    )
    figure.suptitle(
        f"Temporal coefficient-break example (seed {seed})\n"
        f"true={true_break}; raw={positive['raw_breaks']}; "
        f"retained slope={positive['slope_breaks']}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    positive_figure_path = save_figure(
        figure, out / "coefficient_break_walkthrough.png"
    )

    control_break = int(
        control["case"].truth["true_breakpoints_all"][0]
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    t_control = control["case"].discovery["timestamps"]
    axes[0].plot(
        t_control,
        control["case"].discovery["Y"][:, 0],
        linewidth=0.9,
    )
    axes[0].axvline(
        control_break,
        linewidth=2.0,
        label="ground truth",
    )
    for index, value in enumerate(control["raw_breaks"]):
        axes[0].axvline(
            value,
            linestyle="--",
            linewidth=2.0,
            label="raw detection" if index == 0 else None,
        )
    axes[0].set_title(
        "Intercept-only change\n"
        f"retained slope breaks: {control['slope_breaks']}"
    )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Y1")
    axes[0].legend(fontsize=8)

    control_limit = symmetric_limit(
        true_control_delta, estimated_control_delta
    )
    annotate_matrix(
        axes[1],
        true_control_delta,
        row_labels=[
            f"X{i + 1}" for i in range(true_control_delta.shape[0])
        ],
        col_labels=[
            f"Y{i + 1}" for i in range(true_control_delta.shape[1])
        ],
        title="True slope change (all zero)",
        limit=control_limit,
    )
    annotate_matrix(
        axes[2],
        estimated_control_delta,
        row_labels=[
            f"X{i + 1}" for i in range(estimated_control_delta.shape[0])
        ],
        col_labels=[
            f"Y{i + 1}" for i in range(estimated_control_delta.shape[1])
        ],
        title="Estimated slope difference",
        limit=control_limit,
    )
    figure.suptitle(
        f"Specificity control: structural break without slope change (seed {seed})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    control_figure_path = save_figure(
        figure, out / "intercept_only_control.png"
    )

    summary = {
        "seed": seed,
        "coefficient_break": {
            "true_breakpoints": positive["case"]
            .truth["true_breakpoints_all"]
            .tolist(),
            "raw_breakpoints": positive["raw_breaks"],
            "retained_slope_breakpoints": positive["slope_breaks"],
            "slope_p_values": positive["slope_p_values"],
            "global_search_p": positive["p_value"],
        },
        "intercept_only_control": {
            "true_breakpoints": control["case"]
            .truth["true_breakpoints_all"]
            .tolist(),
            "true_slope_breakpoints": control["case"]
            .truth["true_breakpoints_slope"]
            .tolist(),
            "raw_breakpoints": control["raw_breaks"],
            "retained_slope_breakpoints": control["slope_breaks"],
            "slope_p_values": control["slope_p_values"],
            "global_search_p": control["p_value"],
        },
    }
    write_json(out / "summary.json", summary)
    return {
        "name": "temporal_breakpoints",
        "seed": seed,
        "directory": str(out),
        "figures": {
            "positive": positive_figure_path,
            "control": control_figure_path,
        },
        "tables": {
            "positive_excerpt": positive_window_paths,
            "control_excerpt": control_window_paths,
            "positive_coefficients": positive_coefficient_paths,
            "control_coefficients": control_coefficient_paths,
        },
        "summary": summary,
    }
