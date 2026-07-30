\
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io_utils import add_project_to_path, load_case, save_table, write_json
from ..plot_utils import annotate_matrix, save_figure, symmetric_limit


METHOD = "temporal_split_detection_method"


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
        _fit_piecewise,
    )
    from docs.data.benchmark.temporal_varx import (
        _classify_b_breaks,
        _segment_relevance,
        make_lagged,
    )

    case = load_case(project_root, METHOD, scenario, seed)
    xd_raw, yd_raw = case.discovery["X"], case.discovery["Y"]
    x_mean, x_scale = standardize_fit(xd_raw)
    y_mean, y_scale = standardize_fit(yd_raw)
    xd = standardize_apply(xd_raw, x_mean, x_scale)
    yd = standardize_apply(yd_raw, y_mean, y_scale)
    design, response = make_lagged(
        xd, yd, case.discovery["sequence_id"]
    )
    n_outcomes = yd.shape[1]
    n_features = xd.shape[1]
    min_segment = max(90, 3 * (1 + n_outcomes + n_features))
    scan_step = 5
    block_size = 12 if scenario == "high_persistence" else 1
    rng = np.random.default_rng(seed + 49031)
    critical, p_value, bootstrap_statistics = _calibrate_full_search(
        design,
        response,
        min_segment,
        scan_step,
        rng,
        19,
        block_size,
    )
    raw_breaks_lagged = (
        _binary_segmentation(
            design,
            response,
            critical,
            min_segment,
            scan_step,
        )
        if p_value <= 0.05
        else []
    )
    raw_breaks = [value + 1 for value in raw_breaks_lagged]
    b_breaks, b_p_values = _classify_b_breaks(
        design,
        response,
        raw_breaks_lagged,
        n_outcomes,
        n_features,
    )
    relevance, relevance_gains = _segment_relevance(
        design, response, raw_breaks_lagged, n_outcomes
    )
    coefficients, fitted = _fit_piecewise(
        design, response, raw_breaks_lagged
    )
    estimated_segment = segment_labels(len(xd_raw), raw_breaks)
    return {
        "case": case,
        "raw_breaks_lagged": raw_breaks_lagged,
        "raw_breaks": raw_breaks,
        "b_breaks": b_breaks,
        "b_p_values": b_p_values,
        "relevance": relevance,
        "relevance_gains": relevance_gains,
        "coefficients": coefficients,
        "fitted": fitted,
        "estimated_segment": estimated_segment,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "critical": critical,
        "p_value": p_value,
        "bootstrap_statistics": bootstrap_statistics,
        "n_outcomes": n_outcomes,
        "n_features": n_features,
    }


def _raw_matrices(fit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    a_matrices = []
    b_matrices = []
    n_outcomes = fit["n_outcomes"]
    for coefficient in fit["coefficients"]:
        a_standardized = coefficient[1 : 1 + n_outcomes].T
        b_standardized = coefficient[1 + n_outcomes :].T
        a_raw = (
            a_standardized
            * fit["y_scale"][:, None]
            / fit["y_scale"][None, :]
        )
        b_raw = (
            b_standardized
            * fit["y_scale"][:, None]
            / fit["x_scale"][None, :]
        )
        a_matrices.append(a_raw)
        b_matrices.append(b_raw)
    return np.stack(a_matrices), np.stack(b_matrices)


def _lagged_window(fit: dict[str, Any], half_width: int = 4) -> pd.DataFrame:
    case = fit["case"]
    true_breaks = case.truth["true_breakpoints_all"].astype(int).tolist()
    center = (
        int(true_breaks[0])
        if true_breaks
        else len(case.discovery["X"]) // 2
    )
    start = max(1, center - half_width)
    end = min(len(case.discovery["X"]), center + half_width + 1)
    rows = np.arange(start, end)
    frame = pd.DataFrame(
        {
            "t": rows + 1,
            "X1_t_minus_1": case.discovery["X"][rows - 1, 0],
            "X3_t_minus_1": case.discovery["X"][rows - 1, 2],
            f"X{case.discovery['X'].shape[1]}_t_minus_1": case.discovery[
                "X"
            ][rows - 1, -1],
            "Y1_t_minus_1": case.discovery["Y"][rows - 1, 0],
            "Y1_t": case.discovery["Y"][rows, 0],
            "true_segment": case.truth["segment_discovery"][rows],
            "estimated_segment": fit["estimated_segment"][rows],
        }
    )
    return frame


def _matrix_comparison(
    true_delta: np.ndarray,
    estimated_delta: np.ndarray,
    *,
    source_prefix: str,
) -> pd.DataFrame:
    rows = []
    for outcome in range(true_delta.shape[0]):
        for source in range(true_delta.shape[1]):
            rows.append(
                {
                    "relationship": (
                        f"{source_prefix}{source + 1}(t-1) -> "
                        f"Y{outcome + 1}(t)"
                    ),
                    "source": f"{source_prefix}{source + 1}",
                    "outcome": f"Y{outcome + 1}",
                    "true_delta": true_delta[outcome, source],
                    "estimated_delta": estimated_delta[outcome, source],
                    "absolute_error": abs(
                        estimated_delta[outcome, source]
                        - true_delta[outcome, source]
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
    positive_seed: int,
    control_seed: int,
    failure_seed: int,
) -> dict[str, Any]:
    positive = _fit(project_root, "one_B_break", positive_seed)
    control = _fit(project_root, "A_only_break", control_seed)
    failure = _fit(project_root, "intercept_only_break", failure_seed)
    out = output_root / "07_varx_mechanism_classification"
    out.mkdir(parents=True, exist_ok=True)

    cases = {
        "one_B_break": positive,
        "A_only_break": control,
        "intercept_only_failure": failure,
    }
    excerpt_paths: dict[str, Any] = {}
    for name, fit in cases.items():
        excerpt_paths[name] = save_table(
            _lagged_window(fit), out / f"{name}_data_excerpt"
        )

    positive_a, positive_b = _raw_matrices(positive)
    positive_true_delta_b = (
        positive["case"].truth["B"][1]
        - positive["case"].truth["B"][0]
    )
    positive_estimated_delta_b = positive_b[1] - positive_b[0]
    positive_delta_frame = _matrix_comparison(
        positive_true_delta_b,
        positive_estimated_delta_b,
        source_prefix="X",
    )
    positive_delta_paths = save_table(
        positive_delta_frame, out / "one_B_break_parameter_comparison"
    )

    control_a, control_b = _raw_matrices(control)
    control_true_delta_a = (
        control["case"].truth["A"][1]
        - control["case"].truth["A"][0]
    )
    control_estimated_delta_a = control_a[1] - control_a[0]
    control_true_delta_b = (
        control["case"].truth["B"][1]
        - control["case"].truth["B"][0]
    )
    control_estimated_delta_b = control_b[1] - control_b[0]
    control_a_paths = save_table(
        _matrix_comparison(
            control_true_delta_a,
            control_estimated_delta_a,
            source_prefix="Y",
        ),
        out / "A_only_break_A_parameter_comparison",
    )
    control_b_paths = save_table(
        _matrix_comparison(
            control_true_delta_b,
            control_estimated_delta_b,
            source_prefix="X",
        ),
        out / "A_only_break_B_parameter_comparison",
    )

    failure_a, failure_b = _raw_matrices(failure)
    failure_true_delta_b = (
        failure["case"].truth["B"][1]
        - failure["case"].truth["B"][0]
    )
    failure_estimated_delta_b = failure_b[1] - failure_b[0]
    failure_delta_paths = save_table(
        _matrix_comparison(
            failure_true_delta_b,
            failure_estimated_delta_b,
            source_prefix="X",
        ),
        out / "intercept_only_false_B_parameter_comparison",
    )

    true_break = int(
        positive["case"].truth["true_breakpoints_B"][0]
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    t = positive["case"].discovery["timestamps"]
    axes[0].plot(
        t,
        positive["case"].discovery["Y"][:, 0],
        linewidth=0.85,
    )
    axes[0].axvline(
        true_break,
        linewidth=2.0,
        label="true B-break",
    )
    for index, value in enumerate(positive["raw_breaks"]):
        axes[0].axvline(
            value,
            linestyle="--",
            linewidth=2.0,
            label="raw detection" if index == 0 else None,
        )
    for index, value in enumerate(positive["b_breaks"]):
        axes[0].scatter(
            [value],
            [np.nanmax(positive["case"].discovery["Y"][:, 0])],
            marker="v",
            s=80,
            label="retained B-break" if index == 0 else None,
        )
    axes[0].set_title("Timeline and breakpoint output")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Y1")
    axes[0].legend(fontsize=8)
    limit = symmetric_limit(
        positive_true_delta_b, positive_estimated_delta_b
    )
    annotate_matrix(
        axes[1],
        positive_true_delta_b,
        row_labels=[
            f"Y{i + 1}"
            for i in range(positive_true_delta_b.shape[0])
        ],
        col_labels=[
            f"X{i + 1}"
            for i in range(positive_true_delta_b.shape[1])
        ],
        title="Ground-truth B change",
        limit=limit,
    )
    annotate_matrix(
        axes[2],
        positive_estimated_delta_b,
        row_labels=[
            f"Y{i + 1}"
            for i in range(positive_estimated_delta_b.shape[0])
        ],
        col_labels=[
            f"X{i + 1}"
            for i in range(positive_estimated_delta_b.shape[1])
        ],
        title="Estimated B change",
        limit=limit,
    )
    figure.suptitle(
        f"VARX feature-to-outcome mechanism change "
        f"(seed {positive_seed})\n"
        f"true={true_break}; raw={positive['raw_breaks']}; "
        f"retained B={positive['b_breaks']}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    positive_figure_path = save_figure(
        figure, out / "one_B_break_walkthrough.png"
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    a_limit = symmetric_limit(
        control_true_delta_a, control_estimated_delta_a
    )
    annotate_matrix(
        axes[0],
        control_true_delta_a,
        row_labels=[
            f"Y{i + 1}" for i in range(control_true_delta_a.shape[0])
        ],
        col_labels=[
            f"Y{i + 1}" for i in range(control_true_delta_a.shape[1])
        ],
        title="True A change",
        limit=a_limit,
    )
    annotate_matrix(
        axes[1],
        control_estimated_delta_a,
        row_labels=[
            f"Y{i + 1}" for i in range(control_estimated_delta_a.shape[0])
        ],
        col_labels=[
            f"Y{i + 1}" for i in range(control_estimated_delta_a.shape[1])
        ],
        title="Estimated A change",
        limit=a_limit,
    )
    b_limit = symmetric_limit(
        control_true_delta_b, control_estimated_delta_b
    )
    annotate_matrix(
        axes[2],
        control_estimated_delta_b,
        row_labels=[
            f"Y{i + 1}" for i in range(control_estimated_delta_b.shape[0])
        ],
        col_labels=[
            f"X{i + 1}" for i in range(control_estimated_delta_b.shape[1])
        ],
        title=(
            "Estimated B difference\n"
            f"retained B-breaks: {control['b_breaks']}"
        ),
        limit=b_limit,
    )
    figure.suptitle(
        f"A-only negative control (seed {control_seed}): "
        "raw change without a retained B-change",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    control_figure_path = save_figure(
        figure, out / "A_only_negative_control.png"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    t_failure = failure["case"].discovery["timestamps"]
    axes[0].plot(
        t_failure,
        failure["case"].discovery["Y"][:, 0],
        linewidth=0.85,
    )
    true_failure_break = int(
        failure["case"].truth["true_breakpoints_all"][0]
    )
    axes[0].axvline(
        true_failure_break,
        linewidth=2.0,
        label="true intercept break",
    )
    for index, value in enumerate(failure["b_breaks"]):
        axes[0].axvline(
            value,
            linestyle="--",
            linewidth=2.0,
            label="false retained B-break" if index == 0 else None,
        )
    axes[0].set_title("Failure timeline")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Y1")
    axes[0].legend(fontsize=8)
    failure_limit = symmetric_limit(
        failure_true_delta_b, failure_estimated_delta_b
    )
    annotate_matrix(
        axes[1],
        failure_estimated_delta_b,
        row_labels=[
            f"Y{i + 1}" for i in range(failure_estimated_delta_b.shape[0])
        ],
        col_labels=[
            f"X{i + 1}" for i in range(failure_estimated_delta_b.shape[1])
        ],
        title="Estimated B difference; true B change is zero",
        limit=failure_limit,
    )
    figure.suptitle(
        f"Honest failure case: intercept-only change falsely attributed "
        f"to B (seed {failure_seed})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    failure_figure_path = save_figure(
        figure, out / "intercept_only_false_B_failure.png"
    )

    summary = {
        "positive": {
            "seed": positive_seed,
            "true_B_breaks": positive["case"]
            .truth["true_breakpoints_B"]
            .tolist(),
            "raw_breaks": positive["raw_breaks"],
            "retained_B_breaks": positive["b_breaks"],
            "B_p_values": positive["b_p_values"],
            "segment_relevance": positive["relevance"],
        },
        "A_only_control": {
            "seed": control_seed,
            "true_all_breaks": control["case"]
            .truth["true_breakpoints_all"]
            .tolist(),
            "true_B_breaks": control["case"]
            .truth["true_breakpoints_B"]
            .tolist(),
            "raw_breaks": control["raw_breaks"],
            "retained_B_breaks": control["b_breaks"],
            "B_p_values": control["b_p_values"],
        },
        "intercept_only_failure": {
            "seed": failure_seed,
            "true_all_breaks": failure["case"]
            .truth["true_breakpoints_all"]
            .tolist(),
            "true_B_breaks": failure["case"]
            .truth["true_breakpoints_B"]
            .tolist(),
            "raw_breaks": failure["raw_breaks"],
            "retained_B_breaks": failure["b_breaks"],
            "B_p_values": failure["b_p_values"],
        },
    }
    write_json(out / "summary.json", summary)
    return {
        "name": "varx",
        "seeds": {
            "positive": positive_seed,
            "control": control_seed,
            "failure": failure_seed,
        },
        "directory": str(out),
        "figures": {
            "positive": positive_figure_path,
            "control": control_figure_path,
            "failure": failure_figure_path,
        },
        "tables": {
            "excerpts": excerpt_paths,
            "positive_delta_B": positive_delta_paths,
            "control_delta_A": control_a_paths,
            "control_delta_B": control_b_paths,
            "failure_delta_B": failure_delta_paths,
        },
        "summary": summary,
    }
