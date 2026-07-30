\
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..io_utils import add_project_to_path, load_case, save_table, write_json
from ..plot_utils import save_figure


METHOD = "latent_variable_determination2"
SCENARIO = "cross_loadings"
EDGE_THRESHOLD = 0.08


def _edge_map(matrix: np.ndarray, threshold: float) -> dict[tuple[int, int], float]:
    return {
        (i, j): float(matrix[i, j])
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[0])
        if abs(matrix[i, j]) >= threshold
    }


def _draw_graph_panel(
    axis: plt.Axes,
    *,
    title: str,
    matrix: np.ndarray,
    true_matrix: np.ndarray,
    positions: dict[int, np.ndarray],
    ground_truth: bool = False,
) -> None:
    nodes = list(range(matrix.shape[0]))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=850,
        node_color="#F4F4F4",
        edgecolors="#222222",
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: f"Z{node + 1}" for node in nodes},
        ax=axis,
        font_size=9,
    )

    true_edges = _edge_map(true_matrix, 1e-8)
    estimated_edges = (
        true_edges if ground_truth else _edge_map(matrix, EDGE_THRESHOLD)
    )
    correct = sorted(set(true_edges) & set(estimated_edges))
    extra = sorted(set(estimated_edges) - set(true_edges))
    missing = sorted(set(true_edges) - set(estimated_edges))

    if missing:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=missing,
            ax=axis,
            width=2,
            style="dotted",
            edge_color="#E45756",
            alpha=0.8,
        )
    if correct:
        widths = [
            1.5 + 5.0 * abs(estimated_edges[edge]) for edge in correct
        ]
        colors = [
            "#4C78A8" if estimated_edges[edge] >= 0 else "#F58518"
            for edge in correct
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=correct,
            ax=axis,
            width=widths,
            edge_color=colors,
        )
    if extra:
        widths = [
            1.5 + 5.0 * abs(estimated_edges[edge]) for edge in extra
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=extra,
            ax=axis,
            width=widths,
            style="dashed",
            edge_color="#B279A2",
        )

    label_edges = {**{edge: estimated_edges[edge] for edge in correct}, **{edge: estimated_edges[edge] for edge in extra}}
    if label_edges:
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={edge: f"{value:+.2f}" for edge, value in label_edges.items()},
            ax=axis,
            font_size=7,
            rotate=False,
            label_pos=0.55,
        )
    axis.set_title(title)
    axis.axis("off")
    if not ground_truth:
        annotation = []
        if extra:
            annotation.append(
                "extra: " + ", ".join(f"Z{i+1}-Z{j+1}" for i, j in extra)
            )
        if missing:
            annotation.append(
                "missing: " + ", ".join(f"Z{i+1}-Z{j+1}" for i, j in missing)
            )
        if not annotation:
            annotation.append("exact edge set")
        axis.text(
            0.5,
            -0.08,
            "\n".join(annotation),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )


def run(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    add_project_to_path(project_root)
    from docs.data.benchmark.common import standardize_apply, standardize_fit
    from docs.data.benchmark.latent_graph import (
        _align_scores_to_truth,
        _embed_partial,
        _estimated_group_scores,
        _oracle_group_scores,
        fit_network,
    )
    from docs.data.benchmark.latent_variable import (
        fit_sparse_pls,
        tune_sparse_pls,
    )

    case = load_case(project_root, METHOD, SCENARIO, seed)
    out = output_root / "05_latent_graph_error_propagation"
    out.mkdir(parents=True, exist_ok=True)

    xd_raw, yd_raw = case.discovery["X"], case.discovery["Y"]
    xe_raw = case.evaluation["X"]
    x_mean, x_scale = standardize_fit(xd_raw)
    y_mean, y_scale = standardize_fit(yd_raw)
    xd = standardize_apply(xd_raw, x_mean, x_scale)
    xe = standardize_apply(xe_raw, x_mean, x_scale)
    yd = standardize_apply(yd_raw, y_mean, y_scale)
    relevant = case.truth["relevant_latents"].astype(int)
    truth_partial = case.truth["partial_correlation"][
        np.ix_(relevant, relevant)
    ]

    level_a = fit_network(
        case.truth["Z_discovery"][:, relevant],
        case.truth["Z_evaluation"][:, relevant],
    )

    b_train, b_test, b_supports = _oracle_group_scores(
        xd, xe, case.truth["true_group"], relevant
    )
    b_train, b_test, b_matching, b_correlations = _align_scores_to_truth(
        b_train,
        b_test,
        b_supports,
        case.truth["true_group"],
        relevant,
        case.truth["Z_discovery"],
        case.truth["Z_evaluation"],
    )
    level_b = fit_network(b_train, b_test)
    level_b_embedded = _embed_partial(
        level_b["partial"], b_matching, len(relevant)
    )

    components, keep, search = tune_sparse_pls(
        xd_raw, yd_raw, max_components=7, seed=seed
    )
    grouping_model = fit_sparse_pls(xd, yd, components, keep)
    c_train_raw, c_test_raw, c_supports, c_groups = _estimated_group_scores(
        xd, xe, grouping_model["weights"]
    )
    c_train, c_test, c_matching, c_correlations = _align_scores_to_truth(
        c_train_raw,
        c_test_raw,
        c_supports,
        case.truth["true_group"],
        relevant,
        case.truth["Z_discovery"],
        case.truth["Z_evaluation"],
    )
    level_c = fit_network(c_train, c_test)
    level_c_embedded = _embed_partial(
        level_c["partial"], c_matching, len(relevant)
    )

    np.savez_compressed(
        out / "graph_matrices.npz",
        ground_truth=truth_partial,
        level_A=level_a["partial"],
        level_B=level_b_embedded,
        level_C=level_c_embedded,
    )

    matrices = {
        "ground_truth": truth_partial,
        "level_A": level_a["partial"],
        "level_B": level_b_embedded,
        "level_C": level_c_embedded,
    }
    truth_edges = _edge_map(truth_partial, 1e-8)
    edge_rows = []
    for level_name, matrix in matrices.items():
        estimated = (
            truth_edges
            if level_name == "ground_truth"
            else _edge_map(matrix, EDGE_THRESHOLD)
        )
        universe = sorted(set(truth_edges) | set(estimated))
        for edge in universe:
            edge_rows.append(
                {
                    "level": level_name,
                    "edge": f"Z{edge[0] + 1}-Z{edge[1] + 1}",
                    "ground_truth_present": edge in truth_edges,
                    "estimated_present": edge in estimated,
                    "ground_truth_partial_correlation": truth_edges.get(
                        edge, 0.0
                    ),
                    "estimated_partial_correlation": estimated.get(edge, 0.0),
                    "classification": (
                        "true positive"
                        if edge in truth_edges and edge in estimated
                        else (
                            "false positive"
                            if edge in estimated
                            else "false negative"
                        )
                    ),
                }
            )
    edge_frame = pd.DataFrame(edge_rows)
    edge_paths = save_table(edge_frame, out / "edge_comparison")

    matching_map = {
        int(estimated): int(relevant[truth_position])
        for estimated, truth_position in c_matching
    }
    group_rows = []
    for estimated, support in enumerate(c_supports):
        truth_position = next(
            (
                truth_position
                for est, truth_position in c_matching
                if est == estimated
            ),
            None,
        )
        aligned_truth = (
            int(relevant[truth_position])
            if truth_position is not None
            else -1
        )
        counts = {
            f"true_Z{latent + 1}_count": int(
                np.sum(case.truth["true_group"][list(support)] == latent)
            )
            for latent in relevant
        }
        group_rows.append(
            {
                "estimated_node": f"C{estimated + 1}",
                "aligned_truth_node": (
                    f"Z{aligned_truth + 1}" if aligned_truth >= 0 else "unmatched"
                ),
                "features": " ".join(
                    f"X{feature + 1}" for feature in sorted(support)
                ),
                **counts,
            }
        )
    group_frame = pd.DataFrame(group_rows)
    group_paths = save_table(group_frame, out / "estimated_node_composition")

    representative = []
    for latent in relevant:
        representative.append(
            int(np.flatnonzero(case.truth["true_group"] == latent)[0])
        )
    for feature in case.metadata.get("cross_loading_features", [])[:2]:
        if int(feature) not in representative:
            representative.append(int(feature))
    rows = np.linspace(0, len(xe_raw) - 1, 5, dtype=int)
    excerpt: dict[str, Any] = {"row": rows}
    for feature in representative:
        excerpt[f"X{feature + 1}"] = xe_raw[rows, feature]
    for latent in relevant:
        excerpt[f"true_Z{latent + 1}"] = case.truth["Z_evaluation"][
            rows, latent
        ]
    for truth_position, latent in enumerate(relevant):
        if truth_position < c_test.shape[1]:
            excerpt[f"estimated_Z{latent + 1}"] = c_test[rows, truth_position]
    excerpt_frame = pd.DataFrame(excerpt)
    excerpt_paths = save_table(excerpt_frame, out / "small_data_example")

    graph = nx.Graph()
    graph.add_nodes_from(range(len(relevant)))
    positions = nx.circular_layout(graph)
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    _draw_graph_panel(
        axes[0, 0],
        title="Ground truth",
        matrix=truth_partial,
        true_matrix=truth_partial,
        positions=positions,
        ground_truth=True,
    )
    _draw_graph_panel(
        axes[0, 1],
        title="Level A: true latent scores",
        matrix=level_a["partial"],
        true_matrix=truth_partial,
        positions=positions,
    )
    _draw_graph_panel(
        axes[1, 0],
        title="Level B: true groups, reconstructed scores",
        matrix=level_b_embedded,
        true_matrix=truth_partial,
        positions=positions,
    )
    _draw_graph_panel(
        axes[1, 1],
        title="Level C: estimated groups and graph",
        matrix=level_c_embedded,
        true_matrix=truth_partial,
        positions=positions,
    )
    figure.suptitle(
        f"Latent-graph error propagation under cross-loadings (seed {seed})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = save_figure(
        figure, out / "latent_graph_levels.png"
    )

    def differences(matrix: np.ndarray) -> dict[str, list[str]]:
        estimated = _edge_map(matrix, EDGE_THRESHOLD)
        return {
            "missing": [
                f"Z{i + 1}-Z{j + 1}"
                for i, j in sorted(set(truth_edges) - set(estimated))
            ],
            "extra": [
                f"Z{i + 1}-Z{j + 1}"
                for i, j in sorted(set(estimated) - set(truth_edges))
            ],
        }

    summary = {
        "seed": seed,
        "scenario": SCENARIO,
        "edge_threshold": EDGE_THRESHOLD,
        "K_selected": int(grouping_model["n_components"]),
        "keep_per_component": int(keep),
        "level_A": differences(level_a["partial"]),
        "level_B": differences(level_b_embedded),
        "level_C": differences(level_c_embedded),
        "level_B_score_correlations": b_correlations,
        "level_C_score_correlations": c_correlations,
        "estimated_component_to_truth": matching_map,
        "cross_validation_search": search,
    }
    write_json(out / "summary.json", summary)
    return {
        "name": "latent_graph",
        "seed": seed,
        "directory": str(out),
        "figure": figure_path,
        "tables": {
            "edges": edge_paths,
            "groups": group_paths,
            "data_excerpt": excerpt_paths,
        },
        "summary": summary,
    }
