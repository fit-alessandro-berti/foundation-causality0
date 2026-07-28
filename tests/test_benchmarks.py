from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from docs.data.benchmark.causal_model import (
    SCENARIOS as CAUSAL_SCENARIOS,
    evaluate_pair as evaluate_causal,
    generate_pair as generate_causal,
)
from docs.data.benchmark.causal_interference import (
    SCENARIOS as INTERFERENCE_SCENARIOS,
    compute_exposure,
    evaluate_pair as evaluate_interference,
    generate_pair as generate_interference,
)
from docs.data.benchmark.common import match_breakpoints, save_dataset
from docs.data.benchmark.latent_graph import (
    SCENARIOS as GRAPH_SCENARIOS,
    evaluate_pair as evaluate_graph,
    generate_pair as generate_graph,
)
from docs.data.benchmark.latent_variable import (
    SCENARIOS as LATENT_SCENARIOS,
    evaluate_pair as evaluate_latent,
    generate_pair as generate_latent,
)
from docs.data.benchmark.temporal_split import (
    SCENARIOS as TEMPORAL_SCENARIOS,
    evaluate_pair as evaluate_temporal,
    generate_pair as generate_temporal,
)
from docs.data.benchmark.temporal_varx import (
    SCENARIOS as VARX_SCENARIOS,
    evaluate_pair as evaluate_varx,
    generate_pair as generate_varx,
    make_lagged,
)


class GeneratorContractTests(unittest.TestCase):
    def test_every_scenario_generates_finite_shapes(self) -> None:
        for scenario in CAUSAL_SCENARIOS:
            pair = generate_causal(scenario, 0, 140, 160)
            self.assertEqual(pair.discovery["X"].shape[0], 140)
            self.assertEqual(pair.evaluation["Y"].shape[0], 160)
        for scenario in INTERFERENCE_SCENARIOS:
            pair = generate_interference(scenario, 0, 8, 12)
            self.assertEqual(pair.discovery["X"].shape[0], 96)
            self.assertTrue(
                np.allclose(np.diag(pair.discovery["W_observed"]), 0)
            )
            self.assertIn("W_true_discovery", pair.truth)
        for scenario in LATENT_SCENARIOS:
            pair = generate_latent(scenario, 0, 140, 160)
            self.assertEqual(pair.truth["Z_discovery"].shape[0], 140)
            self.assertEqual(pair.truth["true_group"].shape[0], pair.discovery["X"].shape[1])
        for scenario in GRAPH_SCENARIOS:
            pair = generate_graph(scenario, 0, 160, 180)
            eigenvalues = np.linalg.eigvalsh(pair.truth["Theta"])
            self.assertTrue(np.all(eigenvalues > 0))
        for scenario in TEMPORAL_SCENARIOS:
            pair = generate_temporal(scenario, 0, 260)
            self.assertEqual(pair.discovery["timestamps"].shape, (260,))
        for scenario in VARX_SCENARIOS:
            pair = generate_varx(scenario, 0, 320, 80)
            self.assertEqual(pair.discovery["sequence_id"].shape, (320,))

    def test_truth_is_persisted_separately(self) -> None:
        pair = generate_causal("clean_one_split", 1, 100, 120)
        with tempfile.TemporaryDirectory() as directory:
            output = save_dataset(
                Path(directory),
                "method",
                "scenario",
                1,
                pair.discovery,
                pair.evaluation,
                pair.truth,
                pair.metadata,
            )
            discovery = np.load(output / "discovery.npz")
            truth = np.load(output / "truth.npz")
            self.assertNotIn("regime_discovery", discovery.files)
            self.assertIn("regime_discovery", truth.files)

    def test_lagged_rows_never_cross_sequence_boundary(self) -> None:
        x = np.arange(12, dtype=float).reshape(6, 2)
        y = np.arange(6, dtype=float).reshape(6, 1)
        sequence_id = np.asarray([0, 0, 0, 1, 1, 1])
        design, response = make_lagged(x, y, sequence_id)
        self.assertEqual(len(design), 4)
        self.assertEqual(len(response), 4)
        self.assertFalse(np.any((design[:, 0] == 2) & (response[:, 0] == 3)))

    def test_interference_exposure_excludes_own_treatment(self) -> None:
        w = np.ones((4, 4)) - np.eye(4)
        a = np.asarray([1, 0, 0, 0])
        exposure = compute_exposure(a, w, "unweighted")
        self.assertEqual(exposure[0], 0.0)
        self.assertTrue(np.allclose(exposure[1:], 1 / 3))

    def test_breakpoint_matching_is_one_to_one(self) -> None:
        matching = match_breakpoints([100, 110], [105], tolerance=10)
        self.assertEqual(len(matching.pairs), 1)
        self.assertEqual(len(matching.unmatched_true), 1)


class EvaluationSmokeTests(unittest.TestCase):
    def test_all_six_evaluators_return_metrics(self) -> None:
        results = [
            evaluate_causal(generate_causal("clean_one_split", 2, 180, 200), 2, 3, 2),
            evaluate_interference(
                generate_interference("linear_spillover", 2, 9, 12), 2, 3
            ),
            evaluate_latent(generate_latent("clean_balanced_groups", 2, 180, 200), 2, 2),
            evaluate_graph(generate_graph("chain", 2, 180, 200), 2, 2),
            evaluate_temporal(generate_temporal("one_coefficient_break", 2, 280), 2, 3),
            evaluate_varx(generate_varx("one_B_break", 2, 340, 80), 2, 3),
        ]
        for result in results:
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(result["runtime_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
