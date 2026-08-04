from __future__ import annotations

import unittest
from copy import deepcopy

import numpy as np
import torch

from cwfm.baselines import all_baselines, spline_dr_effect
from cwfm.data import (
    EVALUATION_SCENARIOS,
    OOD_DEVELOPMENT_SCENARIOS,
    ROLE_COVARIATE,
    ROLE_PREDICTOR,
    TASK_ATE,
    TASK_INTERFERENCE,
    TASK_REGIME,
    collate_episodes,
    generate_episode,
)
from cwfm.estimators import estimator_experts
from cwfm.external_energy import generate_appliances_energy_episode
from cwfm.model import CWFM, CWFMConfig
from cwfm.train import compute_loss


class CWFMContractTests(unittest.TestCase):
    def test_all_scenarios_are_finite_and_truth_is_not_an_input(self) -> None:
        for index, (task, scenario) in enumerate(EVALUATION_SCENARIOS):
            episode = generate_episode(1000 + index, task, scenario)
            self.assertTrue(np.isfinite(episode.values).all())
            self.assertTrue(np.isfinite(episode.target))
            batch = collate_episodes([episode])
            self.assertNotIn("graph", ["values", "missing", "roles", "adjacency"])
            self.assertEqual(batch["values"].shape[-1], 12)

    def test_training_distribution_covers_every_declared_scenario(self) -> None:
        observed = {
            generate_episode(index).scenario
            for index in range(600)
        }
        expected = {
            f"{prefix}_{scenario}"
            for task, scenario in EVALUATION_SCENARIOS
            for prefix in [
                {
                    0: "ate",
                    1: "regime",
                    2: "interference",
                }[task]
            ]
        }
        self.assertEqual(observed, expected)

    def test_oracle_planes_do_not_change_forward_output(self) -> None:
        episodes = [generate_episode(1500 + index) for index in range(3)]
        batch = collate_episodes(episodes)
        model = CWFM(
            CWFMConfig(hidden_dim=32, heads=4, axial_blocks=1, world_particles=4, dropout=0.0)
        ).eval()
        with torch.no_grad():
            before = model(batch)["effect_mean"]
            batch["target"] = batch["target"] + 10_000
            batch["graph"] = 1 - batch["graph"]
            batch["structure_target"][:] = 0
            after = model(batch)["effect_mean"]
        self.assertTrue(torch.equal(before, after))

    def test_forward_loss_and_backward(self) -> None:
        episodes = [generate_episode(2000 + index) for index in range(5)]
        batch = collate_episodes(episodes)
        model = CWFM(
            CWFMConfig(hidden_dim=32, heads=4, axial_blocks=1, world_particles=4)
        )
        output = model(batch)
        self.assertEqual(output["effect_mean"].shape, (5,))
        self.assertEqual(output["structure_logits"].shape, (5, 13))
        loss, parts = compute_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn("effect", parts)

    def test_row_permutation_invariance_in_eval_mode(self) -> None:
        episode = generate_episode(3011)
        permuted = generate_episode(3011)
        order = np.random.default_rng(9).permutation(len(permuted.values))
        permuted.values = permuted.values[order]
        permuted.adjacency = permuted.adjacency[np.ix_(order, order)]
        permuted.cluster_ids = permuted.cluster_ids[order]
        batch_a = collate_episodes([episode])
        batch_b = collate_episodes([permuted])
        model = CWFM(
            CWFMConfig(hidden_dim=32, heads=4, axial_blocks=1, world_particles=4, dropout=0.0)
        ).eval()
        with torch.no_grad():
            first = model(batch_a)["effect_mean"]
            second = model(batch_b)["effect_mean"]
        self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_effect_path_is_covariate_permutation_invariant(self) -> None:
        saved = torch.load(
            "artifacts/cwfm/checkpoint.pt", map_location="cpu", weights_only=False
        )
        model = CWFM(CWFMConfig(**saved["config"])).eval()
        model.load_state_dict(saved["state_dict"])
        cases = [
            (TASK_ATE, "nonlinear", 123456),
            (TASK_INTERFERENCE, "threshold", 123457),
        ]
        for task, scenario, seed in cases:
            episode = generate_episode(seed, task, scenario)
            permuted = deepcopy(episode)
            eligible = np.flatnonzero(
                np.isin(permuted.roles, [ROLE_COVARIATE, ROLE_PREDICTOR])
            )
            order = np.arange(len(permuted.roles))
            order[eligible] = eligible[::-1]
            inverse = np.argsort(order)
            permuted.values = permuted.values[:, order]
            permuted.roles = permuted.roles[order]
            permuted.graph = permuted.graph[np.ix_(order, order)]
            if permuted.structure_target < len(order):
                permuted.structure_target = int(inverse[permuted.structure_target])

            original_experts = estimator_experts(episode)
            permuted_experts = estimator_experts(permuted)
            self.assertTrue(
                np.allclose(
                    original_experts.estimates,
                    permuted_experts.estimates,
                    atol=1e-6,
                )
            )
            with torch.no_grad():
                original = model(collate_episodes([episode]))
                revised = model(collate_episodes([permuted]))
            self.assertTrue(
                torch.allclose(
                    original["effect_mean"], revised["effect_mean"], atol=1e-6
                )
            )

    def test_baselines_return_declared_outputs(self) -> None:
        for task in range(3):
            episode = generate_episode(4000 + task, task)
            result = all_baselines(episode)
            self.assertTrue(result)
            if task == TASK_REGIME:
                self.assertIn("MOB-like", result)
            else:
                self.assertIn("Linear g-computation", result)

    def test_external_energy_streams_are_disjoint_and_reproducible(self) -> None:
        calibration = generate_appliances_energy_episode(
            87000000, stream="calibration"
        )
        repeated = generate_appliances_energy_episode(
            87000000, stream="calibration"
        )
        test = generate_appliances_energy_episode(88000000, stream="test")
        self.assertTrue(np.array_equal(calibration.values, repeated.values))
        self.assertEqual(calibration.target, repeated.target)
        self.assertTrue(np.all(calibration.cluster_ids % 2 == 0))
        self.assertTrue(np.all(test.cluster_ids % 2 == 1))
        self.assertTrue(np.isfinite(spline_dr_effect(test)))

    def test_revised_model_has_do_no_harm_initialization_and_structural_worlds(self) -> None:
        episodes = [generate_episode(5100 + index) for index in range(6)]
        batch = collate_episodes(episodes)
        model = CWFM(
            CWFMConfig(
                hidden_dim=32,
                heads=4,
                axial_blocks=1,
                world_particles=4,
                dropout=0.0,
            )
        ).eval()
        with torch.no_grad():
            output = model(batch)
        self.assertTrue(
            torch.allclose(
                output["effect_mean"], output["compiled_mean"], atol=1e-6
            )
        )
        self.assertTrue(torch.all(output["residual_gate"] < 0.01))
        self.assertEqual(output["estimator_weights"].shape, (6, 6))
        self.assertEqual(output["world_graph_logits"].shape, (6, 4, 12, 12))
        expected_second_moment = (
            output["world_weights"]
            * (
                output["particle_scales"].square()
                + output["particle_means"].square()
            )
        ).sum(-1)
        expected_scale = (
            expected_second_moment - output["effect_mean"].square()
        ).clamp_min(1e-6).sqrt()
        self.assertTrue(
            torch.allclose(output["effect_scale"], expected_scale, atol=1e-6)
        )

    def test_formal_identification_is_deterministic_and_ood_bank_is_separate(self) -> None:
        episodes = [
            generate_episode(6200 + index, task, scenario)
            for index, (task, scenario) in enumerate(OOD_DEVELOPMENT_SCENARIOS)
        ]
        self.assertFalse(
            set(OOD_DEVELOPMENT_SCENARIOS) & set(EVALUATION_SCENARIOS)
        )
        self.assertTrue(all(ep.metadata["generator_version"] == 2 for ep in episodes))
        batch = collate_episodes(
            [
                generate_episode(6300, 0, "hidden_confounding"),
                generate_episode(6301, 0, "poor_overlap"),
                generate_episode(6302, 0, "linear"),
            ]
        )
        model = CWFM(
            CWFMConfig(
                hidden_dim=32,
                heads=4,
                axial_blocks=1,
                world_particles=3,
            )
        ).eval()
        with torch.no_grad():
            output = model(batch)
        self.assertLess(float(output["identification_logit"][0]), -10)
        self.assertLess(float(output["support_logit"][1]), -10)
        self.assertGreater(float(output["identification_logit"][2]), 10)

    def test_regime_decoder_masks_ineligible_roles(self) -> None:
        episode = generate_episode(6400, TASK_REGIME, "linear_split")
        batch = collate_episodes([episode])
        model = CWFM(
            CWFMConfig(
                hidden_dim=32,
                heads=4,
                axial_blocks=1,
                world_particles=3,
            )
        ).eval()
        with torch.no_grad():
            logits = model(batch)["structure_logits"][0, :-1]
        ineligible = torch.from_numpy(episode.roles != 0)
        self.assertTrue(torch.all(logits[: len(episode.roles)][ineligible] < -100))


if __name__ == "__main__":
    unittest.main()
