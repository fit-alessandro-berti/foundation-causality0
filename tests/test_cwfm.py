from __future__ import annotations

import unittest

import numpy as np
import torch

from cwfm.baselines import all_baselines
from cwfm.data import (
    EVALUATION_SCENARIOS,
    TASK_REGIME,
    collate_episodes,
    generate_episode,
)
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
        batch_a = collate_episodes([episode])
        batch_b = collate_episodes([permuted])
        model = CWFM(
            CWFMConfig(hidden_dim=32, heads=4, axial_blocks=1, world_particles=4, dropout=0.0)
        ).eval()
        with torch.no_grad():
            first = model(batch_a)["effect_mean"]
            second = model(batch_b)["effect_mean"]
        self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_baselines_return_declared_outputs(self) -> None:
        for task in range(3):
            episode = generate_episode(4000 + task, task)
            result = all_baselines(episode)
            self.assertTrue(result)
            if task == TASK_REGIME:
                self.assertIn("MOB-like", result)
            else:
                self.assertIn("Linear g-computation", result)


if __name__ == "__main__":
    unittest.main()
