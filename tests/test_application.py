from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from cwfm.application import (
    ATEQuery,
    AnalysisStatus,
    AssertionState,
    AssignmentDesign,
    AssumptionLedger,
    CWFMRunner,
    ModelBundle,
    ModelBundleStatus,
    ObservedCase,
    RepositoryCase,
    RepositoryCatalog,
    SupportState,
    Task,
    InterferenceQuery,
    load_reference_result,
)
from cwfm.application.adapters import adapt_interference
from cwfm.application.calibration import Calibration
from cwfm.data import (
    ROLE_COVARIATE,
    ROLE_OUTCOME,
    ROLE_TREATMENT,
    collate_observed_inputs,
    generate_ate,
)
from cwfm.experiment import _predict_model
from cwfm.model import CWFM, CWFMConfig


ROOT = Path(__file__).resolve().parents[1]


def available_runner() -> CWFMRunner:
    config = CWFMConfig(
        hidden_dim=16,
        heads=4,
        axial_blocks=1,
        mechanism_experts=2,
        top_k_experts=1,
        world_particles=3,
        dropout=0.0,
    )
    calibration = Calibration(
        normalized_radius={"static_ate": 0.5, "network_interference": 0.5},
        absolute_radius={},
        counts={},
        regime={"evidence_threshold": 0.03},
    )
    placeholder = ROOT / "artifacts" / "cwfm" / "release"
    bundle = ModelBundle(
        ROOT,
        placeholder,
        placeholder / "model_state.pt",
        placeholder / "model_config.json",
        placeholder / "calibration.json",
        placeholder / "manifest.json",
        ModelBundleStatus.AVAILABLE,
        "test model",
        config,
        calibration,
        {},
        "test-checkpoint",
        "test-calibration",
    )
    runner = CWFMRunner(bundle, "cpu", 0)
    torch.manual_seed(123)
    runner._model = CWFM(config).eval()
    return runner


def randomized_case() -> tuple[ObservedCase, ATEQuery, AssumptionLedger]:
    episode = generate_ate(812, "randomized", n=64, p=4)
    names = ("X1", "X2", "X3", "X4", "A", "Y")
    case = ObservedCase(
        episode.values,
        names,
        episode.roles,
        Task.STATIC_ATE,
    )
    query = ATEQuery("A", "Y", names[:4])
    assumptions = AssumptionLedger(
        assignment_design=AssignmentDesign.RANDOMIZED,
        no_unmeasured_confounding=AssertionState.UNKNOWN,
        consistency=AssertionState.ASSERTED,
        treatment_support=SupportState.UNKNOWN,
    )
    return case, query, assumptions


class PublicApplicationTests(unittest.TestCase):
    def test_catalog_has_all_backends_and_model_native_ate_cases(self) -> None:
        catalog = RepositoryCatalog.from_project_root(ROOT)
        self.assertIn("static_ate", catalog.methods())
        self.assertIn("temporal_split_detection", catalog.methods())
        case = catalog.resolve("static_ate", "randomized_linear", 0)
        arrays = case.load_observed()
        self.assertEqual(set(arrays), {"X", "A", "Y"})
        self.assertEqual(arrays["X"].shape[0], len(arrays["A"]))

    def test_missing_checkpoint_is_an_actionable_supported_state(self) -> None:
        bundle = ModelBundle.from_project_root(ROOT)
        self.assertEqual(bundle.status, ModelBundleStatus.UNAVAILABLE)
        self.assertIn("model_state.pt was not found", bundle.reason)
        self.assertIsNotNone(bundle.calibration)

    def test_public_runner_matches_legacy_prediction_on_observed_planes(self) -> None:
        runner = available_runner()
        case, query, assumptions = randomized_case()
        public = runner.analyze(case, query, assumptions)
        episode = generate_ate(812, "randomized", n=64, p=4)
        legacy = _predict_model(runner.model, [episode], torch.device("cpu"))[0]
        self.assertEqual(public.status, AnalysisStatus.ANSWERED)
        self.assertAlmostEqual(public.final_estimate, legacy["final"], places=6)
        self.assertAlmostEqual(public.compiled_estimate, legacy["compiled"], places=6)
        expected_half_width = 0.5 * legacy["raw_scale"]
        self.assertAlmostEqual(public.interval[0], legacy["final"] - expected_half_width, places=6)
        self.assertAlmostEqual(public.interval[1], legacy["final"] + expected_half_width, places=6)

    def test_observed_collator_drops_every_oracle_plane(self) -> None:
        episode = generate_ate(91, "randomized")
        batch = collate_observed_inputs([episode])
        for forbidden in (
            "target",
            "raw_target",
            "identified",
            "supported",
            "structure_target",
            "mechanism",
            "graph",
            "episodes",
            "scenarios",
        ):
            self.assertNotIn(forbidden, batch)

    def test_truth_firewall_never_loads_truth_npz(self) -> None:
        catalog = RepositoryCatalog.from_project_root(ROOT)
        case = catalog.resolve("causal_model_determination", "clean_one_split", 0)
        original_load = np.load

        def guarded_load(path, *args, **kwargs):
            self.assertFalse(str(path).endswith("truth.npz"))
            return original_load(path, *args, **kwargs)

        with mock.patch("numpy.load", side_effect=guarded_load):
            preview = case.preview()
        self.assertIn("X", preview["arrays"])
        self.assertNotIn("true_regime_features", preview["metadata"])
        self.assertNotIn("true_tree", preview["metadata"])

    def test_default_interference_query_selects_all_observed_covariates(self) -> None:
        catalog = RepositoryCatalog.from_project_root(ROOT)
        case = catalog.resolve("causal_interference_detection", "linear_spillover", 0)
        observed = adapt_interference(
            case, InterferenceQuery(outcome="Y1", exposure_mapping="weighted")
        )
        self.assertEqual(observed.values.shape[1], 10)
        self.assertEqual(observed.variable_names[-3:], ("A", "G", "Y1"))

    def test_reference_result_is_classical_and_does_not_use_cwfm(self) -> None:
        result = load_reference_result(
            ROOT, "temporal_split_detection", "one_coefficient_break", 0
        )
        self.assertEqual(result.backend, "classical reference evaluator")
        self.assertFalse(result.cwfm_checkpoint_used)
        self.assertEqual(result.metrics["scenario"], "one_coefficient_break")

    def test_scenario_name_does_not_change_model_result(self) -> None:
        runner = available_runner()
        catalog = RepositoryCatalog.from_project_root(ROOT)
        first = catalog.resolve("static_ate", "randomized_linear", 0)
        renamed = RepositoryCase(
            first.project_root,
            first.method,
            "renamed_without_semantics",
            first.seed,
            first.directory,
            first.task,
            first.backend,
        )
        names = tuple(first.safe_metadata()["feature_names"])
        query = ATEQuery("A", "Y", names)
        assumptions = AssumptionLedger(
            assignment_design=AssignmentDesign.RANDOMIZED,
            consistency=AssertionState.ASSERTED,
        )
        left = runner.analyze(first, query, assumptions)
        right = runner.analyze(renamed, query, assumptions)
        self.assertEqual(left.analysis_id, right.analysis_id)
        self.assertAlmostEqual(left.final_estimate, right.final_estimate, places=7)

    def test_variable_limit_fails_before_inference(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=(60, 13)).astype(np.float32)
        values[:, -2] = np.arange(60) % 2
        roles = np.asarray([ROLE_COVARIATE] * 11 + [ROLE_TREATMENT, ROLE_OUTCOME])
        names = tuple([f"X{i}" for i in range(11)] + ["A", "Y"])
        case = ObservedCase(values, names, roles, Task.STATIC_ATE)
        query = ATEQuery("A", "Y", names[:11])
        result = available_runner().analyze(
            case,
            query,
            AssumptionLedger(
                assignment_design=AssignmentDesign.RANDOMIZED,
                consistency=AssertionState.ASSERTED,
            ),
        )
        self.assertEqual(result.status, AnalysisStatus.INVALID_INPUT)
        self.assertIsNone(result.estimate)
        self.assertIn("at most 12", result.decision_reasons[0])

    def test_identification_abstention_suppresses_principal_estimate(self) -> None:
        case, query, _ = randomized_case()
        result = available_runner().analyze(
            case,
            query,
            AssumptionLedger(
                assignment_design=AssignmentDesign.OBSERVATIONAL,
                consistency=AssertionState.ASSERTED,
                no_unmeasured_confounding=AssertionState.UNKNOWN,
            ),
        )
        self.assertEqual(
            result.status, AnalysisStatus.ABSTAINED_IDENTIFICATION_NOT_ESTABLISHED
        )
        self.assertIsNone(result.estimate)
        self.assertIsNone(result.interval)

    def test_result_json_is_finite_and_reproducible(self) -> None:
        runner = available_runner()
        case, query, assumptions = randomized_case()
        first = runner.analyze(case, query, assumptions)
        second = runner.analyze(case, query, assumptions)
        self.assertEqual(first.analysis_id, second.analysis_id)
        self.assertAlmostEqual(first.final_estimate, second.final_estimate, places=7)
        parsed = json.loads(first.to_json())
        self.assertEqual(parsed["status"], "ANSWERED")
        reconstructed = type(first).from_json(first.to_json())
        self.assertEqual(reconstructed.to_dict(), first.to_dict())

    def test_calibration_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "artifacts" / "cwfm" / "release"
            release.mkdir(parents=True)
            (release / "model_config.json").write_text("{}", encoding="utf-8")
            (release / "calibration.json").write_text(
                json.dumps(
                    {
                        "normalized_radius": {},
                        "absolute_radius": {},
                        "counts": {},
                        "regime": {"evidence_threshold": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            (release / "manifest.json").write_text(
                json.dumps({"calibration_sha256": "not-the-real-hash"}),
                encoding="utf-8",
            )
            bundle = ModelBundle.from_project_root(root)
            self.assertEqual(bundle.status, ModelBundleStatus.INVALID)
            self.assertIn("calibration SHA-256", bundle.reason)


class StreamlitSmokeTests(unittest.TestCase):
    def test_home_page_renders_missing_model_state(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=20)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "CWFM Application")
        self.assertTrue(any("Model unavailable" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
