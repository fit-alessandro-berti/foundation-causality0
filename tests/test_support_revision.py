"""Behavioral regression checks for the public support and contract policy."""
import unittest
from unittest import mock

import numpy as np

from cwfm.application import (
    AnalysisStatus, AssertionState, AssignmentDesign, AssumptionLedger,
    CWFMRunner, InterferenceQuery, ObservedCase, Task,
)
from cwfm.application.decision_policy import decide
from cwfm.application.validation import evaluate_exposure_support, ValidationReport
from cwfm.data import ROLE_EXPOSURE, ROLE_OUTCOME, ROLE_TREATMENT
from cwfm.support_experiment import population_ate_support, summarize, run


class SupportRevisionTests(unittest.TestCase):
    def test_missing_own_treatment_group_cannot_pass_exposure_screen(self):
        g = np.tile([0.0, 0.25, 0.75, 1.0], 20)
        case = ObservedCase(np.column_stack([np.zeros(80), g, np.zeros(80)]),
            ("A", "G", "Y"), np.array([ROLE_TREATMENT, ROLE_EXPOSURE, ROLE_OUTCOME]),
            Task.NETWORK_INTERFERENCE)
        adequate, details, _ = evaluate_exposure_support(case, 0.25, 0.75)
        self.assertFalse(adequate)
        self.assertEqual(details["by_treatment"]["1"]["count"], 0)

    def test_exposure_screen_cannot_override_prior_support_failure(self):
        # Isolate orchestration: an otherwise valid case already failed the
        # minimum treatment-group size requirement, but exposure screening passes.
        runner = CWFMRunner.__new__(CWFMRunner)
        runner.bundle = mock.Mock()
        runner.bundle.config.max_variables = 12
        runner._adapt = mock.Mock(return_value=(mock.Mock(), {}))
        sentinel = mock.Mock(interference_result=None)
        runner._empty_result = mock.Mock(return_value=sentinel)
        assumptions = AssumptionLedger(assignment_design=AssignmentDesign.RANDOMIZED,
            consistency=AssertionState.ASSERTED, exposure_mapping_predeclared=True,
            interference_restricted_to_observed_graph=True)
        report = ValidationReport(support_adequate=False)
        with mock.patch('cwfm.application.runner.validate_case', return_value=report), \
             mock.patch('cwfm.application.runner.evaluate_exposure_support', return_value=(True, {}, [])):
            runner.analyze(mock.Mock(), InterferenceQuery(), assumptions)
        status = runner._empty_result.call_args.args[3]
        self.assertEqual(status, AnalysisStatus.ABSTAINED_INADEQUATE_SUPPORT)

    def test_declared_identification_does_not_verify_unobserved_truth(self):
        # The same observed support state passes when the user asserts
        # ignorability, and fails when it is unknown. Truth is not an input.
        asserted = AssumptionLedger(assignment_design=AssignmentDesign.OBSERVATIONAL,
            consistency=AssertionState.ASSERTED,
            no_unmeasured_confounding=AssertionState.ASSERTED)
        unknown = AssumptionLedger(assignment_design=AssignmentDesign.OBSERVATIONAL,
            consistency=AssertionState.ASSERTED)
        self.assertTrue(decide(Task.STATIC_ATE, asserted, True)[0].answered)
        self.assertFalse(decide(Task.STATIC_ATE, unknown, True)[0].answered)

    def test_population_reference_and_rate_denominators(self):
        self.assertTrue(population_ate_support('linear_logistic', 0)[0])
        self.assertFalse(population_ate_support('linear_logistic', 8)[0])
        self.assertFalse(population_ate_support('quadratic_logistic', 8)[0])
        protocol = dict(seed=123, ate_families=['linear_logistic'], sample_sizes=[32],
            ate_severities=[0,8], covariates=5, replicates=3,
            network_assignment_probabilities=[])
        result = summarize(run(protocol))
        self.assertEqual(result.episodes.tolist(), [3,3])
        good = result[result.population_adequate].iloc[0]
        bad = result[~result.population_adequate].iloc[0]
        self.assertEqual(good.false_refusal_rate, good.refusal_rate)
        self.assertAlmostEqual(bad.false_acceptance_rate, 1-bad.refusal_rate)
        self.assertTrue(np.isnan(good.false_acceptance_rate))
        self.assertTrue(np.isnan(bad.false_refusal_rate))


if __name__ == '__main__':
    unittest.main()
