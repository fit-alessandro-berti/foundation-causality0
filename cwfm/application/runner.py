"""Public CWFM runner used by every script and graphical view."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import (
    DESIGN_NONIDENTIFIED,
    DESIGN_OBSERVED,
    DESIGN_POOR_SUPPORT,
    DESIGN_RANDOMIZED,
    TASK_ATE,
    TASK_INTERFERENCE,
    TASK_REGIME,
    Episode,
    collate_observed_inputs,
)
from ..estimators import EXPERT_NAMES
from .adapters import adapt_ate, adapt_interference, adapt_regime
from .catalog import RepositoryCase
from .contracts import (
    ATEQuery,
    AnalysisQuery,
    AssignmentDesign,
    AssumptionLedger,
    InterferenceQuery,
    ObservedCase,
    RegimeQuery,
    Task,
)
from .decision_policy import decide
from .model_bundle import ModelBundle, ModelBundleStatus
from .provenance import build_provenance
from .results import (
    AnalysisResult,
    AnalysisStatus,
    ExpertResult,
    InterferenceResult,
    ModelDiagnostics,
    RegimeResult,
)
from .validation import evaluate_ate_support, evaluate_exposure_support, validate_case


TASK_IDS = {
    Task.STATIC_ATE: TASK_ATE,
    Task.OBSERVED_REGIME: TASK_REGIME,
    Task.NETWORK_INTERFERENCE: TASK_INTERFERENCE,
}
CHANGE_TYPES = {0: "coefficient", 1: "intercept", 2: "variance"}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class CWFMRunner:
    """Stable inference runner with validation, calibration, and abstention."""

    def __init__(
        self,
        bundle: ModelBundle,
        device: str = "auto",
        random_seed: int = 0,
    ) -> None:
        self.bundle = bundle
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.random_seed = random_seed
        self._model = None

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path = ".",
        device: str = "auto",
        random_seed: int = 0,
    ) -> "CWFMRunner":
        return cls(ModelBundle.from_project_root(project_root), device, random_seed)

    @property
    def model_available(self) -> bool:
        return self.bundle.status == ModelBundleStatus.AVAILABLE

    @property
    def model(self):
        if self._model is None:
            self._model = self.bundle.load_model(self.device)
        return self._model

    def _adapt(
        self,
        case: RepositoryCase | ObservedCase,
        query: AnalysisQuery,
    ) -> tuple[ObservedCase, list[dict[str, object]]]:
        if isinstance(query, ATEQuery):
            return adapt_ate(case, query), []
        if isinstance(query, RegimeQuery):
            return adapt_regime(case, query, self.bundle.config.max_variables)
        if isinstance(query, InterferenceQuery):
            return adapt_interference(case, query), []
        raise TypeError(f"Unsupported query type: {type(query).__name__}")

    @staticmethod
    def _question(query: AnalysisQuery) -> tuple[str, str]:
        if isinstance(query, ATEQuery):
            return (
                f"What is the average effect of {query.treatment}={query.treatment_high:g} versus {query.treatment}={query.treatment_low:g} on {query.outcome}?",
                f"Average treatment effect for {query.treatment}: {query.treatment_low:g} → {query.treatment_high:g}",
            )
        if isinstance(query, RegimeQuery):
            return (
                f"Is the relationship with {query.outcome} split by an observed candidate regime variable?",
                "Calibrated observed-regime split determination",
            )
        return (
            f"How does changing mapped peer exposure from {query.exposure_low:g} to {query.exposure_high:g} change {query.outcome}?",
            "Average mapped-exposure spillover contrast, averaging over observed own treatment",
        )

    def _analysis_id(
        self,
        case: ObservedCase,
        query: AnalysisQuery,
        assumptions: AssumptionLedger,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(np.nan_to_num(case.values, nan=9.87654321e12).tobytes())
        digest.update(json.dumps(asdict(query), sort_keys=True).encode())
        digest.update(json.dumps(asdict(assumptions), sort_keys=True, default=str).encode())
        digest.update((self.bundle.checkpoint_hash or "unavailable").encode())
        return digest.hexdigest()[:16]

    def _empty_result(
        self,
        case: ObservedCase,
        query: AnalysisQuery,
        assumptions: AssumptionLedger,
        status: AnalysisStatus,
        reasons: list[str],
        warnings: list[str],
    ) -> AnalysisResult:
        question, estimand = self._question(query)
        interference = None
        if isinstance(query, InterferenceQuery):
            interference = InterferenceResult(
                query.exposure_mapping,
                query.exposure_low,
                query.exposure_high,
                estimand,
            )
        return AnalysisResult(
            analysis_id=self._analysis_id(case, query, assumptions),
            task=query.task,
            status=status,
            question=question,
            estimand=estimand,
            estimate=None,
            interval=None,
            interval_level=None,
            assumptions=assumptions,
            decision_reasons=reasons,
            warnings=warnings,
            interference_result=interference,
            provenance=build_provenance(
                case, query, assumptions, self.bundle, self.random_seed
            ),
        )

    def _episode(
        self,
        case: ObservedCase,
        query: AnalysisQuery,
        status: AnalysisStatus,
        support_adequate: bool,
    ) -> Episode:
        if not status.answered:
            design = DESIGN_NONIDENTIFIED
        elif not support_adequate:
            design = DESIGN_POOR_SUPPORT
        elif query.task == Task.OBSERVED_REGIME:
            design = DESIGN_OBSERVED
        elif isinstance(query, (ATEQuery, InterferenceQuery)):
            design = (
                DESIGN_RANDOMIZED
                if status == AnalysisStatus.ANSWERED
                or (
                    isinstance(query, InterferenceQuery)
                    and status == AnalysisStatus.ANSWERED_CONDITIONAL_ON_ASSUMPTIONS
                )
                else DESIGN_OBSERVED
            )
        else:
            design = DESIGN_OBSERVED
        n, p = case.values.shape
        adjacency = (
            case.adjacency
            if case.adjacency is not None
            else np.zeros((n, n), dtype=np.float32)
        )
        clusters = (
            case.cluster_ids
            if case.cluster_ids is not None
            else np.arange(n, dtype=np.int64)
        )
        query_exposure = (
            (query.exposure_low, query.exposure_high)
            if isinstance(query, InterferenceQuery)
            else (0.25, 0.75)
        )
        return Episode(
            values=case.values,
            roles=case.roles,
            task=TASK_IDS[case.task],
            design=design,
            target=0.0,
            identified=0,
            supported=0,
            structure_target=self.bundle.config.max_variables,
            mechanism=0,
            route_flexible=0.0,
            graph=np.zeros((p, p), dtype=np.float32),
            adjacency=adjacency,
            scenario="public_observed_case",
            seed=self.random_seed,
            query_exposure=query_exposure,
            cluster_ids=clusters,
            metadata={
                "threshold_quantiles": list(query.threshold_quantiles)
                if isinstance(query, RegimeQuery)
                else [0.2, 0.35, 0.5, 0.65, 0.8]
            },
        )

    def analyze(
        self,
        case: RepositoryCase | ObservedCase,
        query: AnalysisQuery,
        assumptions: AssumptionLedger,
    ) -> AnalysisResult:
        try:
            observed, screening = self._adapt(case, query)
        except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
            fallback = case if isinstance(case, ObservedCase) else ObservedCase(
                np.zeros((30, 1), dtype=np.float32),
                ("invalid",),
                np.asarray([0]),
                query.task,
                source=case.source,
            )
            return self._empty_result(
                fallback,
                query,
                assumptions,
                AnalysisStatus.INVALID_INPUT,
                [str(error)],
                [],
            )
        validation = validate_case(observed, self.bundle.config.max_variables)
        support_adequate = validation.support_adequate
        exposure_support: dict[str, Any] = {}
        if isinstance(query, ATEQuery) and validation.valid:
            overlap_adequate, overlap, overlap_reasons = evaluate_ate_support(observed)
            support_adequate = support_adequate and overlap_adequate
            validation.diagnostics["treatment_overlap"] = overlap
            validation.support_adequate = support_adequate
            validation_warnings = validation.warnings + overlap_reasons
        elif isinstance(query, InterferenceQuery) and validation.valid:
            support_adequate, exposure_support, support_reasons = evaluate_exposure_support(
                observed, query.exposure_low, query.exposure_high
            )
            validation.issues.extend([])
            validation.diagnostics["exposure_support"] = exposure_support
            validation.support_adequate = support_adequate
            validation_warnings = validation.warnings + support_reasons
        else:
            validation_warnings = validation.warnings
        if not validation.valid:
            return self._empty_result(
                observed,
                query,
                assumptions,
                AnalysisStatus.INVALID_INPUT,
                validation.errors,
                validation_warnings,
            )
        status, reasons = decide(query.task, assumptions, support_adequate)
        if not status.answered:
            result = self._empty_result(
                observed, query, assumptions, status, reasons, validation_warnings
            )
            result.diagnostics.preprocessing = validation.diagnostics
            if result.interference_result:
                result.interference_result.support = exposure_support
            return result
        if not self.model_available:
            result = self._empty_result(
                observed,
                query,
                assumptions,
                AnalysisStatus.MODEL_UNAVAILABLE,
                [self.bundle.reason],
                validation_warnings,
            )
            result.diagnostics.preprocessing = validation.diagnostics
            if result.interference_result:
                result.interference_result.support = exposure_support
            return result

        episode = self._episode(observed, query, status, support_adequate)
        batch = _to_device(
            collate_observed_inputs([episode], self.bundle.config.max_variables),
            self.device,
        )
        with torch.no_grad():
            output = self.model(batch)
        result = self._result_from_output(
            observed,
            query,
            assumptions,
            status,
            reasons,
            validation_warnings,
            validation.diagnostics,
            exposure_support,
            screening,
            batch,
            output,
        )
        return result

    def _result_from_output(
        self,
        observed: ObservedCase,
        query: AnalysisQuery,
        assumptions: AssumptionLedger,
        status: AnalysisStatus,
        reasons: list[str],
        warnings: list[str],
        preprocessing: dict[str, Any],
        exposure_support: dict[str, Any],
        screening: list[dict[str, object]],
        batch: dict[str, Any],
        output: dict[str, torch.Tensor],
    ) -> AnalysisResult:
        raw_scale = float(batch["raw_scale"][0].cpu())
        predictive_scale = float(
            (output["effect_scale"][0] * batch["raw_scale"][0]).cpu()
        )
        weights = output["estimator_weights"][0].cpu().numpy()
        expert_estimates = (
            batch["expert_estimates"][0] * batch["raw_scale"][0]
        ).cpu().numpy()
        expert_ses = (
            batch["expert_standard_errors"][0] * batch["raw_scale"][0]
        ).cpu().numpy()
        mask = batch["expert_mask"][0].cpu().numpy()
        selected_expert = EXPERT_NAMES[int(weights.argmax())] if mask.any() else None
        compiled = float((output["compiled_mean"][0] * batch["raw_scale"][0]).cpu())
        final = float((output["effect_mean"][0] * batch["raw_scale"][0]).cpu())
        correction = float((output["correction"][0] * batch["raw_scale"][0]).cpu())
        world_weights = output["world_weights"][0].cpu().numpy()
        world_particles = (
            output["particle_means"][0] * batch["raw_scale"][0]
        ).cpu().numpy()
        world_entropy = float(-(world_weights * np.log(np.maximum(world_weights, 1e-9))).sum())
        world_effective = float(1.0 / np.maximum(np.square(world_weights).sum(), 1e-9))
        mechanism = output["mechanism_logits"][0].softmax(-1).cpu().numpy().tolist()
        graph_logits = output["graph_logits"][0, : len(observed.variable_names), : len(observed.variable_names)].cpu().numpy().tolist()
        diagnostics = ModelDiagnostics(
            expert_estimates=[float(value) if available else None for value, available in zip(expert_estimates, mask)],
            expert_standard_errors=[float(value) if available else None for value, available in zip(expert_ses, mask)],
            expert_weights=weights.tolist(),
            compiled_estimate=compiled,
            final_estimate=final,
            residual_correction=correction,
            residual_gate=float(output["residual_gate"][0].cpu()),
            identification_gate="formal assumption token",
            support_probability=float(output["support_logit"][0].sigmoid().cpu()),
            prior_mismatch_probability=float(output["prior_mismatch_logit"][0].sigmoid().cpu()),
            flexible_route_probability=float(output["flexible_route_logit"][0].sigmoid().cpu()),
            mechanism_routing=mechanism,
            mechanism_expert_routing=output["expert_routing"][0].cpu().numpy().tolist(),
            world_weights=world_weights.tolist(),
            world_particle_estimates=world_particles.tolist(),
            world_entropy=world_entropy,
            world_effective_count=world_effective,
            graph_logits=graph_logits,
            nuisance={
                key: (None if not np.isfinite(value) else float(value))
                for key, value in batch["nuisance"][0].items()
            },
            preprocessing=preprocessing,
        )
        expert_results = [
            ExpertResult(
                name=name,
                estimate=float(expert_estimates[index]) if mask[index] else None,
                standard_error=float(expert_ses[index]) if mask[index] else None,
                weight=float(weights[index]),
                available=bool(mask[index]),
            )
            for index, name in enumerate(EXPERT_NAMES)
        ]
        question, estimand = self._question(query)
        regime_result = None
        interference_result = None
        estimate = final
        interval = None
        interval_level = None
        if isinstance(query, RegimeQuery):
            estimate = None
            calibrated_threshold = self.bundle.calibration.regime_evidence_threshold
            evidence = batch["regime_evidence"][0, ..., 0].detach().cpu().numpy()
            candidate_roles = observed.roles == 0
            evidence[~candidate_roles, :] = -np.inf
            flat = int(np.argmax(evidence))
            variable_index, threshold_index = np.unravel_index(flat, evidence.shape)
            evidence_score = float(max(evidence[variable_index, threshold_index], 0.0))
            split = evidence_score >= calibrated_threshold
            selected_variable = observed.variable_names[variable_index] if split else None
            selected_threshold = None
            left_count = right_count = None
            if split:
                selected_threshold = float(
                    np.nanquantile(observed.values[:, variable_index], query.threshold_quantiles[threshold_index])
                )
                side = observed.values[:, variable_index] > selected_threshold
                left_count, right_count = int((~side).sum()), int(side.sum())
            table = []
            for index in np.flatnonzero(candidate_roles):
                for q_index, quantile in enumerate(query.threshold_quantiles):
                    table.append(
                        {
                            "variable": observed.variable_names[index],
                            "quantile": quantile,
                            "threshold": float(np.nanquantile(observed.values[:, index], quantile)),
                            "evidence_score": float(max(evidence[index, q_index], 0.0)),
                        }
                    )
            regime_result = RegimeResult(
                split_detected=split,
                evidence_score=evidence_score,
                calibrated_threshold=calibrated_threshold,
                selected_variable=selected_variable,
                selected_threshold=selected_threshold,
                change_type=CHANGE_TYPES[int(output["change_type_logits"][0].argmax().cpu())] if split else None,
                left_count=left_count,
                right_count=right_count,
                evidence_table=table,
                candidate_screening=screening,
            )
        else:
            assert selected_expert is not None
            interval = self.bundle.calibration.interval(
                final,
                predictive_scale,
                query.task,
                selected_expert,
                expert_estimates.tolist(),
            )
            interval_level = self.bundle.calibration.interval_level
        if isinstance(query, InterferenceQuery):
            interference_result = InterferenceResult(
                query.exposure_mapping,
                query.exposure_low,
                query.exposure_high,
                estimand,
                support=exposure_support,
            )
        if world_effective > len(world_weights) - 0.5:
            warnings.append(
                "Research diagnostic: world particles are nearly uniform and should not be interpreted as separated causal worlds."
            )
        if abs(correction) < 1e-4:
            warnings.append(
                "The final estimate is numerically equal to the compiled expert estimate because the residual correction is negligible."
            )
        return AnalysisResult(
            analysis_id=self._analysis_id(observed, query, assumptions),
            task=query.task,
            status=status,
            question=question,
            estimand=estimand,
            estimate=estimate,
            interval=interval,
            interval_level=interval_level,
            assumptions=assumptions,
            decision_reasons=reasons,
            warnings=warnings,
            expert_results=expert_results,
            selected_expert=selected_expert,
            compiled_estimate=compiled,
            final_estimate=final,
            regime_result=regime_result,
            interference_result=interference_result,
            diagnostics=diagnostics,
            provenance=build_provenance(
                observed, query, assumptions, self.bundle, self.random_seed
            ),
        )

    def analyze_ate(
        self, case: RepositoryCase | ObservedCase, query: ATEQuery, assumptions: AssumptionLedger
    ) -> AnalysisResult:
        return self.analyze(case, query, assumptions)

    def analyze_regime(
        self, case: RepositoryCase | ObservedCase, query: RegimeQuery, assumptions: AssumptionLedger
    ) -> AnalysisResult:
        return self.analyze(case, query, assumptions)

    def analyze_interference(
        self,
        case: RepositoryCase | ObservedCase,
        query: InterferenceQuery,
        assumptions: AssumptionLedger,
    ) -> AnalysisResult:
        return self.analyze(case, query, assumptions)

    def analyze_batch(
        self,
        case: RepositoryCase | ObservedCase,
        queries: list[AnalysisQuery] | tuple[AnalysisQuery, ...],
        assumptions: AssumptionLedger,
    ) -> list[AnalysisResult]:
        """Execute explicitly separate outcome-specific analyses."""

        return [self.analyze(case, query, assumptions) for query in queries]
