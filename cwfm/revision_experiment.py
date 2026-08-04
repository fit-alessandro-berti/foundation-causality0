from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .baselines import all_baselines
from .data import (
    ROLE_COVARIATE,
    ROLE_OUTCOME,
    ROLE_PREDICTOR,
    ROLE_TREATMENT,
    TASK_ATE,
    TASK_INTERFERENCE,
    Episode,
    generate_episode,
)
from .estimators import EXPERT_NAMES
from .external_energy import generate_appliances_energy_episode
from .experiment import TASK_NAMES, _conformal_quantile, _predict_model
from .model import CWFM, CWFMConfig


EFFECT_SCENARIOS = (
    (TASK_ATE, "linear"),
    (TASK_ATE, "nonlinear"),
    (TASK_ATE, "randomized"),
    (TASK_INTERFERENCE, "linear"),
    (TASK_INTERFERENCE, "threshold"),
    (TASK_INTERFERENCE, "null"),
)

INVALID_SCENARIOS = (
    (TASK_ATE, "poor_overlap"),
    (TASK_ATE, "hidden_confounding"),
    (TASK_INTERFERENCE, "poor_support"),
    (TASK_INTERFERENCE, "hidden_confounding"),
)

PRIMARY_AGGREGATORS = (
    "Pretrained soft router",
    "Pretrained hard top-1",
    "Temperature-scaled router",
    "Uniform expert mean",
    "Median expert",
    "Best fixed expert",
    "Metadata-only selector",
    "Shallow risk selector",
    "Convex stack",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_name(task: int, scenario: str) -> str:
    prefix = "ate" if task == TASK_ATE else "interference"
    return f"{prefix}_{scenario}"


def _softmax(logits: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.where(mask, logits / temperature, -np.inf)
    scaled = scaled - np.max(scaled)
    weights = np.where(mask, np.exp(scaled), 0.0)
    return weights / weights.sum()


def _metadata_features(row: dict[str, Any]) -> np.ndarray:
    episode: Episode = row["episode"]
    return np.asarray(
        [
            1.0,
            float(episode.design == 1),
            len(episode.values) / 128.0,
            episode.values.shape[1] / 12.0,
            float(episode.query_exposure[0]),
            float(episode.query_exposure[1]),
        ],
        dtype=float,
    )


def _risk_features(row: dict[str, Any]) -> np.ndarray:
    estimates = np.asarray(row["expert_estimates"], dtype=float)
    standard_errors = np.asarray(row["expert_standard_errors"], dtype=float)
    mask = np.asarray(row["expert_mask"], dtype=float)
    active = estimates[mask.astype(bool)]
    summaries = np.asarray(
        [
            np.std(active),
            np.ptp(active),
            np.mean(standard_errors[mask.astype(bool)]),
            np.max(standard_errors[mask.astype(bool)]),
        ]
    )
    return np.r_[_metadata_features(row), estimates, standard_errors, mask, summaries]


def _load_model(checkpoint: Path, device: torch.device) -> CWFM:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CWFM(CWFMConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def _episodes(
    scenarios: Iterable[tuple[int, str]], seed_start: int, count: int
) -> Iterable[Episode]:
    for scenario_index, (task, scenario) in enumerate(scenarios):
        for index in range(count):
            yield generate_episode(
                seed_start + scenario_index * 100_000 + index,
                task,
                scenario,
            )


def _collect_from_episodes(
    model: CWFM,
    device: torch.device,
    episodes: Iterable[Episode],
    include_baselines: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: list[Episode] = []

    def flush() -> None:
        if not pending:
            return
        predictions = _predict_model(model, pending, device)
        for episode, prediction in zip(pending, predictions):
            rows.append(
                {
                    "episode": episode,
                    "scenario": episode.scenario,
                    "task": TASK_NAMES[episode.task],
                    "seed": episode.seed,
                    "truth": episode.target,
                    "outcome_scale": prediction["outcome_scale"],
                    "compiled": prediction["compiled"],
                    "router_weights": np.asarray(
                        prediction["estimator_weights"], dtype=float
                    ),
                    "router_logits": np.asarray(
                        prediction["estimator_router_logits"], dtype=float
                    ),
                    "expert_estimates": np.asarray(
                        prediction["expert_estimates"], dtype=float
                    ),
                    "expert_standard_errors": np.asarray(
                        prediction["expert_standard_errors"], dtype=float
                    ),
                    "expert_mask": np.asarray(
                        prediction["expert_mask"], dtype=bool
                    ),
                    "id_probability": prediction["id_probability"],
                    "support_probability": prediction["support_probability"],
                    "baselines": {},
                }
            )
        pending.clear()

    for episode in episodes:
        pending.append(episode)
        if len(pending) == 32:
            flush()
    flush()
    if include_baselines:
        jobs = min(16, os.cpu_count() or 1)
        baseline_results = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(all_baselines)(row["episode"]) for row in rows
        )
        for row, baselines in zip(rows, baseline_results):
            row["baselines"] = baselines
    return rows


def _collect(
    model: CWFM,
    device: torch.device,
    scenarios: tuple[tuple[int, str], ...],
    seed_start: int,
    count: int,
    include_baselines: bool,
) -> list[dict[str, Any]]:
    return _collect_from_episodes(
        model,
        device,
        _episodes(scenarios, seed_start, count),
        include_baselines,
    )


@dataclass
class AggregationPolicy:
    fixed_expert: dict[str, int]
    temperature: dict[str, float]
    stack_weights: dict[str, np.ndarray]
    metadata_models: dict[str, list[Ridge | None]]
    risk_models: dict[str, list[RandomForestRegressor | None]]

    @classmethod
    def fit(
        cls, rows: list[dict[str, Any]], temperatures: list[float]
    ) -> "AggregationPolicy":
        fixed_expert: dict[str, int] = {}
        selected_temperature: dict[str, float] = {}
        stack_weights: dict[str, np.ndarray] = {}
        metadata_models: dict[str, list[Ridge | None]] = {}
        risk_models: dict[str, list[RandomForestRegressor | None]] = {}
        for task in sorted({str(row["task"]) for row in rows}):
            task_rows = [row for row in rows if row["task"] == task]
            masks = np.stack([row["expert_mask"] for row in task_rows])
            common = masks.all(axis=0)
            estimates = np.stack([row["expert_estimates"] for row in task_rows])
            truth = np.asarray([row["truth"] for row in task_rows])
            expert_mae = np.mean(np.abs(estimates - truth[:, None]), axis=0)
            expert_mae[~common] = np.inf
            fixed_expert[task] = int(np.argmin(expert_mae))

            temperature_errors: dict[float, float] = {}
            for temperature in temperatures:
                predictions = [
                    np.dot(
                        _softmax(
                            row["router_logits"], row["expert_mask"], temperature
                        ),
                        row["expert_estimates"],
                    )
                    for row in task_rows
                ]
                temperature_errors[temperature] = float(
                    np.mean(np.abs(np.asarray(predictions) - truth))
                )
            selected_temperature[task] = min(
                temperature_errors, key=temperature_errors.get
            )

            active = np.flatnonzero(common)
            x = estimates[:, active]

            def objective(weights: np.ndarray) -> float:
                return float(np.mean((x @ weights - truth) ** 2))

            initial = np.full(len(active), 1.0 / len(active))
            fit = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(active),
                constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            )
            weights = np.zeros(len(EXPERT_NAMES))
            weights[active] = fit.x if fit.success else initial
            stack_weights[task] = weights

            meta_x = np.stack([_metadata_features(row) for row in task_rows])
            risk_x = np.stack([_risk_features(row) for row in task_rows])
            meta_task_models: list[Ridge | None] = []
            risk_task_models: list[RandomForestRegressor | None] = []
            for expert in range(len(EXPERT_NAMES)):
                available = masks[:, expert]
                if not available.any():
                    meta_task_models.append(None)
                    risk_task_models.append(None)
                    continue
                target = np.abs(estimates[available, expert] - truth[available])
                meta_task_models.append(
                    Ridge(alpha=1.0).fit(meta_x[available], target)
                )
                risk_task_models.append(
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_leaf=20,
                        max_features=0.8,
                        n_jobs=-1,
                        random_state=1729 + expert,
                    ).fit(risk_x[available], target)
                )
            metadata_models[task] = meta_task_models
            risk_models[task] = risk_task_models
        return cls(
            fixed_expert,
            selected_temperature,
            stack_weights,
            metadata_models,
            risk_models,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "fixed_expert": {
                task: EXPERT_NAMES[index]
                for task, index in self.fixed_expert.items()
            },
            "temperature": self.temperature,
            "convex_stack_weights": {
                task: {
                    name: float(weights[index])
                    for index, name in enumerate(EXPERT_NAMES)
                }
                for task, weights in self.stack_weights.items()
            },
            "metadata_risk_model": "ridge(alpha=1.0), one model per task/expert",
            "shallow_risk_model": (
                "random forest regression, 200 trees, max_depth=4, "
                "min_samples_leaf=20, one model per task/expert"
            ),
        }

    @staticmethod
    def _select(
        models: list[Any], features: np.ndarray, mask: np.ndarray
    ) -> int:
        risks = np.full(len(EXPERT_NAMES), np.inf)
        for index, model in enumerate(models):
            if mask[index] and model is not None:
                risks[index] = float(model.predict(features[None])[0])
        return int(np.argmin(risks))

    def predict(
        self, row: dict[str, Any], include_individual: bool = True
    ) -> dict[str, float]:
        task = str(row["task"])
        estimates = np.asarray(row["expert_estimates"], dtype=float)
        mask = np.asarray(row["expert_mask"], dtype=bool)
        weights = np.asarray(row["router_weights"], dtype=float)
        hard = int(np.argmax(np.where(mask, row["router_logits"], -np.inf)))
        fixed = self.fixed_expert[task]
        metadata = self._select(
            self.metadata_models[task], _metadata_features(row), mask
        )
        risk = self._select(self.risk_models[task], _risk_features(row), mask)
        predictions = {
            "Pretrained soft router": float(np.dot(weights, estimates)),
            "Pretrained hard top-1": float(estimates[hard]),
            "Temperature-scaled router": float(
                np.dot(
                    _softmax(
                        row["router_logits"], mask, self.temperature[task]
                    ),
                    estimates,
                )
            ),
            "Uniform expert mean": float(np.mean(estimates[mask])),
            "Median expert": float(np.median(estimates[mask])),
            "Best fixed expert": float(estimates[fixed]),
            "Metadata-only selector": float(estimates[metadata]),
            "Shallow risk selector": float(estimates[risk]),
            "Convex stack": float(np.dot(self.stack_weights[task], estimates)),
        }
        if include_individual:
            for index, name in enumerate(EXPERT_NAMES):
                if mask[index]:
                    predictions[f"Expert: {name}"] = float(estimates[index])
        predictions.update(
            {str(name): float(value) for name, value in row["baselines"].items()}
        )
        return predictions


def _calibrate(
    rows: list[dict[str, Any]], policy: AggregationPolicy, coverage: float
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    scores: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        predictions = policy.predict(row)
        for method, estimate in predictions.items():
            error = abs(estimate - row["truth"])
            grouped.setdefault(row["scenario"], {}).setdefault(method, []).append(
                error
            )
            scores.append(
                {
                    "scenario": row["scenario"],
                    "task": row["task"],
                    "seed": row["seed"],
                    "method": method,
                    "absolute_residual": error,
                }
            )
    radii = {
        scenario: {
            method: _conformal_quantile(values, coverage)
            for method, values in methods.items()
        }
        for scenario, methods in grouped.items()
    }
    return radii, pd.DataFrame(scores)


def _add_oracles(
    rows: list[dict[str, Any]], predictions: list[dict[str, float]]
) -> None:
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["scenario"] == scenario]
        expert_names = sorted(
            set.intersection(
                *[
                    {
                        name
                        for name in predictions[index]
                        if name.startswith("Expert: ")
                    }
                    for index in indices
                ]
            )
        )
        cell_mae = {
            name: np.mean(
                [
                    abs(predictions[index][name] - rows[index]["truth"])
                    for index in indices
                ]
            )
            for name in expert_names
        }
        best = min(cell_mae, key=cell_mae.get)
        for index in indices:
            expert_predictions = {
                name: estimate
                for name, estimate in predictions[index].items()
                if name.startswith("Expert: ")
            }
            predictions[index]["Oracle: per-cell expert"] = predictions[index][best]
            predictions[index]["Oracle: per-episode expert"] = min(
                expert_predictions.values(),
                key=lambda estimate: abs(estimate - rows[index]["truth"]),
            )


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return float(center - half), float(center + half)


def _evaluate(
    rows: list[dict[str, Any]],
    policy: AggregationPolicy,
    radii: dict[str, dict[str, float]],
) -> pd.DataFrame:
    predictions = [policy.predict(row) for row in rows]
    _add_oracles(rows, predictions)
    records: list[dict[str, Any]] = []
    for row, methods in zip(rows, predictions):
        oracle_error = abs(
            methods["Oracle: per-episode expert"] - row["truth"]
        )
        for method, estimate in methods.items():
            radius = radii.get(row["scenario"], {}).get(method, np.nan)
            is_oracle = method.startswith("Oracle:")
            error = abs(estimate - row["truth"])
            records.append(
                {
                    "scenario": row["scenario"],
                    "task": row["task"],
                    "seed": row["seed"],
                    "method": method,
                    "truth": row["truth"],
                    "estimate": estimate,
                    "absolute_error": error,
                    "standardized_absolute_error": error
                    / max(row["outcome_scale"], 1e-8),
                    "regret_to_episode_oracle": error - oracle_error,
                    "lower": estimate - radius if not is_oracle else np.nan,
                    "upper": estimate + radius if not is_oracle else np.nan,
                    "covered": float(abs(estimate - row["truth"]) <= radius)
                    if not is_oracle
                    else np.nan,
                    "interval_width": 2 * radius if not is_oracle else np.nan,
                    "standardized_interval_width": 2
                    * radius
                    / max(row["outcome_scale"], 1e-8)
                    if not is_oracle
                    else np.nan,
                    "answered": True,
                    "nondeployable_oracle": is_oracle,
                }
            )
    frame = pd.DataFrame(records)
    shared = frame[frame["method"].isin(PRIMARY_AGGREGATORS)].copy()
    shared["method_rank"] = shared.groupby(["scenario", "seed"])[
        "absolute_error"
    ].rank(method="average")
    frame = frame.merge(
        shared[["scenario", "seed", "method", "method_rank"]],
        on=["scenario", "seed", "method"],
        how="left",
    )
    return frame


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cell = (
        frame.groupby(["task", "scenario", "method"], dropna=False)
        .agg(
            mae=("absolute_error", "mean"),
            mae_sd=("absolute_error", "std"),
            standardized_mae=("standardized_absolute_error", "mean"),
            regret=("regret_to_episode_oracle", "mean"),
            coverage=("covered", "mean"),
            mean_width=("interval_width", "mean"),
            standardized_width=("standardized_interval_width", "mean"),
            mean_rank=("method_rank", "mean"),
            episodes=("seed", "count"),
        )
        .reset_index()
    )
    coverage_bounds = []
    for (task, scenario, method), group in frame.groupby(
        ["task", "scenario", "method"]
    ):
        observed = group["covered"].dropna()
        low, high = _wilson(int(observed.sum()), len(observed))
        coverage_bounds.append(
            {
                "task": task,
                "scenario": scenario,
                "method": method,
                "coverage_ci_low": low,
                "coverage_ci_high": high,
            }
        )
    by_cell = by_cell.merge(
        pd.DataFrame(coverage_bounds), on=["task", "scenario", "method"]
    )

    overall_rows = []
    for method, group in frame.groupby("method"):
        cell_mae = group.groupby("scenario")["absolute_error"].mean()
        observed = group["covered"].dropna()
        low, high = _wilson(int(observed.sum()), len(observed))
        overall_rows.append(
            {
                "method": method,
                "micro_mae": group["absolute_error"].mean(),
                "macro_mae": cell_mae.mean(),
                "standardized_mae": group["standardized_absolute_error"].mean(),
                "mean_rank": group["method_rank"].mean(),
                "mean_regret": group["regret_to_episode_oracle"].mean(),
                "worst_cell_mae": cell_mae.max(),
                "coverage": observed.mean() if len(observed) else np.nan,
                "coverage_ci_low": low,
                "coverage_ci_high": high,
                "mean_width": group["interval_width"].mean(),
                "standardized_width": group[
                    "standardized_interval_width"
                ].mean(),
                "episodes": len(group),
            }
        )
    return by_cell, pd.DataFrame(overall_rows)


def _paired_bootstrap(
    frame: pd.DataFrame, replicates: int
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(921)
    pivot = frame[frame["method"].isin(PRIMARY_AGGREGATORS)].pivot_table(
        index=["scenario", "seed"], columns="method", values="absolute_error"
    )
    comparisons: dict[str, dict[str, Any]] = {}
    first = "Pretrained soft router"
    for second in PRIMARY_AGGREGATORS:
        if second == first:
            continue
        paired = pivot[[first, second]].dropna()
        cell_differences = {
            scenario: group[first].to_numpy() - group[second].to_numpy()
            for scenario, group in paired.groupby(level="scenario")
        }
        observed = float(np.mean([values.mean() for values in cell_differences.values()]))
        draws = np.empty(replicates)
        for draw in range(replicates):
            draws[draw] = np.mean(
                [
                    rng.choice(values, size=len(values), replace=True).mean()
                    for values in cell_differences.values()
                ]
            )
        comparisons[second] = {
            "soft_minus_comparator_macro_mae": observed,
            "ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return comparisons


def _contract_property_test(
    model: CWFM,
    device: torch.device,
    policy: AggregationPolicy,
    seed_start: int,
    count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = _collect(
        model,
        device,
        INVALID_SCENARIOS,
        seed_start,
        count,
        include_baselines=False,
    )
    methods = list(PRIMARY_AGGREGATORS) + [
        "Linear g-computation",
        "Random forest g-computation",
        "AIPW / network orthogonal",
        "DR learner (spline nuisances)",
    ]
    records = []
    gate_matches_contract = []
    for row in rows:
        episode: Episode = row["episode"]
        gate = row["id_probability"] >= 0.5 and row["support_probability"] >= 0.5
        declared = bool(episode.identified and episode.supported)
        gate_matches_contract.append(gate == declared)
        reason = "identification" if not episode.identified else "support"
        for method in methods:
            records.append(
                {
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "method": method,
                    "declared_failure": reason,
                    "always_answer_form": True,
                    "gate_wrapped_answer": gate,
                }
            )
    frame = pd.DataFrame(records)
    summary = {
        "episodes": len(rows),
        "methods": len(methods),
        "gate_matches_declared_contract": float(np.mean(gate_matches_contract)),
        "always_answer_rate": float(frame["always_answer_form"].mean()),
        "gate_wrapped_answer_rate": float(frame["gate_wrapped_answer"].mean()),
        "interpretation": (
            "Deterministic enforcement of supplied contract fields; not detection "
            "of hidden confounding or empirical overlap from observations."
        ),
    }
    return frame, summary


def _permuted_columns(episode: Episode, rng: np.random.Generator) -> Episode:
    revised = deepcopy(episode)
    order = np.arange(len(revised.roles))
    for role in np.unique(revised.roles):
        members = np.flatnonzero(revised.roles == role)
        order[members] = rng.permutation(members)
    inverse = np.argsort(order)
    revised.values = revised.values[:, order]
    revised.roles = revised.roles[order]
    revised.graph = revised.graph[np.ix_(order, order)]
    if revised.structure_target < len(order):
        revised.structure_target = int(inverse[revised.structure_target])
    return revised


def _add_covariate(episode: Episode, values: np.ndarray) -> Episode:
    revised = deepcopy(episode)
    outcome = int(np.flatnonzero(revised.roles == ROLE_OUTCOME)[0])
    revised.values = np.insert(revised.values, outcome, values, axis=1)
    revised.roles = np.insert(revised.roles, outcome, ROLE_COVARIATE)
    graph = np.zeros((len(revised.roles), len(revised.roles)), dtype=np.float32)
    old = np.arange(len(revised.roles)) != outcome
    graph[np.ix_(old, old)] = revised.graph
    revised.graph = graph
    if revised.structure_target >= outcome:
        revised.structure_target += 1
    return revised


def _robustness_bank(
    model: CWFM,
    device: torch.device,
    seed_start: int,
    count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs: list[tuple[str, Episode, Episode, float, float]] = []
    for scenario_index, (task, scenario) in enumerate(EFFECT_SCENARIOS):
        for index in range(count):
            seed = seed_start + scenario_index * 100_000 + index
            episode = generate_episode(seed, task, scenario)
            rng = np.random.default_rng(seed ^ 0xBADC0DE)

            row_permuted = deepcopy(episode)
            order = rng.permutation(len(episode.values))
            row_permuted.values = row_permuted.values[order]
            row_permuted.adjacency = row_permuted.adjacency[np.ix_(order, order)]
            row_permuted.cluster_ids = row_permuted.cluster_ids[order]
            pairs.append(("row_permutation", episode, row_permuted, 1.0, 0.0))
            pairs.append(
                (
                    "covariate_permutation",
                    episode,
                    _permuted_columns(episode, rng),
                    1.0,
                    0.0,
                )
            )

            outcome = int(np.flatnonzero(episode.roles == ROLE_OUTCOME)[0])
            translated = deepcopy(episode)
            translated.values[:, outcome] += 3.25
            pairs.append(("outcome_translation", episode, translated, 1.0, 0.0))
            for label, factor in (
                ("positive_outcome_scaling", 2.5),
                ("negative_outcome_scaling", -1.5),
            ):
                scaled = deepcopy(episode)
                scaled.values[:, outcome] *= factor
                scaled.target *= factor
                pairs.append((label, episode, scaled, factor, 0.0))

            if task == TASK_ATE:
                treatment = int(
                    np.flatnonzero(episode.roles == ROLE_TREATMENT)[0]
                )
                recoded = deepcopy(episode)
                recoded.values[:, treatment] = 1.0 - recoded.values[:, treatment]
                recoded.target *= -1.0
                pairs.append(("treatment_recoding", episode, recoded, -1.0, 0.0))

            if episode.values.shape[1] < model.config.max_variables:
                irrelevant = rng.normal(size=len(episode.values)).astype(np.float32)
                pairs.append(
                    (
                        "irrelevant_feature_insertion",
                        episode,
                        _add_covariate(episode, irrelevant),
                        1.0,
                        0.0,
                    )
                )
                covariates = np.flatnonzero(
                    np.isin(episode.roles, [ROLE_COVARIATE, ROLE_PREDICTOR])
                )
                duplicated = episode.values[:, covariates[0]] + rng.normal(
                    0.0, 1e-6, len(episode.values)
                )
                pairs.append(
                    (
                        "near_duplicate_feature",
                        episode,
                        _add_covariate(episode, duplicated.astype(np.float32)),
                        1.0,
                        0.0,
                    )
                )
                near_constant = rng.normal(0.0, 1e-7, len(episode.values))
                pairs.append(
                    (
                        "near_constant_feature",
                        episode,
                        _add_covariate(episode, near_constant.astype(np.float32)),
                        1.0,
                        0.0,
                    )
                )

    records: list[dict[str, Any]] = []
    for start in range(0, len(pairs), 16):
        block = pairs[start : start + 16]
        episodes = [item for pair in block for item in pair[1:3]]
        predictions = _predict_model(model, episodes, device)
        for pair_index, (probe, original, revised, factor, offset) in enumerate(block):
            original_prediction = predictions[2 * pair_index]["compiled"]
            revised_prediction = predictions[2 * pair_index + 1]["compiled"]
            expected = factor * original_prediction + offset
            difference = revised_prediction - expected
            records.append(
                {
                    "scenario": original.scenario,
                    "seed": original.seed,
                    "probe": probe,
                    "original_estimate": original_prediction,
                    "transformed_estimate": revised_prediction,
                    "expected_transformed_estimate": expected,
                    "absolute_equivariance_error": abs(difference),
                    "relative_equivariance_error": abs(difference)
                    / max(abs(expected), 1e-8),
                }
            )
    frame = pd.DataFrame(records)
    exact = frame[
        frame["probe"].isin(
            [
                "row_permutation",
                "covariate_permutation",
                "outcome_translation",
                "positive_outcome_scaling",
            ]
        )
    ]
    summary = {
        probe: {
            "episodes": len(group),
            "mean_absolute_equivariance_error": float(
                group["absolute_equivariance_error"].mean()
            ),
            "maximum_absolute_equivariance_error": float(
                group["absolute_equivariance_error"].max()
            ),
        }
        for probe, group in frame.groupby("probe")
    }
    summary["exact_invariance_pass_rate_at_1e-5"] = float(
        (exact["absolute_equivariance_error"] <= 1e-5).mean()
    )
    return frame, summary


def run(
    protocol_path: Path,
    output_dir: Path,
    device_name: str = "auto",
    quick: bool = False,
) -> dict[str, Any]:
    started = time.time()
    protocol = json.loads(protocol_path.read_text())
    checkpoint = Path(protocol["checkpoint"]["path"])
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("Frozen checkpoint hash does not match revision protocol")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    torch.set_num_threads(min(16, torch.get_num_threads()))
    model = _load_model(checkpoint, device)
    streams = protocol["streams"]

    def count(name: str, key: str) -> int:
        return 8 if quick else int(streams[name][key])

    def seed_start(name: str) -> int:
        # Smoke tests must never inspect any episode from a frozen stream.
        return int(streams[name]["seed_start"]) + (9_000_000 if quick else 0)

    tuning = _collect(
        model,
        device,
        EFFECT_SCENARIOS,
        seed_start("aggregation_tuning"),
        count("aggregation_tuning", "episodes_per_scenario"),
        include_baselines=False,
    )
    policy = AggregationPolicy.fit(tuning, protocol["temperature_candidates"])
    calibration = _collect(
        model,
        device,
        EFFECT_SCENARIOS,
        seed_start("effect_calibration"),
        count("effect_calibration", "episodes_per_scenario"),
        include_baselines=True,
    )
    radii, calibration_scores = _calibrate(
        calibration, policy, float(protocol["uncertainty"]["coverage"])
    )
    test = _collect(
        model,
        device,
        EFFECT_SCENARIOS,
        seed_start("confirmatory_test"),
        count("confirmatory_test", "episodes_per_scenario"),
        include_baselines=True,
    )
    per_seed = _evaluate(test, policy, radii)
    by_cell, overall = _summaries(per_seed)
    comparisons = _paired_bootstrap(
        per_seed,
        200 if quick else int(protocol["bootstrap_replicates"]),
    )
    contract_frame, contract_summary = _contract_property_test(
        model,
        device,
        policy,
        seed_start("contract_property_test"),
        count("contract_property_test", "episodes_per_scenario"),
    )
    robustness_frame, robustness_summary = _robustness_bank(
        model,
        device,
        seed_start("invariance_property_test"),
        count("invariance_property_test", "episodes_per_effect_scenario"),
    )
    external_path = Path(protocol["external_benchmark"]["source_path"])
    if _sha256(external_path) != protocol["external_benchmark"]["source_sha256"]:
        raise RuntimeError("External energy archive hash does not match protocol")
    external_calibration = _collect_from_episodes(
        model,
        device,
        (
            generate_appliances_energy_episode(
                seed_start("external_energy_calibration") + index,
                external_path,
                "calibration",
            )
            for index in range(count("external_energy_calibration", "episodes"))
        ),
        include_baselines=True,
    )
    external_radii, external_calibration_scores = _calibrate(
        external_calibration,
        policy,
        float(protocol["uncertainty"]["coverage"]),
    )
    external_test = _collect_from_episodes(
        model,
        device,
        (
            generate_appliances_energy_episode(
                seed_start("external_energy_test") + index,
                external_path,
                "test",
            )
            for index in range(count("external_energy_test", "episodes"))
        ),
        include_baselines=True,
    )
    external_per_seed = _evaluate(external_test, policy, external_radii)
    external_by_cell, external_overall = _summaries(external_per_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_records = []
    for row in tuning:
        for method, estimate in policy.predict(row).items():
            tuning_records.append(
                {
                    "scenario": row["scenario"],
                    "task": row["task"],
                    "seed": row["seed"],
                    "method": method,
                    "truth": row["truth"],
                    "estimate": estimate,
                    "absolute_error": abs(estimate - row["truth"]),
                }
            )
    pd.DataFrame(tuning_records).to_csv(
        output_dir / "aggregation_tuning.csv", index=False
    )
    calibration_scores.to_csv(output_dir / "calibration_scores.csv", index=False)
    per_seed.to_csv(output_dir / "confirmatory_per_seed.csv", index=False)
    by_cell.to_csv(output_dir / "confirmatory_by_cell.csv", index=False)
    overall.to_csv(output_dir / "confirmatory_overall.csv", index=False)
    contract_frame.to_csv(output_dir / "contract_compliance.csv", index=False)
    robustness_frame.to_csv(output_dir / "robustness.csv", index=False)
    external_calibration_scores.to_csv(
        output_dir / "external_calibration_scores.csv", index=False
    )
    external_per_seed.to_csv(output_dir / "external_per_seed.csv", index=False)
    external_by_cell.to_csv(output_dir / "external_by_cell.csv", index=False)
    external_overall.to_csv(output_dir / "external_overall.csv", index=False)
    (output_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    manifest = {
        "status": "completed",
        "quick_smoke_run": quick,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "foundation_model_retrained": False,
        "estimator_layer_corrected_before_new_streams": True,
        "aggregation_policy": policy.describe(),
        "conformal_radii": radii,
        "external_conformal_radii": external_radii,
        "external_data": {
            "path": str(external_path),
            "sha256": _sha256(external_path),
            "calibration_episodes": len(external_calibration),
            "test_episodes": len(external_test),
        },
        "contract_property_test": contract_summary,
        "robustness": robustness_summary,
        "paired_comparisons": comparisons,
        "counts": {
            "aggregation_tuning": len(tuning),
            "effect_calibration": len(calibration),
            "confirmatory_test": len(test),
            "external_energy_calibration": len(external_calibration),
            "external_energy_test": len(external_test),
        },
        "source_sha256": {
            str(path): _sha256(path)
            for path in [
                Path("cwfm/estimators.py"),
                Path("cwfm/baselines.py"),
                Path("cwfm/experiment.py"),
                Path("cwfm/revision_experiment.py"),
                Path("cwfm/external_energy.py"),
            ]
        },
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen no-retraining revision evaluation"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/revision_protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cwfm/revision"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run(args.protocol, args.output_dir, args.device, args.quick)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
