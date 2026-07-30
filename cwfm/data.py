from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch


TASK_ATE = 0
TASK_REGIME = 1
TASK_INTERFERENCE = 2

ROLE_COVARIATE = 0
ROLE_TREATMENT = 1
ROLE_OUTCOME = 2
ROLE_EXPOSURE = 3
ROLE_PREDICTOR = 4
ROLE_PADDING = 5

DESIGN_OBSERVED = 0
DESIGN_RANDOMIZED = 1
DESIGN_NONIDENTIFIED = 2
DESIGN_POOR_SUPPORT = 3


@dataclass
class Episode:
    values: np.ndarray
    roles: np.ndarray
    task: int
    design: int
    target: float
    identified: int
    supported: int
    structure_target: int
    mechanism: int
    route_flexible: float
    graph: np.ndarray
    adjacency: np.ndarray
    scenario: str
    seed: int
    threshold: float = 0.0
    change_type: int = 0
    query_exposure: tuple[float, float] = (0.25, 0.75)
    cluster_ids: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    metadata: dict[str, str | int | float] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _empty_graph(p: int) -> np.ndarray:
    return np.zeros((p, p), dtype=np.float32)


def generate_ate(
    seed: int,
    scenario: str = "random",
    n: int = 96,
    p: int = 6,
) -> Episode:
    rng = np.random.default_rng(seed)
    if scenario == "random":
        scenario = rng.choice(
            ["linear", "nonlinear", "randomized", "poor_overlap", "hidden_confounding"]
        ).item()
    rho = rng.uniform(0.0, 0.55)
    covariance = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    x = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    u = rng.normal(size=n)
    beta = rng.normal(0.0, 0.45, size=p)
    tau0 = rng.uniform(-1.5, 1.5)
    heterogeneity = (
        rng.uniform(-0.65, 0.65)
        if scenario in {"nonlinear", "ood_multiplicative"}
        else 0.0
    )
    if scenario == "randomized":
        propensity = np.full(n, rng.uniform(0.3, 0.7))
        design = DESIGN_RANDOMIZED
    elif scenario == "poor_overlap":
        propensity = _sigmoid(4.5 * x[:, 0] + 2.0 * x[:, 1])
        design = DESIGN_POOR_SUPPORT
    elif scenario == "hidden_confounding":
        propensity = _sigmoid(0.8 * x[:, 0] - 0.5 * x[:, 1] + 1.3 * u)
        design = DESIGN_NONIDENTIFIED
    else:
        propensity = _sigmoid(0.65 * x[:, 0] - 0.45 * x[:, 1] + 0.2 * x[:, 2])
        design = DESIGN_OBSERVED
    a = rng.binomial(1, propensity).astype(float)
    tau = tau0 + heterogeneity * np.tanh(x[:, 0])
    base = x @ beta
    if scenario == "nonlinear":
        base += 0.8 * np.sin(x[:, 0]) + 0.45 * (x[:, 1] ** 2 - 1.0)
    elif scenario == "ood_hinge":
        base += 0.9 * np.maximum(x[:, 0] - 0.2, 0.0) - 0.6 * np.maximum(
            -x[:, 1] - 0.3, 0.0
        )
    elif scenario == "ood_multiplicative":
        base += 0.7 * x[:, 0] * x[:, 1]
        tau += 0.45 * x[:, 0] * x[:, 1]
    elif scenario == "ood_saturation":
        base += 1.1 * np.tanh(1.7 * x[:, 0] - 0.8 * x[:, 1])
    hidden_term = 1.2 * u if scenario == "hidden_confounding" else 0.0
    y = base + tau * a + hidden_term + rng.normal(0.0, rng.uniform(0.65, 1.05), n)
    values = np.column_stack([x, a, y]).astype(np.float32)
    roles = np.asarray(
        [ROLE_COVARIATE] * p + [ROLE_TREATMENT, ROLE_OUTCOME], dtype=np.int64
    )
    graph = _empty_graph(p + 2)
    graph[:p, p] = 1
    graph[:p, p + 1] = 1
    graph[p, p + 1] = 1
    target = float(np.mean(tau))
    return Episode(
        values,
        roles,
        TASK_ATE,
        design,
        target,
        int(scenario != "hidden_confounding"),
        int(scenario != "poor_overlap"),
        p + 2,
        int(scenario in {"nonlinear", "ood_hinge", "ood_multiplicative", "ood_saturation"}),
        float(scenario in {"nonlinear", "ood_hinge", "ood_multiplicative", "ood_saturation"}),
        graph,
        np.zeros((n, n), dtype=np.float32),
        f"ate_{scenario}",
        seed,
        cluster_ids=np.arange(n, dtype=np.int64),
        metadata={
            "generator_version": 2,
            "template_id": f"ate:{scenario}",
            "mechanism_family": scenario,
            "graph_family": "iid",
            "sample_size": n,
            "variable_count": p + 2,
            "signal_strength": "weak" if abs(tau0) < 0.5 else "regular",
            "overlap_bucket": "poor" if scenario == "poor_overlap" else "regular",
        },
    )


def generate_regime(
    seed: int,
    scenario: str = "random",
    n: int = 96,
    p: int = 7,
) -> Episode:
    rng = np.random.default_rng(seed)
    if scenario == "random":
        scenario = rng.choice(["linear_split", "weak_split", "stationary_nonlinear", "null"]).item()
    x = rng.normal(size=(n, p))
    beta = rng.normal(0.0, 0.45, p)
    regime_feature = int(rng.integers(2, p))
    threshold = rng.uniform(-0.45, 0.45)
    delta = rng.uniform(0.75, 1.45)
    if scenario == "weak_split":
        delta = rng.uniform(0.18, 0.38)
    has_split = scenario in {
        "linear_split",
        "weak_split",
        "ood_intercept_split",
        "ood_variance_split",
        "ood_threshold_split",
    }
    y = x @ beta
    change_type = 0
    if scenario == "ood_intercept_split":
        y += delta * (x[:, regime_feature] > threshold)
        change_type = 1
    elif scenario == "ood_variance_split":
        noise_scale = 0.35 + delta * (x[:, regime_feature] > threshold)
        y += rng.normal(0.0, noise_scale, n)
        change_type = 2
    elif has_split:
        if scenario == "ood_threshold_split":
            threshold = rng.uniform(-1.0, 1.0)
        y += delta * (x[:, regime_feature] > threshold) * x[:, 0]
    elif scenario == "stationary_nonlinear":
        y += rng.uniform(0.8, 1.4) * (x[:, 0] ** 2 - 1.0)
    if scenario != "ood_variance_split":
        y += rng.normal(0.0, 0.65, n)
    values = np.column_stack([x, y]).astype(np.float32)
    roles = np.asarray(
        [ROLE_PREDICTOR, ROLE_PREDICTOR]
        + [ROLE_COVARIATE] * (p - 2)
        + [ROLE_OUTCOME],
        dtype=np.int64,
    )
    graph = _empty_graph(p + 1)
    graph[:p, p] = 1
    return Episode(
        values,
        roles,
        TASK_REGIME,
        DESIGN_OBSERVED,
        float(delta if has_split else 0.0),
        1,
        1,
        regime_feature if has_split else p + 1,
        int(scenario == "stationary_nonlinear"),
        float(scenario == "stationary_nonlinear"),
        graph,
        np.zeros((n, n), dtype=np.float32),
        f"regime_{scenario}",
        seed,
        threshold=float(threshold),
        change_type=change_type,
        cluster_ids=np.arange(n, dtype=np.int64),
        metadata={
            "generator_version": 2,
            "template_id": f"regime:{scenario}",
            "mechanism_family": scenario,
            "graph_family": "iid",
            "sample_size": n,
            "variable_count": p + 1,
            "signal_strength": "weak" if scenario == "weak_split" else "regular",
            "overlap_bucket": "not_applicable",
        },
    )


def _network(
    rng: np.random.Generator,
    n: int,
    cluster_size: int = 12,
    family: str = "block",
) -> np.ndarray:
    w = np.zeros((n, n), dtype=np.float32)
    if family == "erdos_renyi":
        block = rng.random((n, n)) < 0.08
        block = np.triu(block, 1)
        return (block | block.T).astype(np.float32)
    if family == "small_world":
        for i in range(n):
            for distance in (1, 2):
                j = (i + distance) % n
                w[i, j] = w[j, i] = 1
        rewires = rng.random((n, n)) < 0.01
        rewires = np.triu(rewires, 1)
        return np.maximum(w, rewires | rewires.T).astype(np.float32)
    for start in range(0, n, cluster_size):
        stop = min(start + cluster_size, n)
        block = rng.random((stop - start, stop - start)) < 0.28
        block = np.triu(block, 1)
        block = block | block.T
        for i in range(len(block)):
            if not block[i].any() and len(block) > 1:
                j = (i + 1) % len(block)
                block[i, j] = block[j, i] = True
        w[start:stop, start:stop] = block
    return w


def generate_interference(
    seed: int,
    scenario: str = "random",
    n: int = 96,
    p: int = 4,
) -> Episode:
    rng = np.random.default_rng(seed)
    if scenario == "random":
        scenario = rng.choice(
            ["linear", "threshold", "null", "poor_support", "hidden_confounding"]
        ).item()
    x = rng.normal(size=(n, p))
    u = rng.normal(size=n)
    graph_family = (
        "small_world"
        if scenario == "ood_small_world"
        else "erdos_renyi"
        if scenario == "ood_erdos_renyi"
        else "block"
    )
    w = _network(rng, n, family=graph_family)
    cluster_size = 12
    clusters = np.arange(n) // cluster_size
    if scenario == "poor_support":
        saturations = rng.choice([0.02, 0.98], size=clusters.max() + 1)
        design = DESIGN_POOR_SUPPORT
    else:
        saturations = rng.choice([0.25, 0.5, 0.75], size=clusters.max() + 1)
        design = DESIGN_RANDOMIZED
    logits = np.log(saturations[clusters] / (1 - saturations[clusters]))
    if scenario == "hidden_confounding":
        logits = logits + 1.2 * u
        design = DESIGN_NONIDENTIFIED
    a = rng.binomial(1, _sigmoid(logits)).astype(float)
    degree = w.sum(1)
    g = (w @ a) / np.maximum(degree, 1.0)
    beta = 0.0 if scenario == "null" else rng.uniform(-1.4, 1.4)
    if abs(beta) < 0.45 and scenario != "null":
        beta = np.copysign(0.45, beta if beta else 1.0)
    if scenario in {"threshold", "ood_small_world", "ood_erdos_renyi"}:
        threshold = 0.5 if scenario == "threshold" else rng.uniform(0.3, 0.7)
        spill = beta * (g >= threshold)
        target = beta * (
            float(0.75 >= threshold) - float(0.25 >= threshold)
        )
    elif scenario == "ood_nonmonotone":
        threshold = 0.5
        spill = beta * np.sin(2 * np.pi * g)
        target = beta * (
            np.sin(2 * np.pi * 0.75) - np.sin(2 * np.pi * 0.25)
        )
    else:
        threshold = 0.5
        spill = beta * g
        target = 0.5 * beta
    hidden_term = 1.1 * u if scenario == "hidden_confounding" else 0.0
    y = (
        x @ rng.normal(0.0, 0.35, p)
        + rng.uniform(0.3, 0.9) * a
        + spill
        + hidden_term
        + rng.normal(0.0, 0.7, n)
    )
    degree_z = (degree - degree.mean()) / max(degree.std(), 1.0)
    values = np.column_stack([x, degree_z, a, g, y]).astype(np.float32)
    roles = np.asarray(
        [ROLE_COVARIATE] * (p + 1)
        + [ROLE_TREATMENT, ROLE_EXPOSURE, ROLE_OUTCOME],
        dtype=np.int64,
    )
    graph = _empty_graph(p + 4)
    graph[: p + 1, -3:] = 1
    graph[-3, -1] = 1
    graph[-2, -1] = 1
    return Episode(
        values,
        roles,
        TASK_INTERFERENCE,
        design,
        float(target),
        int(scenario != "hidden_confounding"),
        int(scenario != "poor_support"),
        p + 4,
        2 if scenario in {"threshold", "ood_small_world", "ood_erdos_renyi"} else 0,
        float(scenario in {"threshold", "ood_small_world", "ood_erdos_renyi", "ood_nonmonotone"}),
        graph,
        w,
        f"interference_{scenario}",
        seed,
        threshold=float(threshold),
        query_exposure=(0.25, 0.75),
        cluster_ids=clusters.astype(np.int64),
        metadata={
            "generator_version": 2,
            "template_id": f"interference:{scenario}",
            "mechanism_family": scenario,
            "graph_family": graph_family,
            "sample_size": n,
            "variable_count": p + 4,
            "signal_strength": "null" if scenario == "null" else "regular",
            "overlap_bucket": "poor" if scenario == "poor_support" else "regular",
        },
    )


def _randomize_presentation(ep: Episode, seed: int) -> Episode:
    """Remove row and role-preserving column-order simulator shortcuts."""
    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    row_order = rng.permutation(len(ep.values))
    ep.values = ep.values[row_order]
    ep.adjacency = ep.adjacency[np.ix_(row_order, row_order)]
    if len(ep.cluster_ids) == len(row_order):
        ep.cluster_ids = ep.cluster_ids[row_order]

    column_order = np.arange(ep.values.shape[1])
    for role in np.unique(ep.roles):
        members = np.flatnonzero(ep.roles == role)
        if len(members) > 1:
            column_order[members] = rng.permutation(members)
    inverse = np.argsort(column_order)
    ep.values = ep.values[:, column_order]
    ep.roles = ep.roles[column_order]
    ep.graph = ep.graph[np.ix_(column_order, column_order)]
    if ep.structure_target < len(column_order):
        ep.structure_target = int(inverse[ep.structure_target])
    scalable = np.flatnonzero(
        (ep.roles == ROLE_COVARIATE) | (ep.roles == ROLE_PREDICTOR)
    )
    if len(scalable):
        ep.values[:, scalable] *= rng.lognormal(
            mean=0.0, sigma=0.35, size=len(scalable)
        )
        missing_rate = float(rng.choice([0.0, 0.02, 0.05]))
        if missing_rate:
            missing = rng.random((len(ep.values), len(scalable))) < missing_rate
            block = ep.values[:, scalable]
            block[missing] = np.nan
            ep.values[:, scalable] = block
        ep.metadata["missingness_rate"] = missing_rate
    return ep


def generate_episode(seed: int, task: int | None = None, scenario: str = "random") -> Episode:
    rng = np.random.default_rng(seed)
    task = int(rng.integers(0, 3)) if task is None else task
    training_draw = scenario == "random"
    if training_draw:
        # Keep scenario selection independent of task selection. Reusing the
        # first PCG64 draw created an accidental task/scenario confound.
        scenario_rng = np.random.default_rng(seed ^ 0x5DEECE66D)
        options = {
            TASK_ATE: [
                "linear",
                "nonlinear",
                "randomized",
                "poor_overlap",
                "hidden_confounding",
            ],
            TASK_REGIME: [
                "linear_split",
                "weak_split",
                "stationary_nonlinear",
                "null",
            ],
            TASK_INTERFERENCE: [
                "linear",
                "threshold",
                "null",
                "poor_support",
                "hidden_confounding",
            ],
        }
        scenario = scenario_rng.choice(options[task]).item()
    size_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    n = int(size_rng.choice([64, 80, 96, 128])) if training_draw else 96
    if task == TASK_ATE:
        p = int(size_rng.integers(3, 9)) if training_draw else 6
        episode = generate_ate(seed, scenario, n=n, p=p)
    elif task == TASK_REGIME:
        p = int(size_rng.integers(5, 10)) if training_draw else 7
        episode = generate_regime(seed, scenario, n=n, p=p)
    elif task == TASK_INTERFERENCE:
        p = int(size_rng.integers(3, 7)) if training_draw else 4
        episode = generate_interference(seed, scenario, n=n, p=p)
    else:
        raise ValueError(f"Unknown task {task}")
    return _randomize_presentation(episode, seed) if training_draw else episode


def collate_episodes(
    episodes: Iterable[Episode],
    max_variables: int = 12,
) -> dict[str, torch.Tensor | list[str] | list[Episode]]:
    episodes = list(episodes)
    batch = len(episodes)
    rows = max(ep.values.shape[0] for ep in episodes)
    values = np.zeros((batch, rows, max_variables), dtype=np.float32)
    missing = np.ones_like(values)
    roles = np.full((batch, max_variables), ROLE_PADDING, dtype=np.int64)
    variable_mask = np.zeros((batch, max_variables), dtype=bool)
    adjacency = np.zeros((batch, rows, rows), dtype=np.float32)
    row_mask = np.zeros((batch, rows), dtype=bool)
    cluster_ids = np.full((batch, rows), -1, dtype=np.int64)
    graph = np.zeros((batch, max_variables, max_variables), dtype=np.float32)
    structure = np.full(batch, max_variables, dtype=np.int64)
    raw_scale = np.ones(batch, dtype=np.float32)
    target_scaled = np.zeros(batch, dtype=np.float32)
    expert_estimates = np.zeros((batch, 6), dtype=np.float32)
    expert_standard_errors = np.ones((batch, 6), dtype=np.float32)
    expert_mask = np.zeros((batch, 6), dtype=bool)
    regime_threshold = np.zeros(batch, dtype=np.float32)
    change_type = np.zeros(batch, dtype=np.int64)
    query = np.zeros((batch, 8), dtype=np.float32)
    nuisance: list[dict[str, float]] = []
    regime_evidence = np.zeros((batch, max_variables, 5, 6), dtype=np.float32)
    threshold_values = np.zeros((batch, max_variables, 5), dtype=np.float32)
    from .estimators import estimator_experts

    for i, ep in enumerate(episodes):
        n, p = ep.values.shape
        if p > max_variables:
            raise ValueError(f"Episode has {p} variables; maximum is {max_variables}")
        observed = ep.values.astype(np.float32)
        mean = np.nanmean(observed, axis=0)
        scale = np.nanstd(observed, axis=0)
        scale[scale < 1e-5] = 1.0
        z = np.nan_to_num((observed - mean) / scale)
        raw_role = np.isin(
            ep.roles, [ROLE_TREATMENT, ROLE_EXPOSURE]
        )
        z[:, raw_role] = np.nan_to_num(observed[:, raw_role])
        values[i, :n, :p] = z
        missing[i, :n, :p] = np.isnan(observed)
        roles[i, :p] = ep.roles
        variable_mask[i, :p] = True
        adjacency[i, :n, :n] = ep.adjacency
        row_mask[i, :n] = True
        if len(ep.cluster_ids) == n:
            cluster_ids[i, :n] = ep.cluster_ids
        graph[i, :p, :p] = ep.graph
        structure[i] = ep.structure_target if ep.structure_target < p else max_variables
        outcome_indices = np.flatnonzero(ep.roles == ROLE_OUTCOME)
        y_scale = float(scale[outcome_indices[0]]) if len(outcome_indices) else 1.0
        raw_scale[i] = y_scale
        target_scaled[i] = ep.target / max(y_scale, 1e-5)
        expert = estimator_experts(ep)
        expert_estimates[i] = expert.estimates / max(y_scale, 1e-5)
        expert_standard_errors[i] = expert.standard_errors / max(y_scale, 1e-5)
        expert_mask[i] = expert.mask
        nuisance.append(expert.nuisance)
        if ep.task == TASK_REGIME and ep.structure_target < p:
            regime_threshold[i] = (
                ep.threshold - float(mean[ep.structure_target])
            ) / max(float(scale[ep.structure_target]), 1e-5)
        else:
            regime_threshold[i] = ep.threshold
        change_type[i] = ep.change_type
        treatment_indices = np.flatnonzero(ep.roles == ROLE_TREATMENT)
        outcome_indices = np.flatnonzero(ep.roles == ROLE_OUTCOME)
        exposure_indices = np.flatnonzero(ep.roles == ROLE_EXPOSURE)
        query[i] = np.asarray(
            [
                treatment_indices[0] / max_variables if len(treatment_indices) else -1,
                outcome_indices[0] / max_variables if len(outcome_indices) else -1,
                exposure_indices[0] / max_variables if len(exposure_indices) else -1,
                0.0,
                1.0,
                ep.query_exposure[0],
                ep.query_exposure[1],
                float(ep.design),
            ]
        )
        if ep.task == TASK_REGIME:
            outcome = int(outcome_indices[0])
            predictors = np.flatnonzero(
                (ep.roles == ROLE_PREDICTOR) | (ep.roles == ROLE_COVARIATE)
            )
            response = z[:, outcome]
            base_x = np.column_stack([np.ones(n), z[:, predictors]])
            base_coef = np.linalg.lstsq(base_x, response, rcond=None)[0]
            base_residual = response - base_x @ base_coef
            base_sse = max(float(base_residual @ base_residual), 1e-8)
            for variable in np.flatnonzero(ep.roles == ROLE_COVARIATE):
                for q_index, threshold_value in enumerate(
                    np.quantile(z[:, variable], [0.2, 0.35, 0.5, 0.65, 0.8])
                ):
                    side = z[:, variable] > threshold_value
                    left, right = ~side, side
                    if left.sum() < 10 or right.sum() < 10:
                        continue
                    interaction = side[:, None] * base_x[:, 1:]
                    candidate_x = np.column_stack([base_x, side, interaction])
                    coefficient = np.linalg.lstsq(candidate_x, response, rcond=None)[0]
                    residual = response - candidate_x @ coefficient
                    sse = max(float(residual @ residual), 1e-8)
                    bic_gain = (
                        n * np.log(base_sse / n)
                        + base_x.shape[1] * np.log(n)
                        - n * np.log(sse / n)
                        - candidate_x.shape[1] * np.log(n)
                    )
                    regime_evidence[i, variable, q_index] = np.asarray(
                        [
                            bic_gain / max(n, 1),
                            abs(float(coefficient[-len(predictors) - 1])),
                            float(base_residual[left].var() - base_residual[right].var()),
                            float(left.mean()),
                            float(right.mean()),
                            float(q_index) / 4.0,
                        ],
                        dtype=np.float32,
                    )
                    threshold_values[i, variable, q_index] = threshold_value
    return {
        "values": torch.from_numpy(values),
        "missing": torch.from_numpy(missing),
        "roles": torch.from_numpy(roles),
        "variable_mask": torch.from_numpy(variable_mask),
        "adjacency": torch.from_numpy(adjacency),
        "row_mask": torch.from_numpy(row_mask),
        "cluster_ids": torch.from_numpy(cluster_ids),
        "task": torch.tensor([ep.task for ep in episodes], dtype=torch.long),
        "design": torch.tensor([ep.design for ep in episodes], dtype=torch.long),
        "target": torch.from_numpy(target_scaled),
        "raw_target": torch.tensor([ep.target for ep in episodes], dtype=torch.float32),
        "raw_scale": torch.from_numpy(raw_scale),
        "identified": torch.tensor([ep.identified for ep in episodes], dtype=torch.float32),
        "supported": torch.tensor([ep.supported for ep in episodes], dtype=torch.float32),
        "structure_target": torch.from_numpy(structure),
        "mechanism": torch.tensor([ep.mechanism for ep in episodes], dtype=torch.long),
        "route_flexible": torch.tensor(
            [ep.route_flexible for ep in episodes], dtype=torch.float32
        ),
        "graph": torch.from_numpy(graph),
        "expert_estimates": torch.from_numpy(expert_estimates),
        "expert_standard_errors": torch.from_numpy(expert_standard_errors),
        "expert_mask": torch.from_numpy(expert_mask),
        "regime_threshold": torch.from_numpy(regime_threshold),
        "change_type": torch.from_numpy(change_type),
        "query": torch.from_numpy(query),
        "regime_evidence": torch.from_numpy(regime_evidence),
        "threshold_values": torch.from_numpy(threshold_values),
        "split_target": torch.tensor(
            [ep.structure_target < len(ep.roles) for ep in episodes],
            dtype=torch.float32,
        ),
        "nuisance": nuisance,
        "metadata": [ep.metadata for ep in episodes],
        "scenarios": [ep.scenario for ep in episodes],
        "episodes": episodes,
    }


EVALUATION_SCENARIOS = [
    (TASK_ATE, "linear"),
    (TASK_ATE, "nonlinear"),
    (TASK_ATE, "randomized"),
    (TASK_ATE, "poor_overlap"),
    (TASK_ATE, "hidden_confounding"),
    (TASK_REGIME, "linear_split"),
    (TASK_REGIME, "weak_split"),
    (TASK_REGIME, "stationary_nonlinear"),
    (TASK_REGIME, "null"),
    (TASK_INTERFERENCE, "linear"),
    (TASK_INTERFERENCE, "threshold"),
    (TASK_INTERFERENCE, "null"),
    (TASK_INTERFERENCE, "poor_support"),
    (TASK_INTERFERENCE, "hidden_confounding"),
]


OOD_DEVELOPMENT_SCENARIOS = [
    (TASK_ATE, "ood_hinge"),
    (TASK_ATE, "ood_multiplicative"),
    (TASK_ATE, "ood_saturation"),
    (TASK_REGIME, "ood_intercept_split"),
    (TASK_REGIME, "ood_variance_split"),
    (TASK_REGIME, "ood_threshold_split"),
    (TASK_INTERFERENCE, "ood_small_world"),
    (TASK_INTERFERENCE, "ood_erdos_renyi"),
    (TASK_INTERFERENCE, "ood_nonmonotone"),
]
