from __future__ import annotations

from dataclasses import dataclass
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
    heterogeneity = rng.uniform(-0.65, 0.65) if scenario == "nonlinear" else 0.0
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
        int(scenario == "nonlinear"),
        float(scenario == "nonlinear"),
        graph,
        np.zeros((n, n), dtype=np.float32),
        f"ate_{scenario}",
        seed,
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
    has_split = scenario in {"linear_split", "weak_split"}
    y = x @ beta
    if has_split:
        y += delta * (x[:, regime_feature] > threshold) * x[:, 0]
    elif scenario == "stationary_nonlinear":
        y += rng.uniform(0.8, 1.4) * (x[:, 0] ** 2 - 1.0)
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
    )


def _network(rng: np.random.Generator, n: int, cluster_size: int = 12) -> np.ndarray:
    w = np.zeros((n, n), dtype=np.float32)
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
    w = _network(rng, n)
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
    if scenario == "threshold":
        spill = beta * (g >= 0.5)
        target = beta
    else:
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
        2 if scenario == "threshold" else 0,
        float(scenario == "threshold"),
        graph,
        w,
        f"interference_{scenario}",
        seed,
    )


def generate_episode(seed: int, task: int | None = None, scenario: str = "random") -> Episode:
    rng = np.random.default_rng(seed)
    task = int(rng.integers(0, 3)) if task is None else task
    if scenario == "random":
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
    if task == TASK_ATE:
        return generate_ate(seed, scenario)
    if task == TASK_REGIME:
        return generate_regime(seed, scenario)
    if task == TASK_INTERFERENCE:
        return generate_interference(seed, scenario)
    raise ValueError(f"Unknown task {task}")


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
    graph = np.zeros((batch, max_variables, max_variables), dtype=np.float32)
    structure = np.full(batch, max_variables, dtype=np.int64)
    raw_scale = np.ones(batch, dtype=np.float32)
    target_scaled = np.zeros(batch, dtype=np.float32)
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
        graph[i, :p, :p] = ep.graph
        structure[i] = ep.structure_target if ep.structure_target < p else max_variables
        outcome_indices = np.flatnonzero(ep.roles == ROLE_OUTCOME)
        y_scale = float(scale[outcome_indices[0]]) if len(outcome_indices) else 1.0
        raw_scale[i] = y_scale
        target_scaled[i] = ep.target / max(y_scale, 1e-5)
    return {
        "values": torch.from_numpy(values),
        "missing": torch.from_numpy(missing),
        "roles": torch.from_numpy(roles),
        "variable_mask": torch.from_numpy(variable_mask),
        "adjacency": torch.from_numpy(adjacency),
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
