"""Finite-sample audit of the unchanged public empirical support thresholds.

Run: python -m cwfm.support_experiment
No neural inference, effect-estimator fitting, training, or threshold selection is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.stats import binom, chi2, norm

from .application.contracts import ObservedCase, Task
from .application.validation import evaluate_ate_support, evaluate_exposure_support
from .data import ROLE_COVARIATE, ROLE_EXPOSURE, ROLE_OUTCOME, ROLE_TREATMENT


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = float(norm.ppf(0.975))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def population_ate_support(family: str, severity: float) -> tuple[bool, float]:
    """Apply the fixed tail policy to the known population propensity law.

    This reference is practical overlap under a specified policy, not a test
    of strict positivity. All finite logistic severities remain strictly positive.
    """
    if severity == 0:
        return True, 0.0
    if family == "linear_logistic":
        outside = 2 * norm.sf(logit(0.95) / severity)
        q05, q95 = expit(severity * norm.ppf([0.05, 0.95]))
    elif family == "quadratic_logistic":
        outside = chi2.cdf(1 + logit(0.05) / severity, 1) + chi2.sf(1 + logit(0.95) / severity, 1)
        q05, q95 = expit(severity * (chi2.ppf([0.05, 0.95], 1) - 1))
    else:
        raise ValueError(f"Unknown assignment family: {family}")
    return bool(outside <= 0.10 and q05 > 0.01 and q95 < 0.99), float(outside)


def run(protocol: dict) -> pd.DataFrame:
    records = []
    for family_index, family in enumerate(protocol["ate_families"]):
        for n in protocol["sample_sizes"]:
            for severity_index, severity in enumerate(protocol["ate_severities"]):
                population_ok, population_tail = population_ate_support(family, severity)
                for replicate in range(protocol["replicates"]):
                    seed = [protocol["seed"], family_index, n, severity_index, replicate]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    x = rng.normal(size=(n, protocol["covariates"]))
                    signal = x[:, 0] if family == "linear_logistic" else x[:, 0] ** 2 - 1
                    true_propensity = expit(severity * signal)
                    a = rng.binomial(1, true_propensity)
                    # Outcome is irrelevant to both support functions.
                    case = ObservedCase(np.column_stack([x, a, np.zeros(n)]),
                        tuple(f"X{i}" for i in range(x.shape[1])) + ("A", "Y"),
                        np.array([ROLE_COVARIATE] * x.shape[1] + [ROLE_TREATMENT, ROLE_OUTCOME]),
                        Task.STATIC_ATE)
                    screen_ok, diagnostics, _ = evaluate_ate_support(case)
                    arm_ok = min(a.sum(), n - a.sum()) >= 10
                    records.append(dict(family=family, n=n, severity=severity,
                        replicate=replicate, population_adequate=population_ok,
                        population_tail=population_tail, screen_refused=not screen_ok,
                        arm_refused=not arm_ok, refused=not (screen_ok and arm_ok),
                        fitted_tail=diagnostics["outside_0.05_0.95_fraction"]))
    for n in protocol["sample_sizes"]:
        for severity_index, probability in enumerate(protocol["network_assignment_probabilities"]):
            # A ring with neighbors at offsets -2, -1, +1, +2. Independent
            # treatment makes each exposure marginally Binomial(4, p)/4,
            # independent of own treatment, despite dependence between neighbors.
            q05, q95 = binom.ppf([0.05, 0.95], 4, probability) / 4
            population_ok = bool(q05 <= 0.25 < 0.75 <= q95)
            for replicate in range(protocol["replicates"]):
                seed = [protocol["seed"], 2, n, severity_index, replicate]
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                a = rng.binomial(1, probability, n)
                g = sum(np.roll(a, offset) for offset in (-2, -1, 1, 2)) / 4
                case = ObservedCase(np.column_stack([a, g, np.zeros(n)]),
                    ("A", "G", "Y"), np.array([ROLE_TREATMENT, ROLE_EXPOSURE, ROLE_OUTCOME]),
                    Task.NETWORK_INTERFERENCE)
                screen_ok, _, _ = evaluate_exposure_support(case, 0.25, 0.75)
                arm_ok = min(a.sum(), n - a.sum()) >= 10
                records.append(dict(family="network_ring", n=n, severity=probability,
                    replicate=replicate, population_adequate=population_ok,
                    population_tail=np.nan, screen_refused=not screen_ok,
                    arm_refused=not arm_ok, refused=not (screen_ok and arm_ok),
                    fitted_tail=np.nan))
    return pd.DataFrame(records)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, n, severity), group in frame.groupby(["family", "n", "severity"]):
        count = len(group)
        refused = int(group.refused.sum())
        rate = refused / count
        lo, hi = wilson(refused, count)
        adequate = bool(group.population_adequate.iloc[0])
        rows.append(dict(family=family, n=n, severity=severity, episodes=count,
            population_adequate=adequate, refusal_rate=rate,
            refusal_ci_low=lo, refusal_ci_high=hi,
            false_refusal_rate=rate if adequate else np.nan,
            false_acceptance_rate=1-rate if not adequate else np.nan,
            screen_refusal_rate=float(group.screen_refused.mean()),
            arm_refusal_rate=float(group.arm_refused.mean()),
            mean_fitted_tail=float(group.fitted_tail.mean()),
            population_tail=float(group.population_tail.iloc[0])))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cwfm/support_revision"))
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    frame = run(protocol)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    hash_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = dict(protocol=protocol, protocol_sha256=hash_file(args.protocol),
        experiment_sha256=hash_file(__file__),
        validation_sha256=hash_file("cwfm/application/validation.py"),
        runner_sha256=hash_file("cwfm/application/runner.py"),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__,
        episodes=len(frame), elapsed_seconds=time.perf_counter()-started,
        neural_training=False, thresholds_tuned=False,
        reference="Known population practical-overlap policy, not strict positivity or causal correctness",
        intervals="Pointwise 95 percent Wilson intervals over independent episode replicates",
        network_reference="Population 5th/95th quantiles of Binomial(4,p)/4; count conditions are finite-sample only",
        artifact_sha256={name: hash_file(args.output_dir / name) for name in ("per_seed.csv", "summary.csv")})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(frame), "elapsed_seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
