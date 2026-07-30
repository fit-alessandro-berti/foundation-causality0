# Revised CWFM experimental results

## Executive summary

The revised Causal World Foundation Model (CWFM) experiment is complete. It
implements the architecture and overfitting guardrails identified after the
Phase-I run, retrains for 1,600 steps on 38,400 online synthetic episodes, and
evaluates a locked 30-seed final bank after separate development and
calibration stages.

The main result is a substantial improvement in supported, identified effect
estimation:

| Confirmatory effect result | Phase I | Revised run |
| --- | ---: | ---: |
| Pooled CWFM MAE | 0.271 | **0.166** |
| Target-90% interval coverage | 0.917 | **0.906** |
| Static ATE MAE | 0.324 | **0.169** |
| Network-interference MAE | 0.218 | **0.164** |

The pooled MAE fell by 38.6% relative to the Phase-I fixed hybrid. The revised
CWFM is statistically indistinguishable from the linear and interaction
specialists on the locked effect bank: its paired MAE difference versus linear
g-computation is -0.0028 (95% bootstrap CI [-0.0132, 0.0080]) and versus
interaction g-computation is -0.0003 [-0.0111, 0.0109]. It is materially better
than the random-forest baseline, with a paired MAE difference of -0.1629
[-0.2031, -0.1218].

The regime branch also improved. Exact accuracy rose from 0.508 to 0.617.
Strong-split recall rose from 0.067 to 0.633, while the complete-null
false-positive rate remained zero. Stationary nonlinear hard negatives still
produced a 0.167 false-positive rate and weak splits remained undetected, so
regime detection is improved but not solved.

The result is deliberately conservative. The selected checkpoint keeps the
neural residual gate almost closed, making the final point estimate numerically
equal to the risk-routed compiled estimator. This is a successful no-harm
outcome, not evidence that the learned residual or world particles add value.
The report therefore attributes the gain to the query-correct estimator layer,
risk-supervised routing architecture, deterministic safety logic, and
constraint-based selection—not to a free-form neural effect decoder.

## What changed

The fixed 50/50 average and incorrect differentiable anchor were removed. The
revised executable system contains:

- a task-specific library of query-correct effect estimators;
- out-of-fold orthogonal nuisance estimates for static ATE;
- cluster/component-held-out nuisance estimates for interference;
- linear, spline, interaction, piecewise, randomized-design, and null experts;
- an episode-level risk-supervised estimator router;
- a compiled anchor followed by a zero-initialized gated neural residual;
- a no-harm loss and uncertainty-normalized correction penalty;
- exact Gaussian-mixture likelihood and the corrected mixture variance;
- structural world decoders for mechanisms and graphs;
- a shared raw-data encoder plus statistical-summary and formal-query streams;
- task-specific adapters and decoders;
- a cluster-aware graph message path;
- deterministic formal identification and learned empirical support diagnostics;
- hierarchical regime detection, variable localization, threshold localization,
  change type, and magnitude heads;
- role-eligible regime masking;
- finite-sample split-conformal calibration;
- append-only training, validation, and episode diagnostic logs.

The simulator now randomizes sample size, variable count, row order,
role-preserving column order, covariate scales, and covariate missingness.
Generated episodes carry template, mechanism, graph-family, sample-size,
variable-count, signal, overlap, and generator-version metadata. These fields
are logged for grouping but are not passed directly to the main network.

## Strictly separated streams

The run uses five non-overlapping streams.

| Stream | Seed range or content | Use |
| --- | --- | --- |
| Online training | starts at 1,729 | Gradient updates |
| ID development | starts at 10,001,729 | Checkpoint eligibility and ID ranking |
| OOD development | starts at 11,001,729; held-out mechanisms and graph families | Robustness constraint |
| Calibration | starts at 19,000,000 | Intervals and regime threshold only |
| Locked final test | starts at 20,000,000 | One-time report |

The OOD development families are held out from online training: hinge,
multiplicative, and saturation outcomes; intercept, variance, and shifted
regime changes; small-world and Erdős–Rényi graphs; and nonmonotone
interference.

The final test contains the same 14 declared scenario families as Phase I so
that the before/after comparison is direct. Each cell has 30 seeds. The
calibration bank contains 360 fresh episodes.

## Training and checkpoint selection

The model was trained on one NVIDIA L40S with PyTorch 2.13.0+cu130. Runtime was
2,000.5 seconds. Training was staged:

1. compiled-estimator and router specialization with the residual frozen;
2. gated residual training with no-harm and correction-scale penalties;
3. lower-rate joint fine-tuning.

Validation ran every 100 steps on fixed ID and OOD episode banks. A checkpoint
was eligible only if it had finite outputs, did not materially harm the
compiled estimator, respected the regime false-positive constraint, retained
expert utilization, and kept false answers on unsupported or nonidentified
queries at zero. Eligible checkpoints were ranked by OOD worst-group MAE, ID
MAE, and router regret.

The selected checkpoint is step 100. Later checkpoints reduced the composite
score and increased split power, but violated the raw stationary-nonlinear
false-positive constraint. The selector therefore retained the earlier safe
checkpoint. A regime evidence threshold was subsequently fit using only the
calibration bank; it achieved 0.410 recall and 0.085 false-positive rate within
that bank.

At the selected checkpoint:

| Development metric | ID | OOD |
| --- | ---: | ---: |
| Identified/supported effect MAE | 0.164 | 0.419 |
| Worst-group effect MAE | 0.273 | 1.357 |
| Router mean regret | 0.103 | 0.300 |
| Residual harm rate | 0.000 | 0.000 |
| False-answer rate | 0.000 | 0.000 |
| Active estimator experts | 6 / 6 | 6 / 6 |

The OOD aggregate is dominated by the nonmonotone interference family (MAE
1.357). OOD hinge, multiplicative, saturation, Erdős–Rényi, and small-world
MAEs were 0.157, 0.172, 0.307, 0.163, and 0.361, respectively. This large
prior-shift gap is retained as a central limitation.

## Final effect results

Primary effect metrics include only supported and identified queries: static
linear, nonlinear, and randomized ATE; and network linear, threshold, and null
interference. Rejected stress queries are evaluated separately.

| Method | Pooled MAE | 90% coverage | Mean width |
| --- | ---: | ---: | ---: |
| **CWFM-Revised** | **0.166** | 0.906 | 0.793 |
| CWFM-Compiled | **0.166** | 0.928 | 0.780 |
| Interaction g-computation | 0.167 | 0.900 | 0.789 |
| Linear g-computation | 0.169 | 0.928 | 0.804 |
| Piecewise exposure-response | 0.193 | 0.911 | 0.772 |
| Spline g-computation | 0.218 | 0.878 | 0.817 |
| Random forest g-computation | 0.329 | 0.911 | 1.337 |

The compiled and revised point estimates are equal to numerical precision.
The calibrated intervals differ because CWFM-Revised uses normalized,
estimator/disagreement-stratified conformal scores, while CWFM-Compiled uses
absolute residual calibration.

### By task

| Task | Method | MAE | Coverage | Width |
| --- | --- | ---: | ---: | ---: |
| Static ATE | CWFM-Revised | **0.169** | 0.944 | 0.876 |
| Static ATE | Linear | 0.173 | 0.922 | 0.666 |
| Static ATE | Interaction | 0.169 | 0.867 | 0.658 |
| Static ATE | Spline | **0.165** | 0.911 | 0.696 |
| Network interference | CWFM-Revised | **0.164** | 0.867 | 0.710 |
| Network interference | Linear | 0.166 | 0.933 | 0.942 |
| Network interference | Interaction | 0.165 | 0.933 | 0.920 |
| Network interference | Piecewise | 0.193 | 0.911 | 0.772 |

### By scenario

| Scenario | CWFM-Revised | Linear | Interaction | Spline / piecewise | Random forest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static linear | 0.168 | **0.160** | 0.189 | 0.181 | 0.412 |
| Static nonlinear | 0.190 | 0.218 | **0.163** | 0.169 | 0.502 |
| Static randomized | 0.150 | **0.141** | 0.154 | 0.145 | 0.449 |
| Network linear | 0.107 | **0.097** | 0.104 | 0.201 | 0.207 |
| Network threshold | 0.283 | 0.311 | 0.312 | **0.173** | 0.351 |
| Network null | 0.100 | 0.090 | 0.079 | 0.206 | 0.055 |

The expert library provides useful headroom, especially the explicit piecewise
expert in the threshold scenario, but the selected router does not exploit all
of it. Its weights vary modestly rather than making decisive
scenario-dependent selections. This is reflected in nonzero router regret and
is a target for the next revision.

## Safety and abstention

CWFM-Revised answered all 180 supported, identified effect queries and rejected
all 120 nonidentified or poor-support queries. Its false-answer rate was
therefore zero.

The safety result is formal rather than learned: declared nonidentification and
poor support are handled by deterministic logic, while the learned heads
provide empirical support and prior-mismatch diagnostics. This avoids
presenting design-token recognition as discovery of hidden confounding.

Always-answer baselines still return numbers in stress cells. Those values are
logged but are not included in primary MAE comparisons.

## Regime results

The regime detector uses a calibration-bank threshold over candidate
variable–quantile evidence, then localizes the variable and threshold
conditionally.

| Scenario | CWFM exact | CWFM split rate | MOB-like exact | MOB-like split rate |
| --- | ---: | ---: | ---: | ---: |
| Strong linear split | **0.633** | 0.633 | 0.800 | 0.833 |
| Weak split | 0.000 | 0.000 | 0.033 | 0.033 |
| Stationary nonlinear | **0.833** | 0.167 | 0.000 | 1.000 |
| Complete null | **1.000** | 0.000 | **1.000** | 0.000 |
| **Pooled** | **0.617** | — | 0.458 | — |

Relative to Phase I, strong-split exact recovery rose from 0.067 to 0.633,
stationary-nonlinear exact rejection fell slightly from 0.967 to 0.833, null
specificity stayed perfect, and pooled exact accuracy rose from 0.508 to 0.617.
Weak changes of magnitude 0.18–0.38 at \(n=96\) remain below reliable power.

## Residual, router, and world diagnostics

The do-no-harm initialization worked:

- mean residual gate: 0.0067;
- mean absolute final correction: below \(3\times10^{-8}\) in raw units;
- final-versus-compiled MAE difference: below \(10^{-9}\);
- residual harm rate at selected ID and OOD development checkpoints: 0.

This means the run selected a safe compiled estimator, not a useful neural
correction. It avoids the Phase-I failure where a noisy direct decoder was
forced into every estimate, but it leaves residual headroom unrealized.

All six estimator experts remained active, so there was no dead-expert
collapse. Routing was nevertheless diffuse. For example, average weights in
network threshold episodes were approximately 0.217 linear, 0.138 orthogonal,
0.124 spline, 0.236 interaction, 0.145 piecewise, and 0.140 null, despite the
piecewise expert having the lowest scenario MAE. Better risk prediction—not
more expert capacity—is the next router priority.

The six structural world particles also remained effectively uniform:
entropy was 1.79176, approximately \(\log 6\), and effective particle count was
6.0 with negligible episode-to-episode variation. The variance formula and
mixture likelihood are repaired and the worlds decode mechanisms and graphs,
but the selected checkpoint does not demonstrate a useful posterior over
worlds. This failure is reported rather than hidden behind aggregate coverage.

## Robustness probes

At the selected checkpoint:

| Probe | Absolute effect change |
| --- | ---: |
| Row permutation | \(1.2\times10^{-7}\) |
| Affine outcome rescaling and inversion | \(4.9\times10^{-8}\) |
| Irrelevant-variable insertion | 0.039 |
| Role-preserving variable permutation | 0.170 |

Row and affine invariance are effectively exact. Irrelevant-variable
sensitivity is moderate. Variable-order equivariance remains poor even though
training randomizes role-preserving order, indicating residual positional
dependence through the formal-query encoding or finite-sample expert fits.

## Reproducibility

Run:

```bash
python -m unittest discover -s tests -v
python -m cwfm.train \
  --steps 1600 \
  --batch-size 24 \
  --validation-interval 100 \
  --output artifacts/cwfm/checkpoint.pt
python -m cwfm.experiment \
  --checkpoint artifacts/cwfm/checkpoint.pt \
  --output-dir artifacts/cwfm/results \
  --seeds 30 \
  --calibration-episodes 360
```

Recorded artifacts:

```text
artifacts/cwfm/
├── checkpoint.pt
├── checkpoint.history.json
├── checkpoint_manifest.json
├── train_metrics.jsonl
├── validation_metrics.jsonl
├── validation_episode_diagnostics.jsonl
└── results/
    ├── per_seed.csv
    ├── diagnostics.csv
    ├── summary.csv
    └── manifest.json
```

The checkpoint SHA-256 is
`8c3fa941e7bb515f1276281bd339e3032e40fc4247edccfea076067f932bb856`.
The experiment manifest records source hashes, package versions, seed ranges,
calibration values, expert-by-scenario weights, and paired bootstrap
comparisons.

## Conclusion

The main architectural correction succeeded: replacing the fixed hybrid with
a query-correct, risk-routed compiled anchor and a gated no-harm residual
reduced pooled effect MAE from 0.271 to 0.166 while preserving selective
answering and approximately nominal coverage. Strong regime recovery also
improved substantially without sacrificing complete-null specificity.

Three limitations remain decisive:

1. the neural residual contributes no measurable point-estimation gain;
2. estimator routing is diffuse and leaves piecewise-expert gains unused;
3. world particles remain uniform and OOD nonmonotone interference fails.

The next experiment should therefore improve risk prediction and structural
particle specialization, and explicitly fix variable-order equivariance. It
should not increase backbone size until those failures improve on untouched
OOD families.
