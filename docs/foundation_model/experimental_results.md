# CWFM Phase-I experimental results

Evaluation date: 2026-07-28

## Executive summary

The first executable Causal World Foundation Model (CWFM) experiment is
complete. The run pretrained one shared PyTorch checkpoint on 38,400 synthetic
episodes, calibrated intervals on 360 disjoint episodes, and evaluated 14
declared scenarios with 30 held-out seeds each (420 evaluation episodes).
Every method saw the same episode for a scenario and seed.

The result is mixed:

- **CWFM-Calibrated was safer than estimators that always answered.** It rejected
  all 60 queries whose supplied assumptions declared non-identification and
  all 60 queries whose requested contrast lacked support. It answered all 300
  supported, identified effect and regime queries.
- **Simple specialists remained best when their assumptions were correct.**
  Across the six supported and identified effect scenarios, linear
  g-computation had mean absolute error (MAE) 0.169, versus 0.271 for
  CWFM-Calibrated.
- **CWFM-Calibrated improved on the flexible random-forest baseline.** Its pooled
  effect MAE was 0.271 versus 0.329. The paired MAE difference was -0.058
  (95% paired bootstrap interval [-0.094, -0.025]).
- **The direct neural path was not competitive by itself.** CWFM-Direct MAE was
  0.424. Combining it with the compiled classical path reduced MAE to 0.271.
- **Regime behavior showed a safety–power tradeoff.** CWFM almost never made a
  false split for stationary nonlinear or null data, but recovered only 2 of
  30 strong splits and none of the weak splits. The MOB-like specialist found
  24 of 30 strong split variables but falsely split every stationary nonlinear
  episode.
- **External calibration was useful but not uniformly nominal.** The
  90%-target CWFM-Calibrated intervals covered 91.7% of supported identified effect
  queries overall. Scenario coverage ranged from 80.0% to 100%.

These are engineering results from 30 seeds per cell, not publication-sized
Monte Carlo evidence. The run establishes that the code and safety interface
work, and it falsifies any claim that this compact checkpoint already
outperforms well-specified classical specialists.

## 1. What was implemented

The executable model follows the initial scope in
[`initial_architecture.md`](initial_architecture.md):

| Component | Executable implementation |
| --- | --- |
| Episode interface | Values, missingness, variable roles, design/assumption status, formal task, and relation matrix |
| Typed encoder | Scalar/missingness projection plus role, task, and design embeddings |
| Axial backbone | Three alternating variable-axis and sample-axis attention blocks |
| Network route | Normalized relation-matrix messages at every axial block |
| Mechanism route | Four feed-forward experts with learned top-2 sparse routing |
| Causal worlds | Eight learned world queries with posterior weights and particle effect distributions |
| Structure | Directed graph logits and a query-specific regime-variable/no-split head |
| Direct estimation | Typed differentiable regression anchor plus learned world-particle correction |
| Compiled estimation | Linear g-computation or a flexible classical estimator selected by the mechanism route |
| Hybrid estimation | Equal-weight direct/compiled combination after learned classical routing |
| Reliability | Separate identification and support probabilities; point answer suppressed if either is below 0.5 |
| Calibration | Split-conformal absolute-residual radii by task and method |

The checkpoint contains 778,584 trainable parameters. Its runnable
configuration uses hidden dimension 96, three axial blocks, four experts,
top-2 routing, four attention heads, and eight particles. This is a compact
reference model, not the proposed 384-dimensional production configuration.

The following proposed components are **not** implemented in this experiment:
temporal decoding, latent-variable slots, PAG/ADMG enumeration, symbolic
do-calculus, front-door/IV/longitudinal identification, sensitivity bounds,
counterfactual distributions, process-event adapters, and real-data
fine-tuning.

## 2. Experimental protocol

### 2.1 Data separation

Three seed ranges were disjoint:

| Plane | Seed/configuration | Use |
| --- | --- | --- |
| Pretraining | seed 1729; 1,600 batches × 24 episodes | Gradient updates |
| Calibration | begins at 19,000,000; 360 episodes | 90% residual radii only |
| Final evaluation | begins at 20,000,000; 30 seeds × 14 scenarios | Reported metrics only |

Oracle targets, graphs, identification labels, and support labels are produced
by the simulator and passed only to the training loss or evaluator. The model
input contains observed values, roles, relations, task, and the
design/assumption declaration. Final truth is never included in the visible
value tensor.

### 2.2 Scenarios

The evaluation contains:

- static ATE: linear, nonlinear outcome, randomized, poor overlap, and
  declared hidden confounding;
- observed regimes: strong linear split, weak split, stationary nonlinearity,
  and null;
- network interference: linear spillover, threshold spillover, null, poor
  exposure support, and declared hidden confounding.

Each episode has 96 rows. Static tasks use six covariates. Network tasks use
eight independent clusters and an observed within-cluster relation matrix.
The spillover query is the outcome change from neighbor exposure 0.25 to 0.75.

### 2.3 Compared systems

- **CWFM-Direct:** the amortized direct effect or structure decoder.
- **CWFM-Calibrated:** average of the direct decoder and a compiled classical
  estimate; the learned mechanism router mixes linear and flexible compiled
  estimators. Identification and support gates can suppress the answer, and
  disjoint split-conformal residuals calibrate its interval.
- **Linear g-computation:** ordinary least-squares outcome regression; the
  network version includes treatment, exposure, and their interaction.
- **Random forest g-computation:** 120-tree response model evaluated under the
  requested treatment or exposure contrast.
- **AIPW:** logistic propensity model plus random-forest outcome regression
  for static ATE only.
- **MOB-like:** a BIC-penalized exhaustive one-split parameter-instability
  scan for the regime task.

Hyperparameters and method names were fixed in code before the corrected final
run. A first attempted run was discarded after an audit found that task and
scenario selection reused the same pseudorandom draw, confounding the
pretraining distribution. The corrected generator independently selects task
and scenario; its balance is covered by tests and the checkpoint was trained
from scratch.

### 2.4 Metrics

Effect tasks report MAE, interval coverage, and answer rate. Regime tasks
report exact split-variable/no-split accuracy and the frequency of returning
any split. Safety reports rejection of declared nonidentified or unsupported
queries. Paired comparisons resample the scenario–seed MAE difference 2,000
times.

## 3. Main effect-estimation results

The table below excludes poor-support and hidden-confounding cells because a
safe system should not return a point estimand there.

| Task | Method | MAE | 90%-target coverage |
| --- | --- | ---: | ---: |
| Static ATE | Linear g-computation | **0.173** | 0.811 |
| Static ATE | AIPW | 0.201 | 0.878 |
| Static ATE | CWFM-Calibrated | 0.324 | 0.900 |
| Static ATE | Random forest g-computation | 0.454 | 0.944 |
| Static ATE | CWFM-Direct | 0.563 | 0.633 |
| Network interference | Linear g-computation | **0.166** | 0.944 |
| Network interference | Random forest g-computation | 0.204 | 0.911 |
| Network interference | CWFM-Calibrated | 0.218 | 0.933 |
| Network interference | CWFM-Direct | 0.285 | 0.911 |
| **All supported identified effect cells** | **Linear g-computation** | **0.169** | 0.878 |
|  | AIPW (static only) | 0.201 | 0.878 |
|  | CWFM-Calibrated | 0.271 | 0.917 |
|  | Random forest g-computation | 0.329 | 0.928 |
|  | CWFM-Direct | 0.424 | 0.772 |

The linear specialist's advantage over CWFM-Calibrated was 0.102 MAE (paired 95%
bootstrap interval [0.074, 0.133]). AIPW's advantage on its static cells was
0.123 [0.077, 0.169]. CWFM-Calibrated's advantage over random-forest
g-computation was 0.058 [0.025, 0.094].

### Scenario detail

| Scenario | CWFM-Direct MAE | CWFM-Calibrated MAE | Linear MAE | RF MAE | AIPW MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static linear | 0.544 | 0.314 | **0.160** | 0.412 | 0.188 |
| Static nonlinear outcome | 0.582 | 0.348 | **0.218** | 0.502 | 0.242 |
| Static randomized | 0.562 | 0.311 | **0.141** | 0.449 | 0.173 |
| Network linear spillover | 0.233 | 0.156 | **0.097** | 0.207 | — |
| Network threshold spillover | 0.568 | 0.442 | **0.311** | 0.351 | — |
| Network null | 0.055 | 0.057 | 0.090 | **0.055** | — |

The threshold result contradicts the forecast that mechanism routing would
make CWFM clearly superior under nonlinearity. The random forest improved
relative to the linear model but did not beat it in this small-sample cell;
CWFM's learned route assigned only 0.336 average probability to its flexible
expert for both linear and threshold network data. Mechanism discrimination
therefore remains inadequate.

## 4. Identification and support safety

The formal design token states whether randomization/adjustment assumptions
identify the query or whether possible hidden confounding is explicitly
declared. This experiment tests compliance with supplied assumptions; it does
not claim to discover arbitrary hidden confounding from observational values.

| Scenario class | Queries | CWFM-Calibrated answer rate | Always-answer baselines |
| --- | ---: | ---: | ---: |
| Supported and identified effects | 180 | 1.000 | 1.000 |
| Identified regime queries | 120 | 1.000 | 1.000 |
| Declared hidden confounding | 60 | **0.000** | 1.000 |
| Declared poor support | 60 | **0.000** | 1.000 |

Mean gate probabilities were:

| Scenario | Identification probability | Support probability |
| --- | ---: | ---: |
| Static identified | 0.993–0.995 | 0.993–0.997 |
| Static declared hidden confounding | 0.003 | 0.997 |
| Static poor overlap | 0.997 | 0.004 |
| Network identified | 0.997 | 0.997 |
| Network declared hidden confounding | 0.003 | 0.998 |
| Network poor exposure support | 0.998 | 0.004 |

This clean separation is expected because the gate receives formal
design/assumption metadata and the same declaration types occur in
pretraining. It demonstrates interface compliance, not empirical detection of
unrecorded violations.

## 5. Regime results

| Scenario | Method | Exact variable/no-split accuracy | Any-split rate |
| --- | --- | ---: | ---: |
| Strong linear split | CWFM | 0.067 | 0.067 |
|  | MOB-like | **0.800** | 0.833 |
| Weak split | CWFM | 0.000 | 0.000 |
|  | MOB-like | **0.033** | 0.033 |
| Stationary nonlinear | CWFM | **0.967** | 0.033 |
|  | MOB-like | 0.000 | 1.000 |
| Null | CWFM | 1.000 | 0.000 |
|  | MOB-like | 1.000 | 0.000 |

Across all four cells, CWFM exact accuracy was 0.508 and MOB-like accuracy was
0.458, but this pooled number is misleading: CWFM obtained its score by nearly
always selecting no split. The model controls the specified stationary
nonlinearity failure but lacks useful split power. The correct engineering
conclusion is to retain the explicit nonlinear-vs-regime contrast while
redesigning the set-valued regime decoder and increasing regime-specific
training, not to claim an overall win.

## 6. Calibration and uncertainty

Separate calibration episodes produced task/method absolute-residual radii.
CWFM-Calibrated's radii were 0.679 for static ATE and 0.708 for network
interference. Across supported identified effects, its 90%-target interval
coverage was 0.917:

- static linear: 0.933;
- static nonlinear: 0.833;
- static randomized: 0.933;
- network linear: 1.000;
- network threshold: 0.800;
- network null: 1.000.

CWFM-Direct's raw particle intervals covered only 0.772 overall, including
0.600–0.667 in the three static cells. This supports the architecture
document's warning that neural posterior scales should not be treated as
frequentist confidence intervals without correction.

## 7. Runtime and reproducibility

The recorded environment used Python 3.12.3, PyTorch 2.13.0+cu130, NumPy
2.5.1, and pandas 3.0.5 on one CUDA device.

| Stage | Runtime |
| --- | ---: |
| Pretraining, 1,600 × 24 episodes | 95.4 s |
| Calibration and final evaluation | 59.4 s |

The checkpoint SHA-256 is
`a650686f05787e9d65126758b54283c0de77f70732e314cef874a792f8412017`.
The authoritative artifacts are:

```text
artifacts/cwfm/
├── checkpoint.pt
├── checkpoint.history.json
└── results/
    ├── diagnostics.csv
    ├── manifest.json
    ├── per_seed.csv
    └── summary.csv
```

Reproduction commands are in the repository `README.md`. The manifest records
the exact seed ranges, checkpoint hash, versions, calibration radii, paired
comparisons, and runtimes.

## 8. Conclusions against the original hypotheses

| Hypothesis | Phase-I evidence |
| --- | --- |
| H1: non-inferiority in simple settings | **Not supported.** CWFM-Calibrated was materially worse than linear specialists. |
| H2: superiority under heterogeneity/misspecification | **Not supported in these cells.** It beat RF overall but not the best method, and failed to route threshold interference effectively. |
| H3: safer under identification/support failure | **Supported for explicitly supplied design declarations.** All 120 unsafe queries were rejected with no rejected supported query. |
| H4: parameter-shift generalization | **Not tested broadly.** Final numerical seeds were held out, but no continuous or mechanism-family OOD sweep was run. |
| H5: reduced downstream error propagation | **Not tested.** Latent slots and latent-graph tasks are outside this implementation. |
| H6: amortized reuse | **Partially supported operationally.** Neural inference is shared across tasks, but total comparison runtime was dominated by classical fits; no formal latency benchmark was recorded per method. |

The compact implementation validates the hybrid system boundary—shared neural
representation, explicit causal-world uncertainty, compiled estimators, and
safety gates—but does not yet validate a performance advantage over classical
specialists. The next highest-value work is:

1. replace the single-variable regime classifier with a calibrated set-valued
   multi-break decoder and explicitly balance split/no-split power;
2. supervise mechanism routing with held-out risk rather than only mechanism
   family labels;
3. add cross-fitted one-step correction instead of an equal direct/compiled
   average;
4. run the 200-seed expanded replication and continuous stress surfaces from
   `experimental_design.md`;
5. add temporal and latent modules only after the three implemented task
   families meet their specialist non-inferiority targets.
