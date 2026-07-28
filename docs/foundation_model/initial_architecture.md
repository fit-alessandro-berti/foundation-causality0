# Proposed architecture: a query-conditioned causal world foundation model

A credible design should **not** be a single transformer that maps a dataset directly to one graph or one treatment-effect estimate. That would reproduce several weaknesses seen in the ZIP experiments: predictive shortcuts, confident answers under hidden confounding, forced extrapolation under poor overlap, and error propagation from latent-variable preprocessing.

A better design is a hybrid system with four defining properties:

1. **One shared pretrained backbone, but several modality-specific routes.**
2. **An approximate posterior over multiple causal worlds rather than one estimated graph.**
3. **A formal identification layer between structure learning and effect estimation.**
4. **Explicit support, calibration, prior-mismatch, and abstention modules.**

Recent causal foundation-model work already demonstrates individual components of this proposal. Arrow uses variable-level representations and a skeleton–order factorization that guarantees acyclic graph outputs; TabCausal and CDFM use alternating attention across observations and variables; CDFM also uses missingness-aware encoding and masked-data reconstruction; and recent graph-conditioned causal foundation models inject partial causal knowledge through attention biases or graph encoders. ([arXiv][1])

I will call the proposed architecture the **Causal World Foundation Model**, or **CWFM**.

```text
                            FORMAL CAUSAL QUERY
                  treatment, outcome, intervention, target
                    population, time, exposure, estimand
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAUSAL EPISODE INTERFACE                        │
│                                                                         │
│  Values │ missingness │ interventions │ schema │ time │ network │       │
│  clusters/environments │ partial graph knowledge │ design information   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TYPED AXIAL CAUSAL ENCODER                           │
│                                                                         │
│  Variable-axis attention                                                │
│  Time-axis attention                                                    │
│  Unit/sample-axis attention                                             │
│  Relation-aware network messages                                        │
│  Sparse mixture-of-mechanism experts                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  POSTERIOR OVER CAUSAL WORLDS                           │
│                                                                         │
│  Graph particles │ mechanism particles │ latent groups │ regimes │      │
│  exposure mappings │ assignment/selection mechanisms │ world weights    │
└─────────────────────────────────────────────────────────────────────────┘
                    │                                  │
                    ▼                                  ▼
       STRUCTURAL DECODERS                    IDENTIFICATION ENGINE
 graph / PAG / latent groups /          adjustment / front door / IV /
 regimes / temporal lags / exposure      longitudinal / interference /
 mappings / measurement reliability       partial identification
                    │                                  │
                    └──────────────┬───────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      DUAL ESTIMATION PATH    │
                    │                              │
                    │  Direct do-distribution path │
                    │  Compiled estimator path     │
                    └──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RELIABILITY LAYER                               │
│                                                                         │
│ support │ graph ambiguity │ prior mismatch │ sensitivity bounds │       │
│ posterior checks │ self-compatibility │ calibration │ abstention        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    STRUCTURED CAUSAL ANSWER OR REFUSAL
```

## 1. The basic input unit should be a causal episode

The model should receive more than a rectangular table. Its input should be a structured episode

[
\mathcal E=
\left(
X,,
M,,
I,,
S,,
C,,
T,,
W,,
K,,
q
\right),
]

where:

| Component | Meaning                                                                            |
| --------- | ---------------------------------------------------------------------------------- |
| (X)       | Observed values                                                                    |
| (M)       | Missingness and censoring indicators                                               |
| (I)       | Intervention indicators and intervention values, when available                    |
| (S)       | Variable schema: type, unit, admissible values, and measurement role               |
| (C)       | Cluster, site, subject, policy, or environment membership                          |
| (T)       | Timestamps, order information, and elapsed times                                   |
| (W)       | Observed relation matrix between units, such as a social or organizational network |
| (K)       | Structured background knowledge and partial graph constraints                      |
| (q)       | The formal causal query                                                            |

A convenient common representation is

[
X\in\mathbb R^{U\times T\times P},
]

with (U) units, (T) time points, and (P) variables.

* For ordinary i.i.d. tabular data, (T=1) and (U=N).
* For one multivariate time series, (U=1).
* For panel data, both (U>1) and (T>1).
* For network interference, (W) connects the (U) units.
* For multiple hospitals, factories, or process variants, (C) identifies environments.

This common tensor permits substantial parameter sharing without pretending that static rows, temporal observations, and network nodes are exchangeable in the same way.

### The query must be formal, not only natural language

The core model should receive a causal query language rather than a vague prompt. For example:

```yaml
claim_type: interventional
treatment: A
outcome: Y_1
intervention:
  type: hard
  contrast: [0, 1]
conditioning_variables: [X_1, X_2]
target_population: all_eligible_units
estimand: ATE
time_horizon: null
interference:
  enabled: false
sensitivity_model: null
```

For the network benchmark, a query could instead specify:

```yaml
claim_type: interventional
treatment: A
outcome: Y_2
estimand: average_spillover_effect
own_treatment: 0
exposure_mapping: one_hop_treated_fraction
exposure_contrast: [0.25, 0.75]
target_population: common_support
```

A language model may be used as a front end to translate a scientific question into this representation, but the causal core should operate on the structured query. Otherwise, apparently minor wording differences could silently change the estimand.

The query also contains an explicit **claim type**:

[
c_q\in
{
\text{association},
\text{predictive/Granger},
\text{interventional},
\text{counterfactual}
}.
]

This prevents the system from relabelling a conditional-association graph or a VARX relation as intervention-level causality, a distinction emphasized by the ZIP report’s temporal and latent-graph results.

## 2. Typed cell and schema embeddings

For each cell (x_{utj}), the initial representation could be

[
h^{(0)}_{utj}
=============

f_{\text{cell}}
\left(
\widetilde{x}*{utj},
m*{utj},
i_{utj},
a_{utj}
\right)
+
e_{\text{type}(j)}
+
e_{\text{role}(j,q)}
+
e_{\text{environment}(u,t)}
+
e_{\text{time}(t)}.
]

Here:

* (\widetilde{x}_{utj}) is a robustly standardized value;
* (m_{utj}) indicates missingness;
* (i_{utj}) indicates whether the cell was intervened on;
* (a_{utj}) gives the intervention value;
* (e_{\text{type}}) distinguishes continuous, binary, categorical, ordinal, count, and survival variables;
* (e_{\text{role}}) says whether a variable is the queried treatment, outcome, exposure, mediator, instrument candidate, or ordinary observed variable.

Missingness should always be an explicit input channel rather than silently imputed before the model. CDFM similarly represents each value together with a missingness indicator and uses reconstruction of masked values as a self-supervised objective. ([arXiv][2])

There should normally be **no fixed positional embedding for variable index**. Otherwise, the model might treat “column 4” as intrinsically different from “column 7.” Random variable permutations during training should result in correspondingly permuted outputs.

Optional variable descriptions, such as “blood pressure” or “machine temperature,” may be encoded by a separate semantic adapter. They should be treated as soft background information, not as evidence that an edge is causally identified.

## 3. A typed axial causal encoder

The main backbone should alternate attention across the dimensions that have distinct causal meanings.

For one layer,

[
H^{(l,1)}
=========

\operatorname{Attn}_{\text{variable}}
\left(H^{(l)};B_K\right),
]

[
H^{(l,2)}
=========

\operatorname{Attn}*{\text{time}}
\left(H^{(l,1)};M*{\text{temporal}}\right),
]

[
H^{(l,3)}
=========

\operatorname{Attn}_{\text{unit}}
\left(H^{(l,2)};B_W,B_C\right),
]

followed by a mixture-of-experts feed-forward layer.

### Variable-axis attention

At fixed unit and time, variables attend to one another. This captures:

* conditional relationships;
* interactions;
* treatment–outcome–confounder configurations;
* correlated measurements;
* contemporaneous mechanism changes.

Partial causal knowledge (K) can be injected as an attention bias. Known non-ancestral relationships can receive a negative bias, known ancestral relationships a positive bias, and uncertain relationships no bias. Hard constraints should be enforced only when the knowledge is declared trustworthy. Recent work on causal foundation models with partial graphs explores both attention biases and GCN-conditioned adaptive normalization for this purpose. ([arXiv][3])

### Time-axis attention

At fixed unit and variable, temporal attention captures:

* autoregression;
* delayed treatment effects;
* changing mechanisms;
* abrupt breaks;
* gradual drift;
* irregular observation times.

The mask depends on the task:

* change-point analysis may use the full observed historical sequence;
* forecasting and counterfactual trajectory prediction use a causal mask;
* query tokens should not attend to future query outcomes.

Recent temporal causal PFN proposals represent unit–time observations as tokens with relative-time, pre/post-treatment, and elapsed-time encodings, and use restricted cross-attention to prevent leakage across prediction horizons. ([arXiv][4])

### Unit-axis or sample-axis attention

For i.i.d. data, this operation is permutation invariant over rows. For network data, it becomes relation-aware:

[
\operatorname{Attn}_{\text{unit}}
\left(H;B_W\right),
]

where (B_W) biases or restricts messages according to the observed relation matrix.

For clustered experiments, cluster tokens aggregate information at the assignment unit. This is important because uncertainty, cross-fitting, and resampling must preserve clusters rather than treating all rows as independent.

For large (U), full (U^2) attention is unnecessary. Learned summary tokens can cross-attend to observations, similar to Arrow’s use of summary tokens to obtain fixed-dimensional variable representations without full quadratic attention across observations. ([arXiv][1])

## 4. Sparse mixture-of-mechanism experts

A single universal feed-forward network is likely to blur important distinctions between simple and flexible models. The backbone should therefore contain a sparse mixture of mechanism experts:

[
p(x_j\mid \operatorname{pa}_j,w_s)
==================================

\sum_{e=1}^{E}
\rho_{sje},
p_e(x_j\mid \operatorname{pa}_j),
]

where (\rho_{sje}) is the router weight for expert (e), variable (j), and causal-world particle (s).

Initial expert families could include:

| Expert              | Intended mechanisms                               |
| ------------------- | ------------------------------------------------- |
| Linear–Gaussian     | Simple additive effects and VARX relations        |
| Generalized linear  | Binary, count, ordinal, and survival outcomes     |
| Smooth nonlinear    | Splines, additive functions, monotone responses   |
| Threshold/piecewise | Threshold spillovers and abrupt nonlinearities    |
| Interaction         | (A\times G), effect modification, cancellation    |
| Temporal dynamic    | Autoregressive and delayed effects                |
| Network relational  | Degree-sensitive and multi-hop exposure effects   |
| Measurement         | Latent factors, weak measurements, cross-loadings |

The router should have a simplicity prior: when a linear mechanism explains the data adequately, the flexible expert should not automatically dominate. This directly addresses the interference benchmark, where the simple randomized-design estimator performed better for the clean linear case while the flexible model performed better for the threshold response.

The router must also compare structurally different explanations, for example:

[
H_0:\text{one stationary nonlinear mechanism}
]

against

[
H_1:\text{several simpler regime-specific mechanisms}.
]

The misspecified nonlinear experiment in the package produced a false split in 5/5 runs while dramatically improving prediction. The architecture must therefore be trained on matched pairs in which a stationary nonlinear model and a piecewise-linear model have similar predictive performance but different structural truth. Predictive gain alone should never determine the regime decision. These benchmark findings are documented in the [package report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md).

## 5. The central latent object: a posterior over causal worlds

The encoder should not collapse the dataset into one graph. It should construct an approximate posterior represented by (S) weighted world particles:

[
q_\phi(\mathcal W\mid\mathcal E)
\approx
\sum_{s=1}^{S}
\omega_s,
\delta_{\mathcal W_s},
\qquad
\sum_s\omega_s=1.
]

Each world particle contains

[
\mathcal W_s=
\left(
G_s,,
\mathcal M_s,,
Z_s,,
R_s,,
F_s,,
A_s
\right),
]

where:

* (G_s) is a graph or partial graph;
* (\mathcal M_s) contains mechanism-family and noise information;
* (Z_s) describes latent-variable assignments and measurement models;
* (R_s) describes regimes or continuous drift;
* (F_s) describes the network exposure mapping;
* (A_s) describes assignment, selection, missingness, and observation mechanisms.

In an initial implementation, (S=8) to (32) particles would be reasonable. Each world token cross-attends to variable, environment, temporal, and relation summaries. A diversity regularizer prevents all particles from collapsing to the same hypothesis.

This representation separates several kinds of uncertainty that should not be merged into one confidence interval:

1. **Outcome uncertainty:** randomness in (Y) even under a known causal world.
2. **Finite-sample uncertainty:** uncertainty in parameters given limited data.
3. **Structural uncertainty:** different plausible graphs or exposure mappings.
4. **Identification ambiguity:** different causally compatible worlds imply different effects.
5. **Prior mismatch:** the real mechanism may not be adequately represented by the pretraining distribution.

### Static graph decoder

For each particle, a static directed graph can use a skeleton–order factorization:

[
p(G_s\mid\mathcal E)
====================

p(S_s\mid\mathcal E),
p(\pi_s\mid\mathcal E),
]

where (S_s) is an undirected skeleton and (\pi_s) a topological order. Orienting the skeleton according to (\pi_s) guarantees acyclicity, following the central construction used by Arrow. ([arXiv][1])

However, the model should not report a uniquely directed DAG when the data support only an equivalence class. Individual particles may be DAGs, but the aggregated output should be a:

* CPDAG when only Markov equivalence remains;
* PAG when hidden confounding and selection may be present;
* posterior endpoint-probability matrix when uncertainty is substantial.

A separate symmetric head predicts bidirected edges representing possible latent confounding:

[
p!\left(j\leftrightarrow k\mid\mathcal E\right).
]

### Temporal graph decoder

For temporal data, the decoder outputs

[
G^{(0)},G^{(1)},\ldots,G^{(L)},
]

where (G^{(\ell)}_{jk}) represents an effect from variable (j) at lag (\ell) to variable (k).

Positive-lag edges are already ordered by time. Contemporaneous edges require a DAG, cyclic-model, or partial-orientation head. The output must state whether an edge is:

* predictive/Granger;
* structurally directed by temporal ordering;
* interventionally causal under supplied assumptions.

### Soft latent-variable slots

The latent benchmarks argue against hard grouping before graph estimation. The architecture should instead maintain (K_{\max}) latent slots:

[
\ell_1,\ldots,\ell_{K_{\max}},
]

with soft assignment weights

[
P_{jk}=P(\text{observed variable }j
\text{ loads on latent slot }k\mid\mathcal E).
]

Sparse normalization allows most assignments to be zero while permitting cross-loadings. An occupancy head estimates whether each latent slot is active.

Latent scores are then inferred jointly:

[
z_{utk}
=======

\sum_j P_{jk},
g_k(h_{utj}),
]

and the latent graph decoder operates on distributions over (z_{utk}), not on a single reconstructed score matrix.

This directly prevents the Level B-to-Level C error propagation seen in the latent-graph benchmark. The model should still expose oracle-style evaluation modes corresponding to:

* true latent scores;
* true groups with estimated scores;
* fully estimated groups and scores.

That preserves the diagnostic value of the package’s Levels A, B, and C.

### Regime and drift decoder

A fixed “one breakpoint” head is insufficient. The temporal decoder should contain a set of regime tokens

[
r_1,\ldots,r_{R_{\max}}
]

and output:

* a set of abrupt transition points;
* transition-window start and end points;
* continuous mixture weights for gradual drift;
* a change-type distribution.

For time (t), a gradual process can be represented as

[
\mathcal M_t
============

\sum_{r=1}^{R_{\max}}
\alpha_r(t)\mathcal M_r,
\qquad
\sum_r\alpha_r(t)=1.
]

The change-type head should distinguish at least:

[
\Delta\text{slope},\quad
\Delta\text{intercept},\quad
\Delta\text{variance},\quad
\Delta A,\quad
\Delta B,\quad
\Delta P(X),\quad
\text{gradual drift}.
]

The intercept-only VARX scenario should therefore be trained as a hard negative for a (B)-change. This addresses the package result in which an intercept-only break was falsely attributed to (B) in 2/5 runs.

### Exposure-mapping decoder

The unit relation matrix (W) is not the same object as the causal-variable graph (G). A separate network module should generate candidate exposure representations:

[
g_i^{(1)},\ldots,g_i^{(K_F)},
]

such as:

* one-hop treated fraction;
* degree-normalized weighted exposure;
* distance-two exposure;
* thresholded exposure;
* community-level saturation;
* learned relational exposure.

The model then represents uncertainty over exposure mappings:

[
P(F=k\mid W,X,A,K).
]

There should be two operational modes:

* **Declared-map mode:** the scientist supplies the exposure mapping, and the model estimates the corresponding causal estimand.
* **Exploratory-map mode:** the model compares mappings for sensitivity analysis but does not silently redefine the primary estimand.

Choosing an exposure map using the outcomes and then presenting the selected effect as pre-specified would be scientifically problematic. The exploratory mode must therefore be reported as model selection or sensitivity analysis.

## 6. A hybrid identification engine

This is the most important part of the architecture.

A learned model may recognize patterns associated with confounding or identification, but identifiability is a property of a causal model plus assumptions—not a property that can be certified from observational patterns alone. A recent temporal causal PFN explicitly notes that its learned identifiability output is a heuristic correlated with labels in the training prior rather than a formal proof. ([arXiv][4])

CWFM should therefore include a deterministic identification engine:

[
\operatorname{Identify}
\left(
G_s,K,q
\right)
\longrightarrow
\left(
I_s,\Psi_s
\right),
]

where (I_s) is the status and (\Psi_s) is an observable-data functional when available.

Possible statuses are:

[
I_s\in
{
\text{identified},
\text{partially identified},
\text{not identified},
\text{unsupported},
\text{contradictory assumptions},
\text{unknown}
}.
]

The engine may include:

* back-door and adjustment-set enumeration;
* front-door identification;
* instrumental-variable templates;
* general identification procedures for ADMGs;
* sequential g-formula or longitudinal identification;
* randomized-design and cluster-randomized interference identification;
* mediation and policy-effect functionals;
* partial-identification and sensitivity modules.

CausalFM emphasizes the distinction between identifying a causal quantity and statistically estimating an already identified functional. It argues that a PFN should not be expected to repair a fundamentally unidentified setup merely through predictive learning. Do-PFN, by contrast, allows priors containing unidentified structures and seeks to express the resulting ambiguity through broad predictive distributions. The proposed architecture accommodates both views: the formal layer determines what can be claimed, while the world posterior can still describe prior-dependent ranges when point identification fails. ([arXiv][5])

### Identification under graph uncertainty

For each high-probability world,

[
\Theta_q^{(s)}
==============

\begin{cases}
{\theta_s}, & I_s=\text{identified},[4pt]
[\theta^-_s,\theta^+_s], &
I_s=\text{partially identified},[4pt]
\text{undefined}, &
I_s=\text{unsupported or contradictory}.
\end{cases}
]

The final system should distinguish:

* **robustly identified:** identified under essentially all compatible high-probability worlds;
* **assumption-dependent identification:** identified under some structural assumptions but not others;
* **partially identified:** only a set or interval is justified;
* **unsupported:** the requested treatment or exposure contrast lacks data support.

The model must not discard nonidentified particles and average only the convenient identified ones.

## 7. Dual estimation paths

Where an estimand is identified, CWFM should use two complementary estimation paths.

### Path A: direct interventional-distribution decoder

The first path predicts the complete interventional outcome distribution:

[
p_\theta
\left(
Y^{do(a)}
\mid x,\mathcal E,\mathcal W_s,q
\right).
]

The desired effect is then computed as a functional of this distribution. For example,

[
\tau(x)
=======

## \mathbb E[Y^{do(1)}\mid x]

\mathbb E[Y^{do(0)}\mid x].
]

Predicting the distribution instead of only the effect permits:

* nonlinear contrasts;
* quantile effects;
* policy values;
* dose–response functions;
* trajectories;
* probability-of-benefit queries;
* coherent outcome uncertainty.

Do-PFN and CausalFM both use synthetic SCMs and interventional targets to train PFN-style models for causal effect or interventional-distribution prediction. ([arXiv][6])

Outcome-type adapters provide appropriate likelihoods:

| Outcome type | Suggested decoder                                      |
| ------------ | ------------------------------------------------------ |
| Continuous   | Discretized density or mixture-density head            |
| Binary       | Bernoulli                                              |
| Count        | Poisson or negative-binomial mixture                   |
| Ordinal      | Ordered categorical                                    |
| Survival     | Discrete hazard or parametric survival mixture         |
| Multivariate | Factorized plus residual-copula or low-rank joint head |

For temporal queries, query tokens cross-attend to observed context but not to other unknown query outcomes. Under a proposed treatment sequence, the decoder returns a distribution over the complete future trajectory.

### Path B: compiled observable-functional estimator

The identification engine may return an observable functional. For a back-door ATE, for example,

[
\theta
======

\mathbb E_X
\left[
m_1(X)-m_0(X)
\right],
]

where (m_a(x)=\mathbb E[Y\mid A=a,X=x]).

The foundation model predicts the required nuisance quantities, such as:

* outcome regressions;
* propensities;
* mediator distributions;
* instrument–treatment relationships;
* censoring probabilities;
* network dose–response functions.

A deterministic statistical layer then evaluates the identified functional. When influence-function estimators exist, it can compute

[
\widehat{\theta}_{\mathrm{IF}}
==============================

\frac{1}{N}
\sum_{i=1}^{N}
\phi
\left(
O_i;\widehat{\eta}^{-k(i)}
\right),
]

where (\widehat{\eta}^{-k(i)}) denotes nuisance functions estimated without the fold containing observation (i).

The transformer can implement this through **fold-aware attention masks**:

* a query unit cannot attend to its own outcome when constructing nuisance estimates;
* a network cluster cannot attend to outcomes from the same held-out cluster;
* temporal folds use blocked rather than random splits.

This gives the system a route to classical cross-fitting and influence-function inference without fitting new models from scratch for every dataset.

### Agreement as a diagnostic

The model obtains two estimates:

[
\widehat{\theta}*{\text{direct}}
\quad\text{and}\quad
\widehat{\theta}*{\text{compiled}}.
]

Their disagreement,

[
d_q=
\left|
\widehat{\theta}_{\text{direct}}
--------------------------------

\widehat{\theta}_{\text{compiled}}
\right|,
]

is a useful misspecification signal. Strong disagreement may indicate:

* prior mismatch;
* incorrect graph assumptions;
* poor nuisance estimation;
* unsupported extrapolation;
* an exposure-mapping problem;
* insufficient sample size.

Neither path should automatically dominate. A learned aggregation head can combine them, but only after conditioning on support, identification, calibration, and cross-fitted risk.

## 8. Reliability and abstention architecture

A causal foundation model needs more than a confidence interval. It should include several separate diagnostic heads.

| Diagnostic                     | Question answered                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------- |
| Query-specific support         | Are comparable observations available for the requested intervention?             |
| Structural entropy             | Do plausible world particles disagree about the graph or mechanism?               |
| Identification status          | Is a point effect justified under the supplied assumptions?                       |
| Prior-support/OOD              | Does this dataset resemble mechanisms covered during pretraining?                 |
| Posterior predictive check     | Can the inferred world reproduce relevant observed distributions?                 |
| Negative-control compatibility | Does the model find effects where a null should hold?                             |
| Self-compatibility             | Are structures inferred from variable subsets mutually compatible?                |
| Stability                      | Does the result survive row, cluster, time-block, and variable perturbations?     |
| Calibration                    | Do uncertainty statements have empirical coverage on applicable calibration data? |

### Query-specific overlap

Overlap must be evaluated for the exact target, not globally.

For ordinary treatment effects, the support head estimates

[
s(x,a)
======

P(A=a\mid X\approx x).
]

For interference, it estimates joint support:

[
s(x,a,g)
========

P(A=a,G\approx g\mid X\approx x).
]

For longitudinal treatment, it assesses support for the proposed treatment sequence conditional on observed history.

The model may still produce a prior-predictive extrapolation outside support, but it must label that output as **prior dominated**, not as a data-supported causal estimate.

The package’s poor-overlap scenario provides a positive example: the unsupported target led to no end-to-end detection. CWFM should preserve this behavior.

### Partial identification and sensitivity head

When hidden confounding cannot be excluded, the output should be

[
[\theta^-(\Gamma),\theta^+(\Gamma)]
]

over one or several sensitivity levels (\Gamma), rather than a falsely precise point estimate.

Recent work shows that PFNs can amortize causal sensitivity-bound computation across datasets, causal queries, and sensitivity levels. ([arXiv][7])

A similar head can produce bounds over:

* unmeasured confounding strength;
* exposure-mapping misspecification;
* missing network ties;
* invalid instruments;
* measurement error;
* violations of temporal exchangeability.

For the ZIP hidden-confounding scenario, the correct architecture-level behavior would be:

```text
Statistical association/spillover signal: strong
Point identification: failed
Estimated point effect: suppressed
Sensitivity bounds: returned
Required additional assumptions: listed
```

That is a better outcome than reproducing power 1.00 with zero interval coverage.

### Prior-mismatch detector

The synthetic prior is effectively part of the model specification. Do-PFN explicitly identifies prior–reality mismatch as a central unresolved limitation. ([arXiv][6])

The OOD module can combine:

* masked-reconstruction loss;
* residual distribution mismatch;
* mechanism-expert entropy;
* disagreement among world particles;
* distance from synthetic mechanism prototypes;
* performance on held-out portions of the observed dataset;
* disagreement between direct and compiled estimation paths.

A useful output is not simply “OOD probability,” but a decomposition such as:

```text
Marginal distributions: well supported
Noise structure: weakly supported
Temporal persistence: outside training range
Network degree distribution: moderately supported
Exposure-response mechanism: highly uncertain
```

### Self-compatibility audit

The system should rerun its structural decoder on overlapping subsets of variables. Incompatible induced structures provide evidence of finite-sample error or violated assumptions.

Self-compatibility has been proposed as a way to falsify causal-discovery outputs when ground-truth graphs are unavailable, by examining whether graphs learned on different variable subsets can be jointly compatible. ([Proceedings of Machine Learning Research][8])

This is especially relevant for a foundation model because an apparently confident single-pass prediction may otherwise conceal instability.

### External calibration layer

The neural posterior should not automatically be interpreted as a frequentist confidence interval.

Where exchangeability and sample splitting are appropriate, a separate calibration wrapper can operate on:

* row-level splits for i.i.d. data;
* cluster-level splits for interference;
* block splits for temporal data;
* world-weighted estimates under graph uncertainty.

Recent work on conformal inference under graph uncertainty proposes aggregating over graph-conditioned estimates before calibration rather than calibrating only one selected graph. ([arXiv][9])

## 9. Pretraining distribution

The model is only as broad as its synthetic and semi-synthetic causal prior. Training should proceed by sampling complete causal episodes.

For each episode:

1. Sample a structural family.
2. Sample mechanisms and noise.
3. Sample measurement, missingness, and assignment processes.
4. Generate observational context.
5. Generate randomized, soft-interventional, or counterfactual targets when appropriate.
6. Sample one or more formal queries.
7. Compute graph, latent, regime, identification, support, effect, and sensitivity labels.
8. Hide the labels and train the model to recover them.

### Structural families

The prior should include:

* sparse and moderately dense DAGs;
* scale-free, small-world, block, geometric, chain, collider, and mediator structures;
* ADMGs with latent confounding;
* observational equivalence classes;
* cyclic contemporaneous models when explicitly supported;
* temporal lag graphs;
* latent measurement structures;
* network interference structures.

### Mechanism families

The prior should span:

* linear and generalized linear models;
* smooth additive nonlinearities;
* neural-network mechanisms;
* splines and Gaussian-process-like mechanisms;
* thresholds and discontinuities;
* interactions;
* heteroskedasticity;
* mixed continuous, binary, ordinal, and count outcomes;
* heavy-tailed and non-Gaussian noise;
* measurement error;
* weak and near-null effects.

### Temporal families

Temporal episodes should cover:

* no changes;
* one or multiple abrupt changes;
* weak changes;
* close breakpoints;
* short segments;
* gradual drift;
* recurrent regimes;
* intercept-only changes;
* coefficient-only changes;
* noise-only changes;
* covariate-process changes;
* lag-order mismatch;
* treatment–confounder feedback;
* delayed and cumulative treatment effects;
* irregular sampling.

CausalTimePrior is an example of a recent effort to generate paired observational and interventional temporal data with nonlinear dynamics, regime switching, and hard, soft, or time-varying interventions. ([arXiv][10])

### Interference families

Network episodes should include:

* cluster randomization and observational assignment;
* homophily;
* direct effects only;
* no-spillover controls;
* linear, threshold, nonmonotone, and interactive spillovers;
* sparse affected outcomes;
* heterogeneous degree;
* poor exposure overlap;
* hidden confounding;
* noisy, incomplete, and misspecified relation matrices;
* interference through unobserved edges;
* multiple plausible exposure mappings.

### Identification-contrastive pairs

A particularly important training device is to generate pairs of SCMs that are observationally indistinguishable or nearly indistinguishable but imply different causal effects.

The desired training target is not “choose the correct hidden SCM.” It is:

* represent both worlds;
* recognize that the effect is not point identified;
* return the identified set or prior-dependent range;
* avoid confident selection based on simulator-specific artifacts.

Other contrastive pairs should include:

* one global nonlinear mechanism versus two linear regimes;
* intercept change versus lagged-(B) change;
* direct-only network effects versus spillovers;
* true exposure mapping versus plausible misspecified mapping;
* weak measurement versus incorrect sparse support;
* association graph versus causally directed graph.

## 10. Training curriculum and losses

A single-stage multi-task objective would probably suffer from destructive interference. A staged curriculum is safer.

### Stage 1: representation and reconstruction

Train:

* masked cell reconstruction;
* missingness modeling;
* variable-type prediction;
* distributional summaries;
* temporal ordering and network-neighborhood reconstruction.

This develops useful statistical representations before demanding causal outputs.

### Stage 2: structural supervision

Train:

* graph skeleton and endpoint probabilities;
* topological order;
* bidirected edges;
* latent assignments;
* number of active latent factors;
* regime boundaries;
* change types;
* exposure mappings;
* assignment and selection mechanisms.

### Stage 3: interventional prediction

Train the direct decoder on:

[
p(Y^{do(a)}\mid x)
]

and related policy, spillover, and trajectory queries.

### Stage 4: identification, support, and bounds

Introduce:

* formal identification labels;
* unsupported-target labels;
* sensitivity bounds;
* hidden-confounding stress cases;
* partial-graph queries;
* contradictory-assumption episodes.

### Stage 5: joint training and calibration

Jointly optimize the complete architecture using balanced batches, task-specific normalization, sparse routing, and gradient-conflict control.

A schematic objective is

[
\begin{aligned}
\mathcal L
={}&
\lambda_{\text{mask}}\mathcal L_{\text{mask}}
+
\lambda_{\text{world}}\mathcal L_{\text{world}}
+
\lambda_{\text{graph}}\mathcal L_{\text{graph}}\
&+
\lambda_{\text{do}}\mathcal L_{\text{do}}
+
\lambda_{\text{id}}\mathcal L_{\text{id}}
+
\lambda_{\text{support}}\mathcal L_{\text{support}}\
&+
\lambda_{\text{bounds}}\mathcal L_{\text{bounds}}
+
\lambda_{\text{cons}}\mathcal L_{\text{cons}}
+
\lambda_{\text{select}}\mathcal L_{\text{selective}}.
\end{aligned}
]

Important components include:

| Loss                            | Function                                                  |
| ------------------------------- | --------------------------------------------------------- |
| (\mathcal L_{\text{mask}})      | Conditional likelihood of masked observed values          |
| (\mathcal L_{\text{world}})     | Mechanism, regime, exposure, and latent-world recovery    |
| (\mathcal L_{\text{graph}})     | Skeleton, orientation, bidirected, and lag-edge losses    |
| (\mathcal L_{\text{do}})        | Proper scoring rule for interventional distributions      |
| (\mathcal L_{\text{id}})        | Identification-status classification                      |
| (\mathcal L_{\text{support}})   | Query-specific overlap and support                        |
| (\mathcal L_{\text{bounds}})    | Lower and upper partial-identification bounds             |
| (\mathcal L_{\text{cons}})      | Structural and query consistency                          |
| (\mathcal L_{\text{selective}}) | Penalizes confident errors more than justified abstention |

Consistency losses should include:

* invariance to row permutations;
* equivariance to variable permutations;
* compatibility across variable subsets;
* zero effect when no directed causal path exists in a sampled SCM;
* agreement between direct and compiled estimators in identified settings;
* consistency between predicted regimes and predicted mechanism changes;
* monotonicity of sensitivity-bound width as the allowed violation increases;
* no unsupported causal upgrade from association to intervention.

Null, weak-effect, nonidentified, and unsupported episodes must be deliberately oversampled. Otherwise, a model trained mostly on strong causal effects will learn that producing an effect is usually rewarded.

## 11. A realistic reference configuration

The following is a plausible starting configuration rather than an optimized specification.

| Component                   | Initial configuration                                          |
| --------------------------- | -------------------------------------------------------------- |
| Hidden dimension            | 384                                                            |
| Axial blocks                | 10–12                                                          |
| Attention heads             | 8                                                              |
| Sample/time inducing tokens | 8 per aggregation stage                                        |
| Mechanism experts           | 8 experts, top-2 routing                                       |
| World particles             | 16                                                             |
| Maximum latent slots        | 16                                                             |
| Maximum regime slots        | 8                                                              |
| Maximum variables per pass  | 128                                                            |
| Context units/rows per pass | Up to approximately 4,096                                      |
| Temporal length             | Up to approximately 256                                        |
| Maximum explicit lag        | 8                                                              |
| Continuous output           | 128-bin distribution or small mixture-density head             |
| Graph decoder               | Symmetric skeleton head, order head, bidirected head, lag head |
| Large-data strategy         | Coreset/context retrieval plus hierarchical aggregation        |

For more than roughly 128–256 variables, the (P^2) graph head becomes expensive. A coarse-to-fine procedure can first estimate low-rank edge scores, retain the top candidate neighborhoods, and run a high-resolution local graph decoder around the queried treatment and outcome.

“One checkpoint” should therefore not mean “one identical computation for every problem.” The checkpoint can share the causal backbone while activating different temporal, network, latent, and estimation adapters.

## 12. How the modules target the ZIP failures

The architecture is directly motivated by the benchmark outcomes in the [report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md).

| ZIP finding                                                                                     | Architectural response                                                                                                                         |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Stationary nonlinear mechanism falsely split in 5/5 runs under linear misspecification          | Stationary-nonlinear and regime hypotheses compete explicitly; mechanism MoE; complexity prior; structural loss independent of predictive gain |
| Weak and outcome-sparse changes often missed                                                    | Hierarchical multi-outcome decoder; posterior probabilities instead of hard thresholds; weak-effect curriculum                                 |
| Simple estimator better for linear spillover, flexible estimator better for threshold spillover | Mechanism router plus direct and compiled estimation paths; cross-fitted risk-based aggregation                                                |
| Hidden confounding gave high apparent power, large bias, and zero coverage                      | Bidirected/PAG particles; formal identification gate; sensitivity-bound head; suppression of unjustified point estimates                       |
| Poor exposure overlap correctly caused abstention                                               | Query-specific support head; common-support target option; prior-dominated extrapolation label                                                 |
| Misspecified or noisy relation matrices damaged effects                                         | Separate exposure-map posterior; relation uncertainty; map-specific sensitivity analysis                                                       |
| Good prediction coexisted with poor latent structure                                            | Separate predictive and structural objectives; soft latent slots; structure-specific calibration                                               |
| Error propagated from reconstructed latent scores and estimated groups                          | Joint posterior over measurements, groups, latent scores, and graph; no hard intermediate commitment                                           |
| Multiple and weak temporal breaks were missed                                                   | Set-based multi-break decoder; multi-scale temporal attention; weak-break curriculum                                                           |
| Gradual drift was detected but poorly localized as one point                                    | Transition-window and continuous-drift output rather than forced point breakpoint                                                              |
| Intercept change falsely attributed to lagged (B)                                               | Explicit change-type head; intercept-versus-(B) contrastive episodes                                                                           |
| Null calibration differed sharply across methods                                                | Null-rich prior, selective loss, cluster/block-aware audit, and external calibration                                                           |

## 13. Inference procedure in a real application

At deployment, the model would execute the following conceptual sequence:

1. **Validate the episode.** Check types, temporal ordering, network dimensions, cluster structure, missingness, and the formal query.
2. **Encode the data.** Produce cell, variable, unit, environment, temporal, and relation representations.
3. **Infer causal-world particles.** Obtain weighted graph, mechanism, latent, regime, and exposure hypotheses.
4. **Run formal identification.** Determine what can be claimed under each world and supplied assumption set.
5. **Evaluate support.** Determine whether the requested intervention is supported for the target population.
6. **Estimate through both paths.** Predict the interventional distribution and evaluate the compiled observable functional.
7. **Run an audit.** Use row/cluster/block perturbations, variable-subset compatibility, alternative exposure maps, and posterior predictive checks.
8. **Calibrate or widen uncertainty.**
9. **Return a structured result or abstain.**

A suitable output contract would be:

```yaml
claim_type: interventional

estimand:
  name: average_spillover_effect
  own_treatment: 0
  exposure_contrast: [0.25, 0.75]
  target_population: common_support

assumptions:
  assignment: cluster_randomized_saturation
  interference_mapping: one_hop_treated_fraction
  hidden_confounding: not_excluded
  consistency: assumed

identification:
  status: partially_identified
  robust_across_worlds: false
  reason: possible_unmeasured_cluster_confounding

support:
  status: adequate
  minimum_effective_cluster_count: 14

result:
  point_estimate: suppressed
  sensitivity_bounds:
    gamma_1_0: [0.12, 0.34]
    gamma_1_5: [-0.03, 0.46]
    gamma_2_0: [-0.18, 0.58]

structure:
  exposure_mapping_posterior:
    one_hop_fraction: 0.61
    weighted_one_hop: 0.25
    distance_two: 0.14

diagnostics:
  graph_entropy: moderate
  prior_mismatch: low
  direct_vs_compiled_disagreement: moderate
  variable_subset_compatibility: passed
  cluster_bootstrap_stability: 0.78

decision:
  status: no_point_causal_claim
  explanation: conclusion_depends_on_hidden_confounding_strength
```

The output itself thereby distinguishes a detected pattern from an identified causal effect.

## Overall recommendation

The strongest architecture is therefore **neither a purely neural end-to-end estimator nor a loose collection of conventional causal procedures**. It is a hybrid:

[
\boxed{
\text{typed causal encoder}
+
\text{posterior causal worlds}
+
\text{formal identification}
+
\text{query-conditioned estimation}
+
\text{reliability and abstention}
}
]

The key architectural innovation is the **posterior causal-world layer**. It lets the model share statistical strength across graph discovery, latent measurement, regime determination, temporal structure, and network interference without forcing all uncertainty into one averaged representation.

The key safety mechanism is the **identification and support gate**. Without it, the architecture could become a highly efficient version of the hidden-confounding failure already seen in the package: strong apparent detection, severe bias, and unjustified certainty.

The most informative first implementation would cover static tabular effects, observed-regime determination, and network interference while retaining the dual estimation path and reliability layer. Latent slots and temporal regime decoding can then be added without changing the central episode, world-particle, query, or identification interfaces.

[1]: https://arxiv.org/html/2605.07204v1 "Arrow: A Foundation Model for Causal Discovery"
[2]: https://arxiv.org/html/2607.11508v1 "CDFM: Towards a General-Purpose Causal Discovery Foundation Model"
[3]: https://arxiv.org/html/2602.14972v1 "Use What You Know: Causal Foundation Models with Partial Graphs"
[4]: https://arxiv.org/html/2606.20889v1 "Temporal Causal Prior-Data Fitted Networks for Panel Data with Learned Reliability Signals"
[5]: https://arxiv.org/html/2506.10914v1 "Foundation Models for Causal Inference via Prior-Data Fitted Networks"
[6]: https://arxiv.org/html/2506.06039v1 "Do-PFN: In-Context Learning for Causal Effect Estimation"
[7]: https://arxiv.org/html/2605.10590v1 "Amortizing Causal Sensitivity Analysis via Prior Data-Fitted Networks"
[8]: https://proceedings.mlr.press/v238/faller24a.html "Self-Compatibility: Evaluating Causal Discovery without Ground Truth"
[9]: https://arxiv.org/html/2605.21928v1 "CausalGuard: Conformal Inference under Graph Uncertainty"
[10]: https://arxiv.org/html/2603.11090v1 "Interventional Time Series Priors for Causal Foundation Models"
