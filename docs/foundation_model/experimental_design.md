# Experimental design for the Causal World Foundation Model

The experiment should test more than whether the proposed **Causal World Foundation Model (CWFM)** obtains the best average prediction score. The central question is:

> Does a single pretrained causal model improve the joint frontier of causal accuracy, structural recovery, calibration, robustness, computational reuse, and safe abstention compared with well-tuned task-specific methods?

A convincing study should be designed so that classical methods are expected to win in some settings. In particular, simple estimators should often remain superior when their assumptions are exactly correct. The proposed foundation model is primarily expected to help when the analyst does not know the correct method class in advance, when several causal structures interact, or when uncertainty must be propagated across multiple processing stages.

The numerical expectations below are **informed forecasts, not experimental results or formal power calculations**. Package values are taken from the existing [benchmark report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md).

---

## 1. Main hypotheses

### H1. Non-inferiority in simple, correctly specified settings

On clean linear, low-dimensional, well-identified problems, CWFM should be approximately non-inferior to the best classical specialist:

[
\frac{
L_{\text{CWFM}}-L_{\text{oracle}}
}{
L_{\text{classical}}-L_{\text{oracle}}
}
\leq 1.05,
]

where (L) is an appropriate error or regret metric.

The purpose is not to establish superiority here. A correctly specified linear model, graphical lasso, design-based spillover estimator, or classical change-point procedure has a strong variance and calibration advantage.

### H2. Superiority under mechanism heterogeneity and misspecification

CWFM should outperform a fixed classical estimator when the correct mechanism class is unknown or varies between datasets:

* linear versus nonlinear responses;
* smooth versus threshold spillovers;
* one versus several regimes;
* mixed outcome types;
* latent measurement error;
* abrupt versus gradual temporal changes;
* different network exposure mappings.

The expected gain comes from amortized mechanism selection, model averaging, and shared learning across related tasks.

### H3. Better safety under identification and support failures

Compared with estimators that always return a point estimate, the full model should:

* make fewer false point-causal claims under non-identification;
* refuse unsupported treatment or network-exposure contrasts;
* return sensitivity bounds when hidden confounding or exposure-map uncertainty cannot be ruled out;
* distinguish predictive, associational, Granger, and interventional claims.

This is at least as important as reducing RMSE.

### H4. Better parameter-shift generalization, but not universal mechanism-shift robustness

CWFM should generalize reasonably well when familiar causal mechanisms are evaluated at new:

* sample sizes;
* dimensionalities;
* effect magnitudes;
* graph densities;
* missingness rates;
* degrees of temporal dependence.

Its advantage should shrink, and may reverse, when an entire mechanism family is absent from pretraining. The intended success criterion in severe mechanism shift is therefore either competitive accuracy **or reliable out-of-distribution detection and abstention**.

### H5. Reduced downstream error propagation

Joint inference should improve end-to-end performance where the classical pipeline makes hard intermediate decisions:

[
X
\rightarrow \widehat{\text{latent groups}}
\rightarrow \widehat Z
\rightarrow \widehat G
\rightarrow \widehat{\theta}.
]

The largest improvements should appear in the end-to-end latent graph level, not when true latent scores are provided.

### H6. Better reuse across heterogeneous analyses

After pretraining, CWFM should provide faster marginal inference across many new datasets and queries. This should be reported separately from its substantial pretraining cost.

---

# 2. Compared systems

The primary comparison should include four classes of systems.

## 2.1 Proposed model variants

| System              | Description                                                                                                               | Purpose                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **CWFM-Direct**     | Frozen foundation model using its direct interventional or structural decoder                                             | Measures raw amortized performance |
| **CWFM-Hybrid**     | Direct decoder plus posterior causal worlds, formal identification, support diagnostics, and compiled classical estimator | Main proposed architecture         |
| **CWFM-Calibrated** | CWFM-Hybrid plus external bootstrap, randomization, conformal, or one-step calibration where applicable                   | Primary confirmatory system        |
| **CWFM-Few-Shot**   | Frozen backbone with a small adapter trained without test causal labels                                                   | Optional adaptation experiment     |

The primary claim should be based on **CWFM-Calibrated**, because raw neural posterior intervals should not automatically be treated as statistically calibrated intervals. Recent theoretical work reports that causal PFNs can retain prior-induced confounding bias and proposes one-step posterior correction to restore frequentist behavior for appropriate estimands. This strongly motivates comparing both raw and corrected versions. ([arXiv][1])

## 2.2 Important ablations

The ablations should test whether each architectural component addresses a specific weakness from the package.

| Ablation                      | Removed component                                             | Expected consequence                                             |
| ----------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| `CWFM – ID`                   | Formal identification engine                                  | More false point estimates under hidden confounding              |
| `CWFM – support`              | Query-specific overlap detector                               | Invalid extrapolation in poor-overlap scenarios                  |
| `CWFM-single-world`           | Posterior world particles                                     | Worse structural calibration and multimodal uncertainty          |
| `CWFM – MoE`                  | Mixture of mechanism experts                                  | Poorer routing between simple linear and nonlinear mechanisms    |
| `CWFM-hard-latent`            | Joint soft latent inference                                   | Reproduction of the Level B-to-C error propagation               |
| `CWFM-fixed-network-map`      | Exposure-map posterior                                        | Greater sensitivity to misspecified relation matrices            |
| `CWFM-point-break`            | Drift and transition-window decoder                           | Poor localization of gradual drift                               |
| `CWFM – external calibration` | Bootstrap or one-step correction                              | Undercoverage despite good point accuracy                        |
| `CWFM-narrow-prior`           | Broad nonlinear, latent, network, and temporal training prior | Better simple-case specialization but severe OOD degradation     |
| `Predictive-FM`               | Same backbone trained without causal targets                  | Tests whether gains come merely from architecture and data scale |

A task-specific neural model trained from scratch should also be included. This distinguishes the value of **cross-task causal pretraining** from the value of using a transformer architecture.

## 2.3 Classical specialist baselines

The exact methods already implemented in the package should remain the first baselines. They should not be replaced after seeing the foundation-model results.

Stronger canonical alternatives should also be included. These include model-based recursive partitioning for parameter instability, generalized random forests and double/debiased machine learning for heterogeneous treatment effects, randomization-based estimators for interference, sparse PLS for supervised latent components, graphical lasso for conditional-association graphs, and PELT or Bai–Perron procedures for multiple structural breaks. ([Zeileis][2])

## 2.4 Existing specialized causal foundation models

These should be treated as **modern specialized baselines**, not classical methods:

* CausalFM, CausalPFN, and Do-PFN for static effect estimation;
* Arrow and CDFM for causal discovery;
* TCPFN or related temporal PFNs for temporal causal tasks.

They cover narrower task families than the proposed model, so they should be compared only on tasks they are designed to answer. CausalFM and CausalPFN amortize causal-effect estimation from synthetic priors, Arrow and CDFM target zero-shot causal discovery, and TCPFN targets temporal causal analysis with learned reliability outputs. ([arXiv][3])

---

# 3. Three comparison tracks

A single leaderboard would obscure important distinctions. The study should contain three tracks.

## 3.1 Task-aware specialist track

Every method is told the task family:

* regime determination;
* interference;
* latent grouping;
* latent graph;
* contemporaneous change;
* lagged VARX;
* standard treatment-effect estimation.

The classical method is allowed nested hyperparameter tuning and receives the same task-relevant metadata as CWFM. It is not told the true mechanism, graph, breakpoint, or exposure mapping unless that information would genuinely be known by design.

This is the most demanding comparison for CWFM because it compares one general model with a specialist selected for the task.

## 3.2 Unified automation track

One system must process all task families without manual estimator selection.

Compare:

1. CWFM;
2. an automated classical portfolio;
3. a router over existing narrow causal foundation models;
4. a general predictive foundation model followed by classical causal estimators.

The automated classical portfolio may select among candidate methods using only:

* data schema;
* design metadata;
* cross-validated observed-data fit;
* stability diagnostics;
* predeclared specification tests.

It must not use oracle causal effects or structural truth.

This track tests the main practical claim: whether amortized causal learning can reduce the burden of manually choosing among many fragmented procedures.

## 3.3 Safety and selective-inference track

Methods may abstain. Performance is reported as a function of the fraction of queries answered.

For answer coverage (c), define selective risk

[
R(c)
====

\mathbb E
\left[
\ell(\widehat\theta,\theta)
\mid \text{query answered at coverage }c
\right].
]

Report:

* risk–coverage curves;
* area under the risk–coverage curve;
* false point-claim rate;
* supported-query acceptance;
* unsupported-query rejection;
* sensitivity-bound coverage and width.

This prevents a model from appearing safe merely by refusing every difficult query, or appearing powerful by returning an answer for every nonidentified query.

---

# 4. Experimental stages

## Stage A: Expanded replication of the package

Retain all 88 package scenarios. Do not select only those expected to favor CWFM.

### Replication size

The current package uses five final seeds. The confirmatory run should use:

* **200 paired seeds per scenario** for general performance;
* **1,000 paired seeds** for safety-critical null, non-identification, and support scenarios;
* at least **199 calibration repetitions** in the broad experiment;
* **999 repetitions** for final headline type-I-error and coverage results.

The 88 scenarios with 200 seeds produce:

[
88\times200=17{,}600
]

primary evaluation episodes, before the expanded safety cells.

The package itself recommends at least 200 final seeds and 199–999 calibration repetitions, because the present five-seed estimates cannot establish nominal error rates. See the [report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md).

### Paired simulation

For a given scenario and seed, every method receives the same:

* structural model;
* exogenous draws where possible;
* discovery sample;
* evaluation sample;
* observed graph or relation matrix;
* missingness pattern;
* formal causal query.

This provides lower-variance paired comparisons.

### Blind final run

Before the final seeds are generated:

* freeze the CWFM checkpoint;
* freeze all baseline hyperparameter grids;
* freeze evaluation code;
* register primary metrics and hypotheses;
* prohibit test-specific prompt changes or adapters;
* record model, code, data-generator, and configuration hashes.

---

## Stage B: Continuous stress-response surfaces

The package uses informative named scenarios, but a method can look good or bad because of one arbitrary parameter choice. Add continuous stress experiments.

For each task family, sample approximately 100 configurations using a Latin hypercube or Sobol design, with 20 paired replicates per configuration.

Suggested dimensions are:

| Dimension                            | Example range                                   |
| ------------------------------------ | ----------------------------------------------- |
| Sample size                          | (100)–(5{,}000)                                 |
| Number of observed variables         | (5)–(256)                                       |
| (p/n) ratio                          | (0.01)–(2)                                      |
| Signal-to-noise ratio                | (0.2)–(5)                                       |
| Effect or mechanism-change magnitude | Null through strong                             |
| Missingness                          | (0%)–(40%)                                      |
| Nonlinearity                         | Linear through strongly nonlinear               |
| Confounding strength                 | None through severe                             |
| Treatment overlap                    | Strong through almost absent                    |
| Graph density                        | Sparse through near-identifiability limit       |
| Network measurement error            | (0%)–(40%) edge corruption                      |
| Temporal persistence                 | Weak through near-unit-root                     |
| Breakpoint separation                | Below through well above minimum segment length |

Fit response surfaces for each method rather than reporting only average ranks. This will reveal, for example, the effect magnitude at which CWFM begins to outperform MOB, or the graph-noise level at which neither CWFM nor the classical interference estimator is reliable.

---

## Stage C: Out-of-distribution evaluation

Four different forms of OOD generalization should be distinguished.

### C1. Parameter OOD

Use familiar mechanism families outside the pretraining range:

* larger (n) or (p);
* denser graphs;
* stronger persistence;
* weaker measurement;
* more severe imbalance;
* more breakpoints.

### C2. Mechanism-family OOD

Completely exclude one mechanism family from pretraining and evaluate on it:

* threshold effects;
* nonmonotone exposure responses;
* gradual drift;
* heavy-tailed noise;
* cyclic temporal mechanisms;
* cross-loadings;
* distance-two interference;
* selection bias.

This is a harder and more meaningful test than merely changing numerical parameters.

### C3. Generator OOD

The test generator should be independently implemented by another researcher or team.

Otherwise, the foundation model may learn artifacts of:

* a particular random graph routine;
* a characteristic noise scale;
* a simulator’s variable ordering;
* a specific discretization procedure;
* the exact form of missing-value generation.

### C4. Schema and semantic OOD

Vary:

* column order;
* units of measurement;
* variable names;
* category labels;
* event vocabularies;
* sampling frequencies;
* process versions;
* network sizes.

Numerical transformations should preserve the causal problem while changing superficial presentation.

---

## Stage D: Semi-synthetic process-event and inventory benchmark

This stage tests whether process event data genuinely act as the “glue” between temporal, network, latent, and policy causal reasoning.

### 4.1 Discrete-event causal simulator

Construct a multi-echelon inventory and order-fulfilment simulator with entities such as:

* products or SKUs;
* customer orders;
* suppliers;
* warehouses;
* transport resources;
* planners or automated decision systems.

Generate event logs containing:

```text
order_created
forecast_updated
inventory_checked
reorder_decision
purchase_order_created
supplier_confirmed
shipment_dispatched
goods_received
allocation_performed
order_backordered
expedite_requested
order_fulfilled
```

### 4.2 Interventions

Potential interventions include:

* change reorder point;
* change order quantity;
* expedite an order;
* switch supplier;
* prioritize a customer order;
* reserve inventory;
* alter batching policy;
* introduce a new forecasting model.

### 4.3 Outcomes

Use several outcomes simultaneously:

* stockout days;
* service level or fill rate;
* order lateness;
* lead time;
* holding cost;
* shortage cost;
* expedite cost;
* total cost;
* customer cancellation.

### 4.4 Causal complexities

The simulator should deliberately combine:

* time-varying confounding from demand forecasts and inventory position;
* latent supplier reliability or demand state;
* interference through shared supplier and warehouse capacity;
* regime changes from seasonal demand, disruptions, and policy rollouts;
* missing events and delayed recordings;
* variable treatment timing;
* transport and supplier networks;
* heterogeneous effects by product and location.

### 4.5 Process-specific baselines

Compare CWFM with:

1. hand-engineered prefix features plus linear g-computation;
2. prefix features plus AIPW or DML;
3. prefix features plus causal forest;
4. sequence embedding followed by DML;
5. process-state aggregation followed by a classical VARX or change-point method;
6. design-based analysis when the simulated policy is randomized;
7. an oracle state representation containing the simulator’s true state.

The primary question is whether CWFM gains come from retaining the full event history or simply from using better downstream regression.

### 4.6 Process holdouts

Hold out entire:

* warehouses;
* process versions;
* supplier-network topologies;
* product categories;
* disruption types;
* treatment policies;
* activity vocabularies.

This is more realistic than randomly splitting events from the same operational process.

### 4.7 Semi-synthetic real-log variant

A real event log can provide:

* trace-length distribution;
* activity order;
* temporal gaps;
* resource utilization;
* missingness;
* case attributes.

A separately defined simulator can then overlay treatments, outcomes, hidden states, and counterfactuals. Real observational logs alone generally cannot supply true unit-level counterfactual labels, so they should not be presented as direct ground-truth causal benchmarks. ([Proceedings of Machine Learning Research][4])

---

## Stage E: Real-data plausibility evaluation

Real-data experiments should complement, not replace, the synthetic ground-truth tests.

Prefer:

* randomized trials;
* randomized encouragement designs;
* cluster-randomized interventions;
* randomized network experiments;
* operational A/B tests;
* staggered policy rollouts with defensible designs.

Report:

* agreement with design-based estimates;
* negative-control behavior;
* placebo-period results;
* sensitivity to adjustment-set changes;
* stability across sites;
* transport to held-out environments;
* decision or policy value where it can be validly estimated.

For purely observational datasets, the study should describe results as plausibility, compatibility, or sensitivity evidence rather than proof that the estimated causal structure is correct.

---

# 5. Task-by-task classical comparisons

## 5.1 Observed regime determination

### Compared methods

1. Package MOB-like ridge tree with full split search and permutation root gate.
2. Canonical model-based recursive partitioning.
3. Predictive CART.
4. Random forest or boosted tree as a predictive diagnostic.
5. Finite-mixture regression without observed partition variables.
6. CWFM variants.
7. Oracle regime tree.

For a separate known-treatment heterogeneity extension, include causal forest or generalized random forest. This extension should not be confused with the package’s broader parameter-instability target.

### Primary metrics

* false split rate under the complete null;
* detection power;
* true root-variable recovery;
* breakpoint or threshold error;
* regime ARI;
* number of unnecessary leaves;
* parameter error;
* held-out loss;
* percentage of oracle loss improvement recovered;
* bootstrap or posterior selection stability.

### Critical contrast

The most important pair is:

[
\text{stationary nonlinear mechanism}
\quad\text{versus}\quad
\text{true regime-specific mechanisms}.
]

A method fails causally if it obtains excellent predictive loss by inventing a nonexistent regime.

---

## 5.2 Network interference

### Compared methods

1. Package flexible cross-fitted outcome regression.
2. Package simple (A,G,A\times G) estimator.
3. Design-based Horvitz–Thompson or Hájek exposure estimator.
4. Discrete-exposure AIPW.
5. Generalized propensity/dose-response estimator for observational assignment.
6. Flexible GAM or spline outcome model.
7. CWFM with a declared exposure map.
8. CWFM with exposure-map uncertainty.
9. Oracle estimator with true assignment probabilities, true map, and correct outcome model.

Randomization-based exposure estimators provide an especially important reference in randomized settings because they use the known design rather than relying entirely on outcome-model flexibility. ([Project Euclid][5])

### Primary metrics

* target support detection;
* false positive rate under no spillover;
* global and outcome-specific power;
* ASE bias and RMSE;
* confidence-interval coverage and width;
* dose-response integrated squared error;
* affected-outcome F1;
* exposure-map posterior accuracy;
* policy value and regret;
* refusal accuracy under poor overlap;
* sensitivity-bound coverage under hidden confounding or omitted edges.

### Two versions of the comparison

Run an **estimator-only version** with the true declared exposure map and verified randomization, and an **end-to-end version** where map uncertainty and identification must be handled by the system.

This prevents the proposed model from winning merely because it receives a stronger safety wrapper.

---

## 5.3 Outcome-relevant latent groups

### Compared methods

1. Package sparse sequential PLS2.
2. Dense PLS2.
3. PCA plus ridge regression.
4. Factor analysis plus multivariate regression.
5. MultiTaskLasso.
6. Sparse factor regression.
7. CWFM soft latent slots.
8. CWFM-hard-latent ablation.
9. Oracle true latent scores.

Sparse PLS is a natural primary classical comparison because it performs supervised dimension reduction and variable selection, whereas PCA tests whether unsupervised covariance structure is sufficient. ([OUP Academic][6])

### Primary metrics

* correct number of relevant groups;
* feature-support precision, recall, and F1;
* grouping ARI;
* subspace distance;
* latent-score correlation;
* reconstruction error;
* held-out outcome loss;
* support and group stability;
* calibration of soft group-membership probabilities.

Predictive and structural metrics must remain separate. The package already demonstrates that near-oracle prediction can coexist with poor group recovery.

---

## 5.4 Relationships between latent groups

### Primary undirected comparison

1. Graphical lasso at package Levels A, B, and C.
2. Neighborhood selection.
3. Shrinkage inverse covariance.
4. Stability-selected graphical lasso.
5. Joint CWFM latent and graph posterior.
6. Oracle precision graph.

Graphical lasso estimates a sparse inverse covariance and is therefore a conditional-association baseline, not automatically a directed causal-discovery method. ([OUP Academic][7])

### Directed extension

For data generated from identifiable directed models, add:

* PC;
* FCI where latent confounding is allowed;
* NOTEARS or another score-based DAG method;
* Arrow;
* CDFM.

Do not penalize a CPDAG or PAG for refusing to orient a direction that is not identifiable from the available data.

### Metrics

* skeleton precision, recall, F1, and AUPRC;
* partial-correlation RMSE;
* held-out Gaussian likelihood;
* edge-selection stability;
* spurious bridges;
* SHD for DAG-compatible outputs;
* endpoint accuracy for CPDAG/PAG outputs;
* SID only where a uniquely directed causal graph is an appropriate target;
* Level A/B/C performance differences.

The primary expected advantage of CWFM is at Level C. A gain at Level A is less likely because graphical lasso already performs near ceiling in several package scenarios.

---

## 5.5 Contemporaneous temporal changes

### Compared methods

1. Package bootstrap-calibrated binary segmentation.
2. PELT with an appropriate regression segment cost.
3. Bai–Perron multiple-break regression.
4. Fused-lasso or total-variation varying-coefficient regression.
5. Wild binary segmentation.
6. Pooled ridge with no breaks.
7. CWFM abrupt-break decoder.
8. CWFM transition-window and gradual-drift decoder.
9. Oracle segmentation.

PELT performs exact penalized multiple-change optimization for appropriate costs, while Bai–Perron procedures are established multiple-structural-break methods. These are particularly important because the package’s recursive method missed all tested multiple-break cases. ([arXiv][8])

### Metrics

* no-break false positive rate;
* breakpoint precision and recall at several tolerances;
* one-to-one localization error;
* segment ARI;
* number-of-breaks error;
* transition-window intersection over union;
* change-type confusion matrix;
* held-out segment loss;
* bootstrap or posterior breakpoint coverage.

For gradual drift, point distance from an arbitrary midpoint should be secondary. The primary metric should be overlap with the true transition interval.

---

## 5.6 Lagged VARX changes and temporal causal structure

### Compared methods

1. Package piecewise VARX detector.
2. PELT using VARX segment cost.
3. Bai–Perron VARX regression.
4. Fused or time-varying VAR.
5. Pooled VARX.
6. PCMCI or PCMCI+ for time-series graph discovery.
7. DYNOTEARS for simultaneous lagged and contemporaneous structure.
8. TCPFN or another temporal causal foundation model.
9. CWFM.
10. Oracle piecewise VARX.

PCMCI and DYNOTEARS represent different classical causal-discovery approaches for time series: conditional-independence testing and score-based dynamic-graph estimation. ([Science][9])

### Metrics

* raw-break detection;
* (B)-specific break precision and recall;
* localization;
* false attribution of (A), intercept, variance, or (X)-process changes to (B);
* coefficient-jump error;
* lag-support F1;
* held-out trajectory loss;
* calibration under strong persistence;
* claim-type correctness.

A lagged predictive relationship should be scored as Granger or dynamic predictive structure unless the simulated assumptions justify an intervention-level interpretation.

---

## 5.7 Standard causal-effect lane

The package primarily evaluates structure, interference, latent representation, and temporal changes. A causality foundation model also needs a direct treatment-effect benchmark.

Include:

* randomized and observational back-door settings;
* heterogeneous effects;
* multiple and continuous treatments;
* front-door identification;
* valid and weak instruments;
* mediation;
* longitudinal treatment;
* hidden confounding;
* poor overlap;
* transport across environments.

Classical baselines should include:

* linear g-computation;
* IPW;
* AIPW;
* TMLE;
* DML;
* causal forest or GRF;
* BART-based causal regression;
* 2SLS and suitable IV learners;
* appropriate front-door estimators;
* sensitivity and partial-identification procedures.

CausalFM, CausalPFN, and Do-PFN should be included here because this is the task family closest to their intended use. ([arXiv][3])

---

# 6. Primary metrics and statistical analysis

## 6.1 No single aggregate score should determine success

Each task family needs a primary metric:

| Family           | Primary metric                                |
| ---------------- | --------------------------------------------- |
| Regimes          | Regime ARI at controlled false split rate     |
| Interference     | ASE RMSE and coverage, conditional on support |
| Latent groups    | Group ARI and score correlation               |
| Latent graph     | End-to-end Level C graph F1/AUPRC             |
| Temporal changes | Break-set F1 and transition-window IoU        |
| VARX             | (B)-break F1 and false-attribution rate       |
| Static effects   | ATE/CATE error and interval coverage          |
| Process policy   | Policy regret                                 |

A secondary normalized regret can support cross-task summaries:

[
R^{\text{norm}}_{s,m}
=====================

\frac{
L_{s,m}-L_{s,\text{oracle}}
}{
L_{s,\text{naive}}-L_{s,\text{oracle}}
}.
]

Interpretation:

* (R^{\text{norm}}=0): oracle performance;
* (R^{\text{norm}}=1): no better than the naive baseline;
* (R^{\text{norm}}>1): worse than naive.

For metrics where higher is better, use the corresponding normalized gain.

## 6.2 Statistical comparisons

For every scenario:

* use paired differences across seeds;
* report paired bootstrap confidence intervals;
* report effect sizes, not only p-values;
* use Wilson or Jeffreys intervals for rates;
* correct the small set of confirmatory hypotheses using Holm’s procedure;
* treat additional scenario-wise comparisons as exploratory and use false-discovery-rate control.

Across scenarios, report:

* macro-average rank;
* median normalized regret;
* win/tie/loss counts;
* worst-case and 10th-percentile performance;
* a hierarchical model with scenario nested within task family.

An average win should not conceal catastrophic failures in hidden-confounding or poor-overlap scenarios.

## 6.3 Calibration metrics

Report separately:

* 90% and 95% interval coverage;
* average interval width;
* Brier score for null, support, and identification probabilities;
* expected calibration error;
* sensitivity-bound coverage;
* selective risk–coverage curves;
* out-of-distribution detection AUROC;
* false point-causal claim rate.

---

# 7. Expected results

## 7.1 Overall expected ranking

| Data regime                                                        | Expected strongest method                                           |
| ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| Clean, linear, low-dimensional, correctly specified                | Classical specialist                                                |
| Small or medium sample, nonlinear but inside pretraining prior     | CWFM or specialized causal FM                                       |
| Mixed outcome types and uncertain mechanism class                  | CWFM                                                                |
| Strong randomized design with known exposure mapping               | Design-based or simple augmented estimator                          |
| Latent measurement plus downstream graph estimation                | CWFM joint model                                                    |
| Single abrupt linear break                                         | Classical change-point specialist or tie                            |
| Several heterogeneous break types                                  | CWFM or a strong classical method selected by mechanism             |
| Correctly piecewise-linear multiple breaks                         | PELT/Bai–Perron may outperform CWFM                                 |
| Poor overlap                                                       | Any correct method should refuse; CWFM expected to automate refusal |
| Explicitly nonidentified query                                     | Formal identification engine and bounds, not a point estimator      |
| Severe unseen mechanism family                                     | Classical specialist or abstaining CWFM                             |
| Long process-event history with interactions across time and units | CWFM expected to have its clearest advantage                        |

The expected contribution is therefore **not universal dominance**. It is better average adaptation across heterogeneous problems, with fewer unsafe causal conclusions.

---

## 7.2 Expected package-specific results

The second column below contains selected results already recorded in the package. The third column is my forecast for the full calibrated CWFM.

| Package scenario                                      |                          Existing package result |                                                           Expected CWFM result |
| ----------------------------------------------------- | -----------------------------------------------: | -----------------------------------------------------------------------------: |
| Clean observed regime                                 |                    ARI (0.947), detection (1.00) |                                           ARI (0.94)–(0.98); essentially a tie |
| Weak regime change                                    |                                 Detection (0.00) |                             Power (0.20)–(0.40) at matched false-positive rate |
| Only some outcomes change                             |                                     Power (0.40) |                                                    Approximately (0.55)–(0.75) |
| Stationary nonlinear model, correct basis             |                                   No false split |                       No meaningful gain; both should remain near nominal null |
| Stationary nonlinear model, misspecified linear basis |                               False split (1.00) |                                        False split approximately (0.10)–(0.25) |
| Linear spillover                                      |       Flexible RMSE (0.149); simple RMSE (0.102) |                             RMSE (0.10)–(0.13); simple estimator may still win |
| Threshold spillover                                   |       Flexible RMSE (0.146); simple RMSE (0.229) |                                                             RMSE (0.12)–(0.16) |
| Mixed outcome interference                            |                               Outcome F1 (0.300) |                                                 F1 approximately (0.45)–(0.65) |
| Sparse spillover outcomes                             |                              Global power (0.40) |                                              Power approximately (0.55)–(0.75) |
| Poor exposure overlap                                 |                            Correctly unsupported |                                                More than (95%) correct refusal |
| Hidden confounding                                    |                       RMSE (0.726), coverage (0) | Point estimate should usually be suppressed when non-identification is encoded |
| Misspecified exposure mapping                         |                   RMSE (0.220), coverage (0.650) |                       Approximately 10–20% lower RMSE with broader uncertainty |
| Correlated latent groups                              |                                      ARI (0.487) |                                                ARI approximately (0.60)–(0.75) |
| Cross-loadings                                        |                                      ARI (0.566) |                                                ARI approximately (0.65)–(0.80) |
| High-dimensional latent groups                        |                                      ARI (0.387) |                                                ARI approximately (0.50)–(0.65) |
| Weak measurement                                      |                        Score correlation (0.709) |                                                    Approximately (0.75)–(0.85) |
| Latent graph, Level A                                 |                             Often near (1.00) F1 |                                                       Little or no improvement |
| Latent graph, cross-loadings Level C                  |                                       F1 (0.829) |                                                 F1 approximately (0.88)–(0.93) |
| Latent graph, weak measurement Level C                |                                       F1 (0.820) |                                                 F1 approximately (0.87)–(0.92) |
| Latent graph, weak edges Level C                      |                                       F1 (0.575) |                                                 F1 approximately (0.62)–(0.72) |
| One temporal coefficient break                        |                                 Detection (1.00) |                                        Tie; no meaningful room for improvement |
| Multiple temporal breaks                              |                                 Detection (0.00) |                                             Recall approximately (0.50)–(0.75) |
| Weak temporal break                                   |                                 Detection (0.00) |                                              Power approximately (0.20)–(0.45) |
| Gradual drift                                         | Any detection (1.00), poor midpoint localization |                              Transition-window IoU approximately (0.60)–(0.80) |
| Several VARX (B)-breaks                               |                                 Detection (0.00) |                                             Recall approximately (0.40)–(0.70) |
| High-dimensional VARX                                 |                                 Detection (0.00) |                                             Recall approximately (0.30)–(0.55) |
| Intercept-only VARX break                             |                     False (B)-attribution (0.40) |                                                    Approximately (0.10)–(0.20) |

These expectations assume that the relevant mechanisms are represented broadly in pretraining and that the final model includes the proposed specialized heads. A generic transformer trained mainly for prediction would not be expected to achieve these improvements.

---

## 7.3 Important qualifications to the forecasts

### Classical methods should win some clean cases

The simple interference model is already better than the flexible package method in the correctly linear setting. I expect it to remain slightly better than CWFM unless the hybrid system routes almost perfectly to the linear expert.

Likewise:

* graphical lasso with true latent scores;
* MOB with a correctly specified local model;
* a correctly calibrated single-break regression;
* design-based estimators under known randomization;

may be statistically more efficient than a broad foundation model.

### Weak-signal gains will be limited

Pretraining can provide useful shrinkage and shared information, but it cannot create information absent from the sample. A rise from zero power to 20–40% is plausible in the package’s weak scenarios. Power near one would be implausible without an increased false-positive rate or a prior that effectively assumes the answer.

### Multiple breaks may favor improved classical methods

The package’s zero power on multiple breaks is partly a property of its conservative recursive implementation. PELT or Bai–Perron with appropriate regression costs could perform very well on correctly piecewise-linear data.

I would expect:

* PELT/Bai–Perron recall around (0.60)–(0.85);
* CWFM recall around (0.50)–(0.75);

when the truth exactly matches the classical method. CWFM should become more competitive when abrupt changes, drift, intercept changes, noise changes, and mechanism changes are mixed.

### Hidden confounding cannot be solved from patterns alone

There are two different hidden-confounding tests.

1. **Non-identification is implied by supplied assumptions or a partial graph.**
   The formal engine should almost always refuse a point estimate.

2. **Hidden confounding is entirely unobserved and no relevant assumption information is supplied.**
   No method can be guaranteed to recognize it from the observational distribution. The learned reliability head might detect suspicious patterns, but this is not an identification theorem.

Accordingly, success in the second setting should be measured by sensitivity analysis, structural uncertainty, and conservative claims—not by expecting perfect hidden-confounder classification.

### Large-sample performance may favor corrected classical estimation

A frozen PFN can be highly competitive at small and medium sample sizes but remain influenced by its synthetic prior. Classical semiparametric estimators can improve as the sample grows under their assumptions. The hybrid one-step or influence-function path is therefore expected to become increasingly important at large (n). The recent consistency analysis of causal PFNs provides a concrete reason to evaluate this explicitly. ([arXiv][1])

---

# 8. Expected process-event and inventory results

I would expect the largest practical CWFM gains in operational event data where several causal structures coexist.

## Against hand-engineered case-prefix features

Expected improvement:

* approximately **10–25% lower policy regret**;
* better transport to unseen process versions;
* better recovery of delayed and heterogeneous effects;
* better handling of shared-resource interference.

The largest gain should occur when the causal state depends on event order, elapsed time, and resource congestion that are not captured by simple aggregate features.

## Against sequence embeddings followed by DML

Expected improvement:

* approximately **0–10% lower policy regret**;
* stronger reliability and identification diagnostics;
* better integration of network and regime uncertainty.

A well-designed sequence representation plus cross-fitted DML will be a strong baseline. It may equal or outperform CWFM when the treatment effect is identified by a simple adjustment set and the learned sequence state is sufficient.

## Under new operational mechanisms

When the test environment contains an unseen mechanism—such as a supplier disruption or capacity-allocation rule absent from pretraining—the CWFM advantage may disappear. A successful result would then be:

* a high OOD score;
* wider intervals;
* reduced answer coverage;
* no confidently incorrect policy recommendation.

---

# 9. Expected ablation results

| Ablation                         | Expected observation                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| Remove formal identification     | Similar RMSE on identified data; severe increase in false confidence under hidden confounding  |
| Remove support detector          | Little change on supported queries; invalid estimates under poor overlap                       |
| Use one causal world             | Similar clean-case performance; worse calibration and failure to represent equivalence classes |
| Remove mechanism experts         | Worse linear-versus-threshold routing and more false regime splits                             |
| Use hard latent preprocessing    | Recreates much of the Level B-to-C graph degradation                                           |
| Fix one exposure map             | Worse performance under noisy or misspecified networks                                         |
| Force point breakpoints          | Good abrupt-break results but poor gradual-drift interpretation                                |
| Remove external calibration      | Narrower intervals but systematic undercoverage                                                |
| Train on a narrow linear prior   | Strong simple-case performance and poor nonlinear/mechanism-OOD results                        |
| Train only predictive objectives | Good held-out prediction but weak causal structure, identification, and support behavior       |

The predictive-FM control is especially important. Without it, an observed improvement could be attributed to a larger neural architecture rather than to causally structured pretraining.

---

# 10. Predeclared success criteria

A credible positive conclusion should require all of the following.

## Accuracy

* No more than 5% median degradation relative to the best classical specialist on clean, correctly specified scenarios.
* At least 10% lower median normalized regret on heterogeneous and moderately OOD scenarios.
* Improvement in at least four of the six package families, not merely one.

## Calibration

* Nominal 5% false-positive rates between approximately 2.5% and 7.5% in the large safety run.
* At least 92% empirical coverage for nominal 95% intervals on identified, supported, in-distribution queries.
* Explicit reporting of interval width, not coverage alone.

## Safety

* At least 95% rejection of clearly unsupported exposure contrasts.
* No more than 5% false point claims when formal inputs imply that the query is not point identified.
* Valid or conservative sensitivity-bound coverage in the predefined hidden-confounding experiments.
* No intervention-level wording for outputs evaluated only as conditional association or Granger prediction.

## Structural robustness

* Improvement over the classical Level C latent graph pipeline without degrading Level A.
* Lower false-split rate in the stationary nonlinear regime experiment.
* Lower intercept-to-(B) false attribution in VARX.

## Process utility

* At least 10% lower policy regret than hand-engineered process-state baselines.
* Non-inferiority to sequence-embedding plus DML on supported, identified process decisions.
* Acceptable performance on held-out process versions or appropriate OOD abstention.

## Computational reporting

Report both:

[
\text{total cost}
=================

\text{pretraining cost}
+
\sum_{\text{applications}}
\text{inference cost},
]

and marginal per-dataset cost after pretraining. The foundation-model claim is strongest when the checkpoint is reused across many datasets and causal queries.

---

# 11. Interpretation of possible outcomes

### Strong positive result

CWFM matches specialists in clean cases, improves heterogeneous and integrated cases, provides better calibrated selective inference, and reliably refuses unsupported or formally nonidentified queries.

This would support the claim that causal pretraining reduces the fragmentation of current causal workflows.

### Mixed but scientifically useful result

CWFM improves latent, process, and heterogeneous settings but is inferior to classical methods in clean linear and multiple-break problems.

This would still justify the architecture as a **router and uncertainty-aware integrator**, rather than as a universal replacement for classical estimators.

### Predictively strong but causally unsuccessful result

CWFM improves held-out loss but:

* invents false regimes;
* returns point effects under hidden confounding;
* extrapolates outside support;
* confuses Granger relations with intervention effects;
* produces undercovered intervals.

This would repeat the main weaknesses already exposed by the package and should not be considered evidence for a causal foundation model.

### Negative result

CWFM loses to task-aware classical methods across both in-distribution and OOD conditions, and its abstention behavior does not compensate for the loss.

This would suggest that the proposed task space is too heterogeneous for a common backbone, or that the synthetic pretraining prior does not transfer sufficiently well to the evaluation generators and real process data.

---

# Final expectation

My overall expectation is that **CWFM will not dominate the best classical estimator on each narrowly defined, correctly specified problem**. It is more likely to achieve the best average performance when:

* the method class is unknown;
* effects are heterogeneous or sparse;
* data types are mixed;
* latent measurement and downstream structure must be estimated jointly;
* temporal, network, and process structures coexist;
* many related analyses must be performed without per-dataset model engineering.

The most important expected improvement is not necessarily a lower point-estimation RMSE. It is a reduction in cases where the system gives a precise but scientifically unjustified answer.

In the package’s hidden-confounding experiment, the current procedure obtains apparent power (1.00) while suffering large bias and zero coverage. The defining success of the proposed foundation model would be to **decline that apparent victory**: recognize that the point effect is not justified, expose the assumptions on which identification depends, and return sensitivity bounds or abstain.

[1]: https://arxiv.org/html/2603.12037v3 "https://arxiv.org/html/2603.12037v3"
[2]: https://www.zeileis.org/papers/Zeileis%2BHothorn%2BHornik-2008.pdf "https://www.zeileis.org/papers/Zeileis%2BHothorn%2BHornik-2008.pdf"
[3]: https://arxiv.org/abs/2506.10914 "https://arxiv.org/abs/2506.10914"
[4]: https://proceedings.mlr.press/v97/alaa19a/alaa19a.pdf "https://proceedings.mlr.press/v97/alaa19a/alaa19a.pdf"
[5]: https://projecteuclid.org/journals/annals-of-applied-statistics/volume-11/issue-4/Estimating-average-causal-effects-under-general-interference-with-application-to/10.1214/16-AOAS1005.pdf "https://projecteuclid.org/journals/annals-of-applied-statistics/volume-11/issue-4/Estimating-average-causal-effects-under-general-interference-with-application-to/10.1214/16-AOAS1005.pdf"
[6]: https://academic.oup.com/jrsssb/article/72/1/3/7076443 "https://academic.oup.com/jrsssb/article/72/1/3/7076443"
[7]: https://academic.oup.com/biostatistics/article-abstract/9/3/432/224260 "https://academic.oup.com/biostatistics/article-abstract/9/3/432/224260"
[8]: https://arxiv.org/abs/1101.1438 "https://arxiv.org/abs/1101.1438"
[9]: https://www.science.org/doi/10.1126/sciadv.aau4996 "https://www.science.org/doi/10.1126/sciadv.aau4996"
