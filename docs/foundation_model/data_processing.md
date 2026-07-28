# Data-processing architecture for a causality-specific foundation model

The data-processing layer should not be treated as a generic ETL stage. In causal analysis, preprocessing determines:

* what the observational unit is;
* which variables existed before an intervention;
* which rows can legitimately be compared;
* whether network or temporal dependence is preserved;
* which causal quantity is being requested;
* and which assumptions are being imposed.

Consequently, the model should **not** receive only an (N\times P) matrix. It should receive a structured **causal episode** containing observations, design information, temporal and relational structure, background knowledge, and a formal causal query.

A useful abstraction is

[
\mathcal E=
\left(
D,,
V,,
I,,
R,,
C,,
K,,
Q
\right),
]

where:

* (D): observed data;
* (V): variable schema and measurement metadata;
* (I): intervention and assignment information;
* (R): temporal, network, and other relations;
* (C): environments, clusters, sites, or cohorts;
* (K): background knowledge and assumptions;
* (Q): the causal query.

The processing pipeline converts this episode into typed value tokens, variable tokens, temporal tokens, relational tokens, environment tokens, and query tokens.

---

## 1. Three separate data planes

The architecture should keep three kinds of data strictly separated.

### 1.1 Model-visible observational plane

This contains everything that could actually be known in the application:

* observed covariates;
* treatments or exposures;
* outcomes;
* timestamps;
* cluster and sequence identifiers;
* observed network relations;
* intervention-design information;
* missingness and censoring indicators;
* variable descriptions;
* background knowledge supplied by the analyst.

This is the only plane available at deployment.

### 1.2 Oracle supervision plane

This exists only for synthetic and semi-synthetic pretraining and evaluation. It can contain:

* the true causal DAG, ADMG, CPDAG, or PAG;
* structural equations;
* true latent scores and loading matrices;
* hidden confounders;
* true regimes and breakpoints;
* the true exposure mapping;
* true intervention distributions;
* individual or population counterfactuals;
* identification labels;
* true causal effects and sensitivity bounds.

None of this information may be visible to the encoder when it produces its prediction.

### 1.3 Held-out audit plane

This contains independent data used for:

* predictive evaluation;
* posterior predictive checks;
* calibration;
* coverage measurement;
* stability analysis;
* effect-estimation evaluation;
* synthetic ground-truth scoring.

The ZIP benchmarks already follow this principle by saving `discovery`, `evaluation`, `truth`, and `metadata` separately and by fitting preprocessing only on the discovery data. That separation should become part of the foundation model’s permanent data contract. See the [benchmark report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md).

A practical storage layout could be:

```text
episode_000123/
├── observations.parquet
├── variables.json
├── interventions.parquet
├── relations.parquet
├── environments.json
├── assumptions.json
├── queries.jsonl
├── split_definition.json
└── truth/                       # synthetic or semi-synthetic only
    ├── graph.npz
    ├── mechanisms.npz
    ├── latent_variables.npz
    ├── interventions.npz
    ├── causal_effects.npz
    └── identification.json
```

The current package’s NPZ-plus-JSON organization is a suitable prototype for this structure.

---

# 2. Which data should be provided?

## 2.1 Observation data

The basic observation table should identify the unit, sequence, time, and environment associated with every value.

For static tabular data, a wide table is sufficient:

```text
unit_id | environment_id | X1 | X2 | A | Y1 | Y2
```

For longitudinal, irregular, or event-log data, a long representation is safer:

```text
unit_id
sequence_id
timestamp
variable_id
value
measurement_status
data_source
quality_flag
```

The following identifiers are structurally important:

| Identifier       | Purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| `unit_id`        | Identifies the statistical or causal unit                |
| `sequence_id`    | Prevents lags from crossing trajectory boundaries        |
| `cluster_id`     | Preserves the treatment-assignment or dependence cluster |
| `environment_id` | Distinguishes sites, policies, batches, or domains       |
| `timestamp`      | Establishes order and treatment–outcome timing           |
| `case_id`        | Identifies a process-mining trace                        |
| `event_id`       | Identifies an event within a trace                       |

Identifiers should be used to define structure, not as ordinary predictive features. Personal names, patient numbers, order numbers, and similar arbitrary identifiers should not receive learned semantic embeddings.

---

## 2.2 Variable dictionary

Each variable needs a schema entry. A variable name and numerical column are not enough.

A recommended variable dictionary is:

```yaml
variable_id: blood_pressure
display_name: Systolic blood pressure
data_type: continuous
unit: mmHg
valid_range: [40, 300]

measurement:
  time_role: pre_treatment
  acquisition_source: clinical_measurement
  repeated: true
  measurement_error_known: false

causal_role:
  role_asserted_by_user: covariate
  manipulable: false
  role_confidence: medium

missingness:
  structural_missingness_possible: false
  censoring_variable: false

availability:
  available_at_decision_time: true
  available_in_all_environments: true
```

Important schema attributes include:

* continuous, binary, categorical, ordinal, count, survival, text, image, or event type;
* unit of measurement;
* possible or valid values;
* whether the variable is measured once or repeatedly;
* whether it is available before treatment;
* whether it may be manipulated;
* whether it represents treatment assignment, treatment receipt, outcome, censoring, selection, or ordinary measurement;
* whether it exists in every environment;
* measurement reliability or known error model;
* whether missingness has a specific meaning.

### Facts and assumptions must be separated

Some metadata are observable facts:

* a measurement was taken at 09:00;
* treatment was randomized by hospital;
* the recorded dosage was 20 mg;
* an edge in a contact network was observed.

Other metadata are scientific assumptions:

* there is no unmeasured confounding;
* an exclusion restriction holds;
* all interference occurs through the supplied network;
* a variable is a valid negative control;
* a proxy measures a particular latent construct.

The input contract should record these separately:

```yaml
assumption:
  statement: no_unmeasured_confounding_given_X
  status: scientist_asserted
  provenance: analysis_protocol
  confidence: medium
  formally_guaranteed_by_design: false
```

This allows the identification engine to state that a result is conditional on an assumption instead of representing the assumption as an observed fact.

---

## 2.3 Treatment and intervention data

The intervention table should distinguish at least:

* assigned treatment;
* treatment actually received;
* intervention time;
* intervention target;
* intervention type;
* dosage or level;
* randomization probability;
* treatment-assignment cluster;
* randomization strata;
* adherence;
* policy or encouragement assignment.

For example:

```text
unit_id
cluster_id
intervention_time
target_variable
assigned_value
received_value
intervention_type
assignment_probability
assignment_stratum
policy_id
```

The distinction between observing (A=a) and intervening with (do(A=a)) must be encoded explicitly. The same numerical value can arise through two very different mechanisms:

[
A_i=1 \quad\text{observationally}
]

versus

[
do(A_i=1).
]

A dedicated intervention mask should therefore accompany every value:

[
i_{utj}=
\begin{cases}
1,&\text{variable (j) was externally intervened on},\
0,&\text{value arose naturally or observationally}.
\end{cases}
]

For longitudinal interventions, the model also needs the complete treatment history and the time at which each decision was made.

---

## 2.4 Assignment-design information

When treatment is randomized, design information is often more valuable than another set of covariates. The processor should accept:

* Bernoulli assignment probability;
* cluster-randomized assignment;
* stratification variables;
* saturation probabilities;
* stepped-wedge timing;
* encouragement assignment;
* known sampling weights;
* policy allocation rules;
* valid randomization or reassignment procedures.

In the interference benchmark, for example, the treatment design includes cluster-level saturation probabilities. This information should be provided directly rather than forcing the model to infer a known randomized design from the realized treatment vector. The corresponding package data contract is described in [causal_interference-detection.md](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/causal_interference-detection.md).

For observational treatment, the input should state that the assignment mechanism is unknown or observational. It should not invent a randomization design.

---

## 2.5 Temporal structure

For temporal data, provide:

* ordered timestamps;
* sequence or entity identifier;
* event durations where applicable;
* irregular time gaps;
* observation windows;
* treatment times;
* censoring or end-of-follow-up times;
* calendar features when scientifically relevant;
* whether an online or offline analysis is intended.

The model should never create a single sequence by sorting and concatenating independent cases. A lagged pair such as

[
(X_{t-1},Y_t)
]

must not cross a patient, machine, process case, or trajectory boundary. The lagged benchmark explicitly requires a `sequence_id` for this reason. See [temporal_split_detection_method.md](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection_method.md).

For irregular observations, both the timestamp and elapsed time should be encoded. “Previous observation” and “one hour earlier” are not equivalent.

---

## 2.6 Network and relational data

For interference, the processor needs an observed unit-to-unit relation graph:

```text
source_unit
target_unit
edge_weight
edge_type
direction
valid_from
valid_until
measurement_quality
```

The relation graph (W) is distinct from the causal graph between variables:

* (W) connects **units**, such as people, hospitals, firms, or process cases;
* the causal graph connects **variables**, such as treatment, exposure, mediator, and outcome.

These two graphs must never be conflated.

The processor should retain:

* raw edge weights;
* normalized edge weights;
* weighted and unweighted degree;
* connected components;
* communities or assignment clusters;
* isolated-node indicators;
* dynamic edge validity;
* uncertainty or confidence in each edge.

The raw matrix may be normalized for neural message passing, but the original values must remain available for interpretation and sensitivity analysis.

### Exposure mappings

An exposure mapping should be supplied or predeclared, for example:

[
G_i=
\frac{\sum_jW_{ij}A_j}
{\sum_jW_{ij}}.
]

Alternative candidate mappings may also be supplied:

* unweighted treated-neighbour fraction;
* weighted fraction;
* two-hop exposure;
* threshold exposure;
* maximum-neighbour exposure;
* community saturation.

Candidate mappings must be specified before examining outcomes. The processor should not search for the exposure mapping that gives the smallest p-value.

---

## 2.7 Environment and population information

Multiple environments are especially valuable for causal learning. An environment can represent:

* hospital;
* factory;
* country;
* process version;
* treatment policy;
* calendar period;
* experiment;
* data source;
* device type;
* batch;
* population subgroup.

Each environment should have metadata such as:

```yaml
environment_id: hospital_04
site_type: tertiary_hospital
data_collection_period: 2025-Q1
sampling_probability: 0.22
treatment_policy: policy_B
variables_available:
  - A
  - X1
  - X2
  - Y
```

Environment indicators should not merely be pooled away. They let the model study invariance, transportability, mechanism changes, and domain shift.

The target population also needs to be declared. An ATE for the observed sample is not necessarily the same as an effect for all eligible patients or all factories.

---

## 2.8 Formal causal query

The same dataset can support several different causal questions. Consequently, the query is part of the input rather than a label attached after processing.

A query should specify:

```yaml
claim_type: interventional

treatment:
  variable: A
  intervention_type: hard
  contrast: [0, 1]

outcome:
  variable: Y
  horizon: 30_days

estimand:
  name: average_treatment_effect
  conditional_on: []
  target_population: eligible_units

interference:
  enabled: false

assumption_set:
  - consistency
  - conditional_exchangeability_given_X
  - positivity

output_request:
  - effect_distribution
  - identification_status
  - overlap_diagnostics
  - sensitivity_bounds
```

For temporal prediction, the claim type may be `predictive_or_Granger`. For structure discovery, it may be `conditional_association_graph`, `CPDAG`, or `PAG`. This prevents the processing layer from silently converting a predictive query into an intervention-level claim.

---

# 3. Minimum input by analysis type

| Analysis                        | Essential model-visible data                     | Strongly recommended metadata                                                 |
| ------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------- |
| Static treatment effect         | (X,A,Y), unit IDs, variable timing, query        | Assignment design, sampling weights, multiple environments, negative controls |
| Regime determination            | Predictors, outcomes, candidate regime variables | Ordinary predictor set, measurement units, independent evaluation data        |
| Network interference            | (X,A,Y,W_{\text{observed}}), cluster IDs         | Assignment design, exposure mapping, alternative maps, edge uncertainty       |
| Latent-group discovery          | Raw indicators (X), outcomes (Y), missingness    | Measurement descriptions, known indicator hints, reliability information      |
| Latent relationship graph       | Raw indicators and outcomes                      | Independent train/test rows, optional partial group knowledge                 |
| Contemporaneous temporal change | (X_t,Y_t), timestamps, sequence IDs              | Time gaps, known interventions, environment changes                           |
| Lagged temporal relation        | (X_t,Y_t), timestamps, sequence IDs              | Candidate lag range, intervention timing, treatment history                   |
| Longitudinal causal effect      | Time-varying (X_t,A_t,Y_t)                       | Decision times, censoring, treatment policy, adherence                        |
| Process-mining causality        | Complete event logs and case outcomes            | Decision points, process version, resources, treatment policy, case relations |

A single static observational matrix without timing, assignment, or environment information can still support descriptive association or conditional-dependence analysis. It generally cannot support unrestricted causal-effect claims.

---

# 4. Processing pipeline

## 4.1 Stage 1: semantic and causal validation

Before any numerical transformation, the processor should perform a causal-data audit.

### Structural checks

It should verify:

* unique and valid unit identifiers;
* monotone timestamps within sequences;
* no lag crossing a sequence boundary;
* graph dimensions match unit IDs;
* no accidental self-edges unless explicitly meaningful;
* cluster IDs are available when treatment is assigned by cluster;
* treatment and outcome columns exist;
* intervention times precede the requested outcome horizon;
* target levels are valid;
* event-log traces are not concatenated.

### Leakage checks

It should detect likely leakage such as:

* post-treatment variables presented as baseline confounders;
* features calculated using future outcomes;
* identifiers that encode outcomes or treatment;
* imputation fitted on evaluation data;
* normalization fitted separately in true temporal segments;
* an exposure mapping selected after examining (Y);
* latent groups estimated using held-out outcomes;
* random row splitting in clustered or longitudinal data.

### Assumption checks

The processor should create an assumption ledger:

```text
Randomization known by design                verified
Treatment precedes outcome                   verified
No unmeasured confounding                    unverified assumption
Interference limited to observed W           scientist asserted
Target exposure levels supported             empirically testable
Selection into sample ignorable              unknown
```

The model can then condition its result on the verified and asserted assumptions separately.

A failed assumption check should not always terminate processing. Some conditions should produce a different output:

* poor overlap → unsupported or partially supported query;
* unknown hidden confounding → sensitivity analysis;
* incomplete network → exposure-mapping uncertainty;
* ambiguous temporal order → partial graph rather than directed graph.

---

## 4.2 Stage 2: split before preprocessing

All splitting must occur before fitting imputation, scaling, encodings, basis functions, or dimensionality reduction.

The unit of splitting depends on the data:

| Data structure               | Correct split unit                                       |
| ---------------------------- | -------------------------------------------------------- |
| Ordinary i.i.d. table        | Unit or individual                                       |
| Repeated measures            | Entire individual trajectory                             |
| Cluster randomization        | Entire assignment cluster                                |
| Network partial interference | Entire independent network cluster                       |
| Process mining               | Entire process case or organizational cluster            |
| Temporal forecasting         | Earlier versus later blocks, or independent trajectories |
| Multi-site generalization    | Entire site or environment                               |
| Latent grouping              | Independent rows or subjects                             |

For a foundation model, the visible part of an episode is the **context set**, while the hidden part is the **target set**:

[
D=D_{\text{context}}\cup D_{\text{target}}.
]

The preprocessing parameters are fitted only on (D_{\text{context}}):

[
\widehat\psi
============

\operatorname{FitPreprocessor}
\left(D_{\text{context}}\right),
]

and then applied unchanged:

[
\widetilde D_{\text{context}}
=============================

T_{\widehat\psi}(D_{\text{context}}),
\qquad
\widetilde D_{\text{target}}
============================

T_{\widehat\psi}(D_{\text{target}}).
]

This matches the package’s discovery-only preprocessing rule.

---

## 4.3 Stage 3: type-specific value processing

The model should use typed encoders rather than converting every variable into an undifferentiated floating-point column.

| Type                | Recommended processing                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Continuous          | Robust centering and scaling; preserve raw unit and transformation parameters                            |
| Binary              | Exact 0/1 or category encoding; no continuous standardization needed                                     |
| Nominal categorical | Context-fitted vocabulary, unknown-level token, category embeddings                                      |
| Ordinal             | Preserve ordering; encode level and normalized ordered position                                          |
| Count               | Preserve raw count; optionally add `log1p` as an auxiliary channel; include exposure or observation time |
| Proportion          | Preserve denominator when available                                                                      |
| Dosage              | Preserve raw dosage and standardized dosage                                                              |
| Survival            | Time-to-event, event indicator, censoring, and competing-risk type                                       |
| Text                | Optional semantic encoder; do not let descriptions override observed evidence                            |
| Timestamp           | Relative time, elapsed time, order, and optional calendar components                                     |
| Event/activity      | Activity embedding plus lifecycle and temporal metadata                                                  |

### Continuous variables

For a continuous variable (j), the processor may calculate:

[
z_{ij}
======

\frac{x_{ij}-\widehat\mu_j}
{\widehat s_j+\epsilon},
]

where (\widehat\mu_j) and (\widehat s_j) are estimated from the context data.

Robust location and scale are useful in heterogeneous real datasets. However:

* the original unit must remain in the variable token;
* transformation parameters must be retained;
* outputs must be transformed back to the original scale;
* aggressive winsorization should not silently remove a true intervention or regime change.

A bounded numerical channel may be used for stable neural computation, but the processor should also provide an outlier indicator and preserve the original value outside the model input store.

### Multiple outcomes

Each outcome should be transformed according to its own type. For multivariate continuous outcomes, training-only standardization prevents a high-variance outcome from dominating all losses. This was important in the latent benchmark, where multiplying one outcome by 25 did not change the results once proper scaling was used.

Effect estimates must subsequently be returned in the original outcome unit.

---

## 4.4 Stage 4: missingness, censoring, and measurement quality

Missingness should not be erased by single-value imputation.

For each value, the encoder should receive at least:

[
\left[
\widetilde x_{utj},
m_{utj},
\Delta t_{utj},
q_{utj}
\right],
]

where:

* (m_{utj}) is the missingness indicator;
* (\Delta t_{utj}) is time since the variable was last observed;
* (q_{utj}) is a measurement-quality flag.

A simple placeholder can be used numerically when a value is missing, but the missingness mask ensures that the placeholder is not interpreted as a genuine measurement.

The system should distinguish:

* ordinary missing measurement;
* structural absence of a variable in an environment;
* treatment non-adherence;
* right censoring;
* loss to follow-up;
* outcome not yet observed;
* measurement below detection limit;
* value suppressed for privacy;
* sensor failure.

Missingness that depends on treatment, outcome, or latent health status may itself be part of the causal process. It should be represented through a selection or missingness mechanism rather than treated solely as a preprocessing nuisance.

---

## 4.5 Stage 5: temporal construction

Temporal preprocessing should be performed only after sorting and validating sequence boundaries.

### Contemporaneous mechanism data

For a same-time conditional query, the model receives:

[
(X_t,Y_t,\text{timestamp}_t,\text{sequence}_t).
]

No lagged variables should be silently introduced unless requested.

### Lagged data

For a VARX-like query, the processor constructs:

[
Q_t=
\left[
Y_{t-1},
X_{t-1},
\Delta t_t
\right],
\qquad
R_t=Y_t.
]

The first row of each sequence is excluded for a one-lag analysis. For irregular timing, the elapsed interval is included.

### Scaling rule

One transformation should be used for the complete discovery sequence or training prefix. Scaling separately inside candidate segments can:

* erase intercept changes;
* suppress variance changes;
* make segment costs incomparable;
* create or remove apparent coefficient changes.

Both temporal benchmark documents impose this principle. See [temporal_split_detection.md](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection.md) and [temporal_split_detection_method.md](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection_method.md).

### Offline versus online mode

The processor should explicitly mark the operating mode:

* **offline:** transformations may use the full observed sequence, but not true breakpoints;
* **online:** transformations are fitted on the available prefix and cannot use future observations.

Results from these two modes should not be compared as though they use the same information.

---

## 4.6 Stage 6: network processing

The network processor should create at least two representations:

1. the raw relation graph;
2. normalized message-passing edges.

For node (i), a normalized weight may be

[
\widetilde W_{ij}
=================

\frac{W_{ij}}
{\sum_k W_{ik}},
]

but (W_{ij}), degree, and denominator should remain available.

For each prespecified exposure mapping (f), the processor computes:

[
G_i^{(f)}=f(A_{-i},W_i).
]

It should then create query-specific support summaries such as the observed distribution of

[
(A_i,G_i^{(f)},X_i).
]

Important rules are:

* exposure is computed from the complete observed treatment allocation;
* outcomes are not used to select the mapping;
* treatment assignment clusters remain intact;
* edge uncertainty is retained;
* isolates are handled through a predeclared rule;
* network relations after treatment should not be treated as baseline relations without justification.

The processor should also keep alternative mappings as separate channels, not average them into one exposure.

---

## 4.7 Stage 7: latent measurement processing

The model should normally receive the individual observed indicators, not a precomputed PCA or PLS representation.

Premature dimensionality reduction can:

* force orthogonality between latent scores;
* erase relationships the causal model is intended to discover;
* commit to incorrect hard groups;
* hide measurement uncertainty;
* propagate one upstream error into every downstream result.

For observed variables (X_1,\ldots,X_P), the processor supplies:

* standardized indicator values;
* missingness masks;
* variable descriptions;
* optional soft group hints;
* measurement reliability if known;
* outcome relevance information only through the observed outcomes, not oracle labels.

Known group information can be supplied as a soft or hard constraint, but estimated groups should remain probabilistic.

This is particularly important because the package’s latent graph experiment found meaningful degradation from true scores to reconstructed scores and then from known groups to estimated groups. The processing layer should preserve all three levels for audit purposes. See [latent_variable_determination2.md](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/latent_variable_determination2.md).

---

# 5. Tensorization and token construction

After preprocessing, a dense episode can be represented as

[
\widetilde X
\in
\mathbb R^{U\times T\times P\times C_v},
]

where:

* (U): units;
* (T): time points;
* (P): variables;
* (C_v): value channels.

Possible value channels are:

[
\begin{aligned}
C_v={&
\text{standardized value},
\text{missing mask},
\text{intervention mask},
\text{censoring mask},\
&\text{measurement age},
\text{quality flag},
\text{raw-scale summary}
}.
\end{aligned}
]

The model also receives separate token sets.

### Variable tokens

Each variable token contains:

* type;
* unit;
* possible range;
* pre/post-treatment timing;
* queried role;
* manipulability;
* measurement source;
* environment availability;
* semantic description if available.

### Unit or sequence tokens

These represent:

* unit-level attributes;
* cluster membership;
* site;
* sequence;
* sampling weight;
* assignment stratum.

### Time tokens

These include:

* relative time;
* time since previous observation;
* time from treatment;
* position within sequence;
* calendar information when relevant.

### Relation tokens

These represent:

* graph edges;
* edge types;
* edge weights;
* direction;
* validity interval;
* edge confidence.

### Environment tokens

These describe:

* site;
* policy;
* experiment;
* process version;
* intervention regime;
* data source.

### Query tokens

These identify:

* treatment;
* outcome;
* contrast;
* target population;
* estimand;
* time horizon;
* interference mapping;
* claim type;
* assumption set.

The same preprocessed dataset can therefore be used for multiple queries by changing the query and role tokens without repeating every numerical preprocessing step.

---

# 6. Processing large datasets

A real dataset may be much larger than the model context. Uniform row sampling is often unsafe because it can remove:

* rare treatments;
* small regimes;
* short transition periods;
* uncommon exposure levels;
* low-frequency outcomes;
* small network communities;
* underrepresented environments.

The processor should use structure-aware context construction.

## Static data

Sample with stratification over:

* treatment;
* outcome type;
* environment;
* important covariates;
* propensity or overlap region;
* rare categories.

## Network data

Sample:

* complete independent clusters;
* complete ego networks;
* connected subgraphs;
* degree strata;
* different exposure levels.

A cluster should not be split arbitrarily between context blocks when cluster-level dependence matters.

## Temporal data

Use:

* contiguous windows;
* complete trajectories;
* windows surrounding potential transitions;
* blocks representing different environments;
* multi-resolution summaries for long history.

## Process-mining data

Use complete traces or valid trace prefixes rather than isolated random events.

### Hierarchical processing

A scalable design can use:

1. local encoders over blocks, trajectories, clusters, or traces;
2. summary tokens from each block;
3. a global encoder over the summaries;
4. query-specific retrieval of detailed blocks.

The processor should retain sampling weights so that the model knows whether a context is representative or deliberately enriched for rare cases.

Several independently sampled views of the same dataset can be processed and their causal-world posteriors aggregated. Disagreement between views becomes a stability diagnostic.

---

# 7. Pretraining-data processing

## 7.1 Generate complete causal worlds first

For synthetic pretraining, the generator first samples a complete causal world:

[
\mathcal W=
(G,\mathcal M,U,L,R,F,S),
]

where:

* (G): causal graph;
* (\mathcal M): structural mechanisms;
* (U): exogenous disturbances;
* (L): latent variables and measurement system;
* (R): temporal regimes;
* (F): network exposure mapping;
* (S): selection and assignment mechanisms.

Only after defining the causal world should it generate observations.

This order prevents the generator from defining “causal truth” retrospectively from statistical patterns.

---

## 7.2 Sample multiple data regimes from the same world

A training episode should ideally include several related datasets:

[
D_{\mathcal W}
==============

\left{
D_{\text{obs}}^{(1)},\ldots,D_{\text{obs}}^{(E)},
D_{\text{int}}^{(1)},\ldots,D_{\text{int}}^{(J)}
\right}.
]

These may represent:

* observational environments;
* randomized experiments;
* soft interventions;
* policy changes;
* different sites;
* temporal regimes;
* different network saturations;
* missing-variable environments.

Some interventional datasets may be visible as context, while others become prediction targets.

This teaches the model the distinction between:

* observational conditional distributions;
* intervention distributions;
* transport across environments;
* mechanism changes;
* sampling differences.

---

## 7.3 Context–target construction

A pretraining example could be:

[
\left(
\mathcal E_{\text{context}},
Q,
T_{\text{oracle}}
\right).
]

The context includes only data that would be visible in a real application. The target may include:

* held-out observations;
* interventional outcome distributions;
* effect values;
* graph endpoints;
* latent assignments;
* regime boundaries;
* exposure mappings;
* identification status;
* overlap status;
* sensitivity bounds;
* calibration labels.

Different losses are activated depending on which oracle information is available.

For counterfactual training, the simulator may retain the same exogenous disturbance (U_i) across alternative interventions:

[
Y_i(a)=f_Y(a,X_i,U_i).
]

That shared-noise construction is necessary for individual counterfactual targets. Independent intervention samples are sufficient for population intervention distributions but not for unit-level counterfactual supervision.

---

## 7.4 Identification labels should be generated formally

The identification label should not be based on whether a neural estimator happened to perform well.

For every sampled world and query, a formal engine should determine:

```text
identified
partially_identified
not_identified
unsupported
contradictory_assumptions
unknown_to_engine
```

The training target can additionally specify:

* valid adjustment sets;
* valid instruments;
* front-door variables;
* identified functional;
* sensitivity parameter;
* lower and upper bounds;
* assumptions required for identification.

This is particularly important for hidden-confounding episodes. The correct target is often “not point identified,” not an oracle point effect that could never be obtained from the model-visible data.

---

## 7.5 Include matched negative and contrastive episodes

The pretraining set should deliberately include pairs that are statistically similar but causally different:

* one global nonlinear mechanism versus two genuine regimes;
* observed association versus intervention effect;
* direct treatment effect versus spillover;
* correct versus misspecified exposure mapping;
* intercept change versus slope change;
* intercept change versus (B)-change in VARX;
* weak measurement versus incorrect feature support;
* hidden confounding versus observed confounding;
* unsupported versus supported treatment contrasts;
* (X)-distribution change versus (P(Y\mid X)) change.

These pairs reduce the risk that the model learns predictive shortcuts.

The global-nonlinearity pair is especially important because the ZIP experiment showed that misspecification could produce a false split in every run while greatly improving predictive loss.

---

## 7.6 Training augmentations

Safe augmentations include:

* row permutation for exchangeable data;
* variable permutation with corresponding metadata and target permutation;
* random missingness masks;
* random hiding of partial graph knowledge;
* unit conversion with updated scale metadata;
* sampling smaller context sets;
* removal of some environments;
* adding irrelevant variables;
* adding correlated proxies;
* changing variable names while preserving schema;
* partial network-edge observation;
* different query selections from the same world.

Potentially unsafe augmentations include:

* shuffling temporal rows;
* breaking clusters;
* randomly rewiring an interference graph without changing the truth;
* mixing pre- and post-treatment variables;
* oversampling unsupported treatment–exposure combinations and then claiming positivity;
* arbitrarily quantile-normalizing variables when parameter-scale changes are the target.

---

# 8. Real-data inference processing

The real-data path is similar to pretraining, except that no oracle truth exists.

A robust inference pipeline is:

```text
Raw data
   ↓
Schema and causal-role validation
   ↓
Unit / cluster / sequence / environment identification
   ↓
Context and audit split
   ↓
Context-only transformation fitting
   ↓
Typed value, missingness, time, intervention, and relation encoding
   ↓
Formal query construction
   ↓
Support and data-quality summaries
   ↓
Foundation-model inference
   ↓
Formal identification and sensitivity analysis
   ↓
Held-out checks, stability views, and calibration
   ↓
De-standardized causal result or abstention
```

The processor should return a structured audit object alongside the model tensors:

```yaml
data_audit:
  units: 12450
  sequences: 12450
  environments: 6
  treatment_precedes_outcome: true
  post_treatment_adjustment_candidates_removed: 3
  unknown_variable_types: []
  missingness_rate:
    X1: 0.04
    X2: 0.21
  overlap:
    status: weak
    affected_population_fraction: 0.18
  temporal_boundary_violations: 0
  graph_isolates: 12
  assumptions_unverified:
    - no_unmeasured_confounding
    - complete_interference_network
```

This audit becomes part of the reliability layer and should be stored with every result.

---

# 9. Specific processing for process-mining event logs

For process mining, the natural input is not a fixed matrix but an event log.

## 9.1 Required event-log data

A useful schema is:

```text
case_id
event_id
activity
start_timestamp
end_timestamp
lifecycle_transition
resource
organizational_unit
event_attributes
case_attributes
process_version
site
```

The causal question additionally requires:

* the decision or intervention point;
* treatment assignment or action taken;
* variables available before that decision;
* the outcome and its evaluation horizon;
* case censoring or incomplete traces;
* treatment policy or assignment mechanism;
* resource or case relations when interference is possible.

## 9.2 Define causal decision points

A treatment in a process is usually attached to a decision point, for example:

* assigning a resource;
* skipping an activity;
* escalating a case;
* selecting a route;
* performing an additional test;
* changing a deadline;
* applying a policy.

For each decision point (d), the processor constructs the observed history:

[
H_{id}
======

{
\text{events and attributes available before decision }d
}.
]

Treatment is:

[
A_{id}=\text{decision taken at }d,
]

and outcome is defined over a future horizon:

[
Y_{id}^{(h)}
============

\text{outcome within horizon }h.
]

No events occurring after the decision may be included in (H_{id}).

## 9.3 Two representations

The processor can produce two complementary views.

### Event-token view

Each event becomes a token containing:

* activity;
* relative time;
* duration;
* resource;
* event attributes;
* position in trace;
* time from decision;
* process version.

This preserves detailed trace structure.

### Decision-panel view

Each eligible decision point becomes a longitudinal row containing:

* pre-decision state;
* treatment;
* subsequent outcome;
* prior treatment history;
* case and environment identifiers.

This is useful for formal treatment-effect estimators and longitudinal identification.

The model can encode the event-token history while the identification and estimation layer operates on the decision-panel representation.

## 9.4 Boundary and leakage rules

The processor must not:

* concatenate unrelated cases;
* construct lags across case boundaries;
* include future activities as baseline covariates;
* classify a process variant using the final completed trace when predicting an earlier decision;
* use realized case duration as a baseline predictor;
* treat incomplete traces as completed cases;
* ignore time-dependent eligibility for a decision.

## 9.5 Relations and interference in processes

Relations may arise through:

* shared resources;
* common queues;
* machine congestion;
* team membership;
* shared suppliers;
* work transferred between cases;
* capacity constraints.

These relations create a unit graph (W). For example, one case’s resource assignment may affect another case’s waiting time. The network and exposure-processing route should then be activated rather than assuming independent cases.

A discovered process model can be supplied as soft background knowledge—for example, impossible activity orderings—but it should not automatically be treated as the true causal graph.

---

# 10. Mapping the ZIP benchmarks into this data contract

| Benchmark                       | Model-visible input                                                              | Training/evaluation-only truth                                    |
| ------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Observed regimes                | (X,Y), variable schema, candidate regime-variable mask, ordinary predictor mask  | True regime variables, tree, regime IDs, parameters               |
| Interference                    | (X,A,Y,W_{\text{observed}}), cluster IDs, assignment design, exposure candidates | (W_{\text{true}}), hidden confounders, true mapping, true effects |
| Outcome-relevant latent groups  | (X,Y), variable types and missingness                                            | (Z,\Lambda,\Gamma), true groups                                   |
| Latent relationship graph       | (X,Y), optional group hints                                                      | True latent scores, precision matrix, graph edges                 |
| Contemporaneous temporal splits | (X_t,Y_t), timestamps, sequence IDs                                              | True segments, breakpoints, coefficients                          |
| Lagged VARX splits              | (X_t,Y_t), timestamps, sequence IDs, lag specification                           | True (A,B,c), parameter-specific breakpoints                      |

The detailed package contracts are available in:

* [Observed-regime data specification](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/causal_model_determination.md)
* [Interference data specification](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/causal_interference-detection.md)
* [Latent-group data specification](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/latent_variable_determination.md)
* [Latent-graph data specification](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/latent_variable_determination2.md)
* [Contemporaneous temporal specification](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection.md)
* [Lagged temporal specification](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection_method.md)

---

# 11. Non-negotiable processing safeguards

The following rules should be built into the data layer rather than left to analyst discipline:

1. **Split before preprocessing.**
2. **Fit transformations only on context or training data.**
3. **Never create lags across sequence or case boundaries.**
4. **Never split treatment-assignment clusters across folds.**
5. **Do not use post-treatment information as baseline adjustment data.**
6. **Do not standardize temporal segments separately.**
7. **Do not select an exposure mapping using outcomes.**
8. **Do not collapse latent indicators into hard groups before uncertainty is represented.**
9. **Do not treat imputed values as observed values; retain missingness masks.**
10. **Do not treat observed treatment as equivalent to an intervention.**
11. **Do not make synthetic rows to manufacture overlap.**
12. **Do not expose simulator truth to the encoder.**
13. **Do not interpret unit relation graphs as causal-variable graphs.**
14. **Do not use predictive holdout performance as the only causal validation criterion.**
15. **Return original-scale effects and preserve all transformation state.**

---

# Recommended final interface

The preprocessing layer should ultimately return something similar to:

```python
CausalEpisodeBatch(
    values=...,                 # typed, transformed observed values
    observed_mask=...,
    intervention_mask=...,
    censoring_mask=...,
    quality_mask=...,

    variable_metadata=...,
    unit_metadata=...,
    timestamps=...,
    sequence_ids=...,
    cluster_ids=...,
    environment_ids=...,

    relation_edges=...,
    relation_attributes=...,
    candidate_exposures=...,

    assignment_metadata=...,
    background_constraints=...,
    assumption_ledger=...,
    query_tokens=...,

    transform_state=...,        # needed for inverse transformation
    sampling_weights=...,
    data_audit=...
)
```

For synthetic pretraining, a separate object is attached:

```python
OracleTargets(
    causal_graph=...,
    latent_structure=...,
    regimes=...,
    mechanisms=...,
    intervention_distributions=...,
    causal_effects=...,
    identification_status=...,
    sensitivity_bounds=...,
    support_truth=...
)
```

The most important conceptual distinction is therefore:

[
\boxed{
\text{observed values}
\neq
\text{design metadata}
\neq
\text{scientific assumptions}
\neq
\text{oracle causal truth}
}
]

A reliable causality-specific foundation model must preserve all four categories separately. The neural encoder can learn from the observed values and design metadata, the identification engine can condition on explicit assumptions, and oracle truth can supervise synthetic pretraining without ever leaking into real-world inference.
