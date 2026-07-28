## The main principle: define a causal observation before extracting features

For causal analysis, a “good feature table” is not primarily the table whose columns best predict the outcome. It is a table in which every row has a clear **unit, reference time, treatment or exposure, and future outcome**.

For case (i) at a predefined landmark (\tau_{i\ell}), construct

[
X_{i\ell}
=========

\phi!\left(
L_i^{\leq \tau_{i\ell}},
L_{-i}^{\leq \tau_{i\ell}}
\right),
\qquad
Y_{i\ell}
=========

\psi!\left(
L_i^{> \tau_{i\ell}}
\right),
]

where:

* (L_i^{\leq \tau_{i\ell}}) is the prefix of the case observed by the landmark;
* (L_{-i}^{\leq \tau_{i\ell}}) is contextual information from other cases available at that time, such as workload;
* (\phi) is the feature-extraction function;
* (\psi) defines one or more future outcomes.

The process-mining literature calls a closely related construction a **situation feature table**: a situation is a trace prefix ending at a meaningful point, and only information available in that prefix is extracted. This was explicitly proposed to enforce the requirement that potential causes occur before the target feature. 

## 1. Choose the correct type of row

The row definition depends on which of your causal challenges you are studying.

| Row type                           | Construction                                                                              | Suitable question                                                           |
| ---------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **One row per case**               | Baseline or early-case features, followed by final outcomes                               | Does a case attribute determine the causal regime?                          |
| **One row per landmark or prefix** | Features from the case history up to a predefined activity, event number, or elapsed time | Does an intervention at this process state affect a later outcome?          |
| **One row per event transition**   | History through event (j), treatment/action at (j), outcome at (j+1) or a later horizon   | How does the current state affect the future of the same case?              |
| **One row per calendar window**    | Aggregates during window (t), outcomes during window (t+1)                                | Does the state of the process in one period influence the following period? |

For your work, **landmark or situation tables are probably the most generally useful representation**.

A landmark should normally be fixed independently of the eventual outcome, for example:

* immediately before a designated decision activity;
* after the first (k) events;
* every 24 hours after case creation;
* when a specified milestone activity is reached.

Do not use “the point three events before failure” as a landmark, because the landmark itself then depends on future outcome information.

For temporal influence, the immediately preceding row is meaningful only if rows describe consecutive states of the same evolving system. Different cases sorted by their start times do not generally satisfy this. For process-level influence, aggregate into fixed calendar windows instead.

## 2. Feature-extraction methods for a conventional event log

Standard process-monitoring work distinguishes last-state, aggregation, index-based, case-level, inter-case and embedding-based encodings. A broader trace-encoding benchmark identified many variants of these families, emphasizing that they differ in expressivity, scalability and interpretability. ([Springer][1])

### Case-level attributes

These are attributes known at case creation or before the relevant treatment:

* customer or application category;
* requested amount;
* channel;
* product type;
* location;
* initial priority;
* initial risk score.

They are particularly suitable as candidate **confounders** or **regime variables**, provided that their values are genuinely fixed before the treatment and outcome.

A case-level value copied from the final event is not automatically a baseline variable. Its recording time and meaning must be checked.

### Last-state or last-(k) encoding

Extract the most recent process state:

* current activity;
* previous (k) activities;
* last resource or role;
* most recent value of each dynamic attribute;
* time since the previous event;
* most recent queue or workload state.

This provides a relatively compact description of the current process state. It is useful when the recent history is more relevant than distant events.

Its limitation is that it can omit earlier rework, repetitions and long-range dependencies.

### Aggregation encoding

Summarize everything observed in the prefix:

* number or relative frequency of each activity;
* number of occurrences of each resource or role;
* first, last, minimum, maximum, mean and standard deviation of numerical attributes;
* number of attribute-value changes;
* number of rework loops;
* number of distinct resources;
* number of resource handovers;
* elapsed time;
* average and maximum event gap.

Aggregation produces a fixed-size and interpretable table even when traces have different lengths. ProLift, a causal-process-monitoring implementation based on uplift trees, uses aggregation or last-state encoding for event-level features. 

The central disadvantage is loss of ordering. Two prefixes containing (A,B,C) and (C,B,A) can have the same activity counts.

### Directly-follows and (n)-gram features

Generate counts or indicators such as:

[
#(A\rightarrow B), \qquad
#(A\rightarrow B\rightarrow C).
]

These retain some ordering while remaining tabular. Useful features include:

* activity bigram and trigram counts;
* occurrence of selected subsequences;
* repeated-loop patterns;
* first occurrence of a pattern;
* time elapsed between two selected activities.

This is often a good compromise between pure aggregation and full index-based encoding. Rare patterns should be pruned using a minimum frequency chosen independently of the outcome, or selected only within training folds.

### Index-based encoding

Create columns for every position:

[
\text{activity}_1,\text{activity}_2,\ldots,\text{activity}_k,
]

and similarly for resources, timestamps and event attributes.

This retains the sequence order, but produces a large and sparse table. It also fragments a semantic variable across positions: “amount at event 2” and “amount at event 5” become different columns, even though they may represent the same underlying process quantity.

Index-based encoding is most defensible when:

* all rows use the same landmark or prefix length;
* maximum (k) is moderate;
* the exact event position has a stable process interpretation.

### Process-model-based features

A discovered or normative process model can provide:

* current Petri-net marking or process state;
* enabled activities;
* number and type of deviations;
* prefix-alignment cost;
* missing or remaining mandatory activities;
* number of unexpected repetitions;
* reached process phase or milestone.

These features can capture behavior more compactly than thousands of sequence indicators.

For temporal evaluation, the process model must be learned using only the training period. Discovering it from the complete log and then extracting features for earlier rows would leak future process behavior into the table.

### Time, resource and operational-context features

The general event-log feature framework of de Leoni and colleagues distinguishes control-flow, data, resource, time and conformance perspectives. It includes derived characteristics such as workload, resource information, activity duration, elapsed time and waiting time. 

Useful variables include:

* elapsed time since case creation;
* time since the previous event;
* service and waiting time, when lifecycle information allows them to be distinguished;
* hour, weekday, month and season;
* number of currently active cases;
* incoming-case rate;
* backlog or queue size;
* workload of the current resource;
* historical experience of the resource;
* recent process-outcome rate, calculated strictly from already resolved cases;
* staffing level and system availability.

A timestamp difference between consecutive completion events is usually an **event gap**, not necessarily the activity’s processing duration. Calling it a duration requires valid start/complete lifecycle semantics.

### Learned sequence representations

Autoencoders, recurrent networks and transformers can transform a prefix into a fixed-dimensional embedding. Such representations can be powerful nuisance-model inputs, but they are less suitable as the main variables in an interpretable causal model because individual embedding dimensions generally have no stable process meaning. The predictive-process-monitoring literature similarly notes the transparency loss associated with embeddings. ([Springer][1])

For causal work, an embedding should preferably be:

* computed only from the prefix;
* trained without access to future cases in the test period;
* cross-fitted when trained on the same observations used for effect estimation;
* used as an additional nuisance representation rather than as a replacement for all interpretable variables.

A supervised embedding trained to predict (Y) is especially unsuitable for deciding which variables are causes, confounders or moderators.

## 3. A practical default representation

For your setting, I would start with a **hybrid situation encoding** rather than trying to identify one universally best encoding.

For every case and predefined landmark, include:

[
X=
[
X^{\mathrm{baseline}},
X^{\mathrm{last}},
X^{\mathrm{aggregate}},
X^{\mathrm{transition}},
X^{\mathrm{time}},
X^{\mathrm{resource}},
X^{\mathrm{context}},
X^{\mathrm{model}}
].
]

A concrete schema could look like:

```text
case_id                         identifier, never a predictor
landmark_time                   reference time

baseline_customer_type
baseline_amount
baseline_channel
baseline_priority

last_activity_1
last_activity_2
last_resource
last_dynamic_attribute_value

count_activity_A
count_activity_B
count_rework
count_A_to_B
count_B_to_C
number_distinct_activities

elapsed_time
last_event_gap
mean_event_gap
maximum_event_gap

number_unique_resources
number_handovers
current_resource_workload
number_active_cases

current_process_phase
prefix_alignment_cost
number_deviations

treatment_or_action
outcome_1 ... outcome_M
censoring_indicator
```

For an initial implementation, use:

* activity and frequent directly-follows counts;
* the last three activities;
* last values and summary statistics for dynamic attributes;
* elapsed time and recent time gaps;
* rework, handover and unique-resource counts;
* workload and backlog at the landmark;
* process phase or marking, when available.

This representation remains interpretable enough to diagnose why a regime split or temporal change was detected.

## 4. Should you select features highly correlated with the outcomes?

**Not as the main feature-selection rule for causal analysis.**

Process-mining methods designed for root-cause or outcome prediction frequently identify associations with outcomes, but association, support or predictive confidence does not establish a causal effect. This distinction is also made explicitly in causal-process-mining work that first mines associated treatment rules and subsequently applies causal adjustment. 

### A model-determining feature can have zero outcome correlation

Consider independent variables

[
A\in{-1,+1}, \qquad Z\in{0,1},
]

with equal probabilities, and

[
Y=(1-2Z)A.
]

Then:

[
Z=0 \Rightarrow Y=A,
\qquad
Z=1 \Rightarrow Y=-A.
]

Thus (Z) completely determines which causal mechanism applies. However,

[
\operatorname{Corr}(Z,Y)=0
]

and

[
\operatorname{Corr}(A,Y)=0.
]

Indeed, (A) and (Y) are marginally independent, and (Z) and (Y) are marginally independent. Consequently, Pearson correlation, Spearman correlation and even univariate mutual information screening would discard both variables.

This is precisely relevant to the model-based recursive partitioning problem in your repository: a variable that changes the coefficients or sign of a mechanism need not directly predict the outcome.

### Highly correlated variables may be causally inappropriate

A feature can be highly associated with the outcome because it is:

* a consequence of the treatment;
* a mediator through which the treatment operates;
* a consequence or proxy of the outcome;
* a collider influenced by two other variables;
* a record generated only because an adverse outcome occurred.

For example, if

[
A\rightarrow M\rightarrow Y,
]

then (M) will normally be associated with (Y), but controlling for (M) blocks part of the total effect of (A). Conditioning on colliders can instead open noncausal paths and create selection bias. 

Event logs contain many likely examples:

* final activity;
* final status;
* number of events in the completed trace;
* total case duration;
* escalation activity performed after difficulties arise;
* final conformance cost;
* a complaint-handling event when complaint is the outcome.

These may be excellent outcome predictors and terrible adjustment variables.

### Weakly outcome-associated features can still be important

A variable may be weakly associated with (Y) but strongly associated with treatment assignment. Omitting it can leave substantial confounding. Similarly, a pure effect modifier may have little or no marginal association with (Y).

Therefore, outcome-only Lasso, correlation screening, SHAP ranking or random-forest importance is not sufficient for choosing causal controls.

Your sparse-PLS approach has the same conceptual limitation for this purpose: sparse PLS deliberately constructs components that covary with (Y). It may be useful for prediction or supervised dimension reduction, but it can merge confounders, mediators, colliders and regime variables into a single latent component. It should not define the causal adjustment set.

## 5. What to use instead of outcome correlation

### First perform causal-admissibility filtering

This stage should not use the outcome values. Remove or flag features that are:

* observed after the relevant treatment or decision;
* computed from the completed trace;
* deterministic versions of the outcome;
* identifiers or almost-unique values;
* duplicates or deterministic transformations of other columns;
* unavailable operationally at the landmark;
* semantically inconsistent across time;
* almost always missing or constant.

Feature-feature correlation can be useful here for detecting redundant columns. That is different from retaining columns because they correlate with (Y).

### Assign each feature a role

Maintain metadata alongside the table:

```text
feature name
source attribute/activity
earliest availability time
latest event used in its calculation
baseline / treatment / mediator / moderator / context / outcome
modifiable or non-modifiable
aggregation rule
```

This is particularly important because temporal precedence over (Y) is necessary but not sufficient. A feature measured before (Y) but after treatment may still be a mediator.

### Use goal-specific selection

| Goal                                         | Appropriate selection strategy                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Estimate the effect of a specified treatment | Select a causally sufficient set of pre-treatment covariates using process knowledge or a DAG                             |
| High-dimensional treatment-effect estimation | Use treatment-and-outcome **double selection**, doubly robust estimation or double machine learning with cross-fitting    |
| Find model-determining attributes            | Give all plausible pre-outcome moderators to MOB or a causal tree; rank them by stability, not marginal correlation       |
| Detect temporal mechanism changes            | Fit the conditional model using plausible pre-outcome features and detect changes in its parameters or residual mechanism |
| Discover outcome parents                     | Use temporally constrained conditional-independence or invariance methods rather than univariate screening                |
| Pure outcome prediction                      | Outcome correlation and predictive feature importance are acceptable inside the training data                             |

A practical confounder-selection principle is to retain plausible pre-treatment causes of the treatment, the outcome, or both, while avoiding known mediators and inappropriate controls. ([Springer][2])

When there are many candidate controls, post-double-selection selects predictors of the treatment and predictors of the outcome and takes their union. This is safer than outcome-only selection because it protects variables that strongly determine treatment assignment even if their outcome association is modest. ([arXiv][3])

## 6. Specific recommendations for your three settings

### The causal model depends on an observed attribute

Construct one case or landmark row and retain all plausible **pre-outcome candidate context variables**. Do not filter them according to (\operatorname{Corr}(X_j,Y)).

Apply your model-based recursive partitioning method to test whether structural parameters vary with each candidate attribute. Validate selected split variables using:

* held-out likelihood or error improvement;
* permutation tests including the split search;
* bootstrap selection frequency;
* adequate subgroup sizes.

### The causal model changes over calendar time

Use a situation table sorted by the landmark timestamp, not by the completion timestamp when that would reveal future information.

Include known seasonal and operational variables such as weekday, month, workload and policy periods in the within-segment model. Otherwise, an unmodelled seasonal pattern may appear as a causal-model break.

Use temporal validation rather than random row splitting. If multiple prefixes originate from one case, keep the complete case in one fold.

### An earlier period influences the next period

Use one of two representations:

1. **Within-case longitudinal table**

[
X_{i,j}
=======

\phi(e_{i,1},\ldots,e_{i,j}),
\qquad
Y_{i,j+h}
=========

\psi(e_{i,j+1},\ldots,e_{i,j+h}).
]

2. **Process-level window table**

[
X_t
===

\phi(\text{information observable by the end of window }t),
]

[
Y_{t+1}
=======

\psi(\text{outcomes in window }t+1).
]

Do not interpret the features of case (i-1) as causes of the outcome of case (i) merely because their start times are adjacent.

## 7. Important validation rules

When creating multiple prefix rows per case:

* split training and testing data by **case**, not by row;
* use case-clustered uncertainty estimates or a longitudinal model;
* do not let an early prefix of a case enter training while a later prefix of the same case enters testing.

When evaluating temporal changes:

* train on earlier periods and test on later periods;
* calculate encodings, frequent patterns, process models and scaling parameters from training data only;
* report sensitivity to alternative landmarks, horizons and encodings.

For unfinished cases, include censoring information or use an analysis that accounts for right censoring. Restricting the data to cases that happened to finish within the observation period can produce a selected sample.

## Bottom line

The most defensible starting point for your project is a **situation or landmark feature table** with a hybrid of interpretable aggregation, last-state, transition, temporal, resource and process-state features.

Use outcome correlation only for descriptive exploration, redundancy checks or a purely predictive baseline. It should **not** decide which variables enter a causal model, because it can simultaneously select mediators and colliders while discarding confounders and the exact regime variables your current research aims to discover.

[1]: https://link.springer.com/article/10.1007/s44311-024-00002-4 "Predictive process monitoring: concepts, challenges, and future research directions | Process Science | Springer Nature Link"
[2]: https://link.springer.com/article/10.1007/s10654-019-00494-6?utm_source=chatgpt.com "Principles of confounder selection - Springer Nature"
[3]: https://arxiv.org/abs/1201.0224?utm_source=chatgpt.com "Inference on Treatment Effects After Selection Amongst High-Dimensional Controls"
