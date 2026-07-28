# Process mining as the “glue” between event data and a causal foundation model

Process mining should not be treated merely as one downstream application of the proposed causal foundation model. It can play a more fundamental role:

[
\boxed{
\text{information systems}
\rightarrow
\text{process event data}
\rightarrow
\text{causal episodes}
\rightarrow
\text{causal foundation model}
\rightarrow
\text{causal process understanding}
}
]

Event data provide a common representation in which temporal behavior, operational decisions, organizational resources, interacting business objects, process variants, and outcomes can be connected. Process-mining techniques can then supply structural information—such as control-flow order, concurrency, decision points, variants, deviations, and object interactions—that helps organize the causal problem. The foundation model adds what conventional process mining generally lacks: explicit reasoning about interventions, counterfactuals, identification, confounding, overlap, and uncertainty.

Conventional process mining is primarily concerned with discovering, checking, and enhancing process behavior from event data. It is therefore well suited to answer questions such as **what happened**, **in which order**, **how often**, and **where delays or deviations occurred**. It does not, by itself, establish that one observed activity or process feature caused another. This distinction is increasingly recognized in causal process-mining research: directly-follows and temporal-precedence relations can guide causal analysis, but they are not equivalent to causal relations. ([Springer Link][1])

The proposed model would connect these two levels:

* **process mining supplies the behavioral and relational structure of the process;**
* **the causal foundation model determines which causal questions can be answered and estimates the effects of well-defined interventions;**
* **the results are projected back onto process models, traces, variants, resources, and objects so that they remain understandable to process analysts.**

---

## 1. From raw event data to causal-model input

## 1.1 Case-centric event logs

In the traditional case-centric setting, an event can be represented as

[
e=
\left(
\text{event ID},
\text{case ID},
\text{activity},
t_{\text{start}},
t_{\text{complete}},
\text{resource},
\mathbf{x}
\right),
]

where (\mathbf{x}) contains event attributes.

A case trace is an ordered sequence

[
\sigma_i=
\langle
e_{i1},e_{i2},\ldots,e_{im_i}
\rangle,
]

and an event log is a collection of traces

[
L={\sigma_1,\ldots,\sigma_N}.
]

For example, an order-fulfilment trace might be:

```text
Create order
Check inventory
Approve order
Pick items
Pack items
Dispatch shipment
Deliver order
```

Each event can additionally contain:

* the employee or system performing the activity;
* the warehouse or organizational unit;
* the activity duration;
* queue length at the event time;
* order value;
* product type;
* priority;
* costs;
* process version;
* customer or supplier attributes.

The event log should not immediately be converted into one static row per completed case. The original sequence, timestamps, resource assignments, and intermediate states should be retained. A static case table is only one possible projection of the richer event data.

---

## 1.2 Object-centric event data

For realistic operational processes, an object-centric representation is often more appropriate. One event may relate simultaneously to several objects, such as:

* an order;
* several order items;
* a shipment;
* an invoice;
* a supplier;
* a warehouse;
* a machine;
* a customer;
* a resource.

OCEL 2.0 explicitly represents events, objects of different types, event-to-object relationships, object-to-object relationships, and changing object attributes. This avoids forcing every event into one artificial case notion. ([OCEL 2.0][2])

For example:

```text
Event: Dispatch shipment
Timestamp: 2026-03-14 10:43

Related objects:
    Order: O-173
    Items: I-914, I-915
    Shipment: S-81
    Warehouse: W-2
    Carrier: C-4
```

A second order may share the same shipment or carrier. An invoice may cover several orders, and one order may generate several shipments. Flattening these relationships into one row sequence per order can duplicate events, introduce artificial loops, or produce misleading directly-follows relations. Object-centric process mining was developed in part to preserve these interacting lifecycles, and graph-based object-centric features have already been used to transform these relationships into machine-learning inputs. ([Springer Link][3])

For the proposed foundation model, the preferred raw representation is therefore:

[
\mathcal L_{\mathrm{OC}}
========================

\left(
E,O,R_{EO},R_{OO},A_E,A_O
\right),
]

where:

* (E) is the set of events;
* (O) is the set of objects;
* (R_{EO}) contains event–object relations;
* (R_{OO}) contains object–object relations;
* (A_E) contains event attributes;
* (A_O) contains potentially time-varying object attributes.

---

# 2. The causal unit is query-dependent

A major transformation step is deciding what constitutes one causal unit. In process data, the unit is not always identical to the case identifier.

Depending on the question, the causal unit may be:

* one customer application;
* one order;
* one order item;
* one shipment;
* one machine cycle;
* one patient episode;
* one resource shift;
* one warehouse-day;
* one decision opportunity within a case;
* one connected component of interacting objects.

Consider the question:

> What is the effect of expediting an order on the probability of late delivery?

The appropriate observational unit may not be the completed order. It is more precisely an **eligible expedite decision opportunity**. For each order (i) and eligible decision time (d), define

[
S_{i,d}
=======

\left(
H_{i,d},
A_{i,d},
Y_{i,d}^{(h)},
E_{i,d},
R_{i,d}
\right),
]

where:

* (H_{i,d}) is the complete process history available before the decision;
* (A_{i,d}) is the action taken at the decision point;
* (Y_{i,d}^{(h)}) is the future outcome over horizon (h);
* (E_{i,d}) identifies the environment, site, or process version;
* (R_{i,d}) describes relationships with other units and shared objects.

A single trace may therefore produce several causal situations:

[
\sigma_i
\rightarrow
{
S_{i,1},S_{i,2},\ldots,S_{i,D_i}
}.
]

This transformation is central. It converts event logs from records of completed executions into a collection of **decision-centered causal episodes**.

---

# 3. Identifying decision points and interventions

## 3.1 Not every activity is an intervention

The occurrence of an activity should not automatically be interpreted as a treatment.

For example:

* “manual review” may be a response to an already difficult case;
* “request additional documents” may be triggered by missing information;
* “expedite shipment” may be triggered because an order is already late;
* “perform diagnostic test” may reflect the severity of the patient;
* “rework product” may be a consequence of an earlier defect.

Comparing cases with and without these activities can therefore produce severe confounding by indication.

The event-data schema should classify process variables into categories such as:

| Process element               | Interpretation                                             |
| ----------------------------- | ---------------------------------------------------------- |
| Controllable decision         | Potential intervention                                     |
| Automatically generated state | State or covariate                                         |
| Human judgment                | Decision and possible confounding mechanism                |
| Process outcome               | Outcome                                                    |
| Intermediate activity         | Potential mediator                                         |
| Error or deviation            | Cause, mediator, or symptom depending on timing            |
| Resource assignment           | Potential intervention                                     |
| Queue or workload             | Time-varying confounder and possible interference exposure |

A causal intervention must be operationally meaningful. Instead of asking

[
do(\text{manual review occurs}),
]

the query should specify something such as

[
do(\text{assign case to manual review at decision point }d).
]

This clarifies who is eligible, when the action can be taken, and what alternative action is being compared.

---

## 3.2 Constructing the risk set

For every decision point, the data processor constructs the set of cases or objects that were actually eligible for each action.

Suppose an expedite decision can only be made after inventory has been checked and before shipment has been dispatched. The eligible risk set is

[
\mathcal R_d
============

\left{
i:
\text{inventory checked before }d,
\text{ shipment not dispatched by }d
\right}.
]

Cases that had already been shipped should not appear as untreated comparison cases. Otherwise, the comparison would contain structural non-overlap and potentially immortal-time bias.

For every eligible unit, the processor extracts:

[
H_{i,d}
=======

\left{
e:
t_e<t_d
\text{ and }
e\text{ is related to unit }i
\right}.
]

Only information available before the decision is included. Events after (t_d) are potential mediators or outcomes and must not be used as baseline adjustment variables.

---

## 3.3 Treatment representation

A treatment token should contain:

```yaml
decision_point: post_inventory_check
action_variable: expedite_order
assigned_action: true
received_action: true
decision_time: 2026-03-14T08:12:00
alternative_action: standard_processing
decision_maker: planner_17
assignment_policy: human_discretion
eligibility_status: eligible
```

The model should distinguish:

* assignment from actual receipt;
* hard interventions from recommendations;
* one-time actions from time-varying treatment policies;
* binary actions from dosage or intensity;
* individual actions from resource-level or policy-level interventions.

A longitudinal process can contain a treatment sequence

[
\bar A_{i,t}
============

(A_{i,1},\ldots,A_{i,t}),
]

rather than one static treatment.

---

# 4. Four complementary input views

The event log should be transformed into several synchronized representations rather than one feature table.

## 4.1 Event-token sequence

Each event becomes a token

[
z_e=
f_{\mathrm{activity}}(a_e)
+
f_{\mathrm{attributes}}(\mathbf{x}*e)
+
f*{\mathrm{time}}(t_e)
+
f_{\mathrm{resource}}(r_e)
+
f_{\mathrm{objects}}(O_e)
+
f_{\mathrm{role}}(e,Q).
]

An event token may contain:

* activity embedding;
* lifecycle transition;
* duration;
* time since the preceding event;
* time since case creation;
* time relative to the treatment decision;
* resource and organizational-unit embedding;
* event attributes;
* related object types;
* treatment, outcome, or mediator role;
* missingness and measurement-quality indicators.

For a decision at time (t_d), the model receives only the prefix

[
\sigma_i^{<d}
=============

\langle
e_{i1},\ldots,e_{ik}
\rangle
\quad\text{with}\quad
t_{ik}<t_d.
]

The future suffix

[
\sigma_i^{\ge d}
]

is used to construct outcomes or training targets.

This representation preserves order and duration and allows the temporal encoder to learn process states that cannot be adequately summarized by fixed counts.

---

## 4.2 Decision-state table

A second view converts every decision situation into a static or longitudinal feature row.

For an expedite decision, the row could contain:

```text
order_id
decision_time
current_inventory
number_of_open_items
time_since_order_creation
promised_delivery_slack
number_of_prior_reworks
supplier_recent_delay_rate
warehouse_work_in_progress
carrier_current_load
last_activity
process_variant_so_far
conformance_deviation_count
treatment_expedite
late_delivery
total_cost
```

Features can be obtained from:

* trace prefixes;
* process-model alignments;
* performance analysis;
* object lifecycles;
* resource workloads;
* object interaction graphs;
* conformance deviations;
* process variant information.

Existing object-centric feature-extraction work demonstrates that lifecycle and interaction information can be transformed into numerical vectors and that the correct feature horizon should depend on the prediction or decision point. ([Springer Link][3])

This table serves several purposes:

1. It provides a compact input for the foundation model’s tabular route.
2. It makes direct comparison with classical causal techniques possible.
3. It allows the formal identification layer to operate on explicitly defined variables.
4. It permits influence-function, AIPW, DML, or design-based estimation after the neural encoder has constructed process-state representations.

The event-token and decision-table views are complementary. The table should not replace the original event sequence.

---

## 4.3 Heterogeneous event–object graph

Object-centric event data naturally induce a heterogeneous graph:

[
G_{\mathrm{process}}
====================

(V_E\cup V_O\cup V_R,\mathcal E),
]

where:

* (V_E) are event nodes;
* (V_O) are object nodes;
* (V_R) are resource or organizational nodes.

Possible edges include:

* event performed on object;
* event performed by resource;
* event directly follows event for a given object type;
* order contains item;
* invoice refers to order;
* shipment transports item;
* order assigned to warehouse;
* cases share a resource;
* objects overlap in time;
* one event transfers work to another organizational unit.

For example:

```text
Order O1 ─ contains ─ Item I1
Order O1 ─ fulfilled-by ─ Shipment S1
Order O2 ─ fulfilled-by ─ Shipment S1
Shipment S1 ─ handled-by ─ Carrier C1
Event E7 ─ affects ─ Order O1
Event E7 ─ affects ─ Shipment S1
```

This graph supplies two different kinds of structure.

### Within-unit process structure

It describes the lifecycle of one order, item, machine, or case.

### Between-unit interference structure

It describes how different units may affect each other through:

* shared resources;
* common queues;
* shared shipments;
* common suppliers;
* machine capacity;
* staff workload;
* batch processing;
* congestion.

The same object-centric representation can therefore support both process analysis and causal interference analysis.

---

## 4.4 Process-model view

A discovered Petri net, BPMN model, process tree, transition system, or directly-follows graph can be supplied as an additional structural input.

It can provide:

* observed activity precedence;
* concurrency information;
* gateways and choices;
* loops;
* reachable future activities;
* candidate decision points;
* process variants;
* deviation and alignment information.

The process model should be interpreted as a **behavioral and temporal prior**, not as the causal graph.

For example, if activity (A) can only occur after activity (B), the process model supports the temporal restriction

[
A \not\rightarrow B
]

when the variables represent those particular executions. But observing that (B) usually precedes (A) does not establish

[
B\rightarrow A.
]

Existing work on causal execution dependencies explicitly uses discovered process structure to constrain or guide causal analysis while emphasizing that temporal precedence is necessary but insufficient for a causal relation. ([Springer Link][4])

The process-model input could therefore enter the foundation model as:

* attention masks;
* candidate-edge masks;
* soft reachability priors;
* gateway tokens;
* loop-context tokens;
* process-position embeddings;
* conformance-deviation features.

---

# 5. Query-centered transformation

The same event log can support many different causal questions. The data transformation must therefore be conditioned on the query.

Consider three questions concerning the same process:

1. Does expediting reduce late deliveries?
2. Does adding a manual check reduce defects?
3. Does high warehouse workload cause longer order lead times?

They require different:

* decision points;
* eligible units;
* treatment variables;
* horizons;
* adjustment histories;
* mediators;
* interference graphs.

The processor should convert a formal query (Q) into a query-specific episode:

[
\mathcal E_Q
============

\left(
Z_{\mathrm{events}},
X_{\mathrm{decision}},
G_{\mathrm{process}},
P_{\mathrm{model}},
M,
I,
Q
\right).
]

A query might be represented as:

```yaml
claim_type: interventional

unit:
  focal_object_type: order

decision_point:
  after_activity: inventory_check
  before_activity: shipment_dispatch

treatment:
  variable: expedite
  contrast: [standard, expedited]

outcome:
  variable: late_delivery
  horizon: 7_days

target_population:
  condition: eligible_orders

interference:
  enabled: true
  relation: shared_carrier_and_overlapping_time
  exposure: fraction_of_other_expedited_orders

assumptions:
  assignment: observational
  hidden_confounding: possible
```

The query controls which parts of the event history can be observed by the model.

---

# 6. Example transformation: inventory and order fulfilment

Consider the following object-centric process.

## 6.1 Raw events

| Time  | Activity                 | Related objects           | Selected attributes |
| ----- | ------------------------ | ------------------------- | ------------------- |
| 08:00 | Create order             | Order O1, Customer C1     | priority=normal     |
| 08:04 | Add item                 | Order O1, Item I1, SKU P7 | quantity=10         |
| 08:11 | Check stock              | Item I1, Warehouse W1     | available=3         |
| 08:13 | Select fulfilment policy | Order O1, Warehouse W1    | expedite=true       |
| 08:25 | Create replenishment     | SKU P7, Supplier S2       | quantity=7          |
| 11:10 | Pick item                | Item I1, Resource R8      | partial=true        |
| 14:40 | Receive replenishment    | SKU P7, Supplier S2, W1   | quantity=7          |
| 15:20 | Complete pick            | Item I1, Resource R4      | complete=true       |
| 16:05 | Dispatch                 | Order O1, Shipment SH3    | carrier=C2          |
| Day 3 | Deliver                  | Order O1, Shipment SH3    | late=false          |

## 6.2 Causal question

> For orders with insufficient stock at the inventory-check point, what is the effect of selecting expedited replenishment rather than standard replenishment on the probability of late delivery?

## 6.3 Decision-centered representation

The decision time is 08:13. The pre-decision history is:

[
H_{O1,d}
========

{
\text{Create order},
\text{Add item},
\text{Check stock}
}.
]

The treatment is

[
A_{O1,d}
========

\mathbb 1(\text{expedited replenishment}).
]

The outcome is

[
Y_{O1,d}
========

\mathbb 1(\text{delivery later than promise}).
]

Potential pre-treatment confounders include:

* order priority;
* stock shortage;
* promised delivery date;
* item quantity;
* recent supplier reliability;
* current warehouse congestion;
* customer category;
* planner workload;
* product scarcity.

Potential mediators include:

* replenishment lead time;
* pick completion time;
* carrier selection;
* shipment dispatch time.

Those mediator events occur after the treatment and should not be used as ordinary baseline confounders when estimating the total effect.

## 6.4 Interference representation

Suppose expedited orders share a limited carrier or warehouse team. Expediting order (i) may delay other orders.

Define a relation matrix

[
W_{ij}(t)=
\mathbb 1
\left(
i\text{ and }j
\text{ use the same warehouse or carrier at time }t
\right).
]

A peer-exposure measure may be

[
G_i(t)
======

\frac{
\sum_{j\neq i}W_{ij}(t)A_j(t)
}{
\sum_{j\neq i}W_{ij}(t)
}.
]

The causal question can then distinguish:

* the direct effect of expediting order (i);
* the spillover effect of other expedited orders;
* the interaction between own treatment and congestion exposure.

This corresponds directly to the network-interference experiments in the ZIP package.

## 6.5 Model input

The foundation model receives:

```text
Event prefix:
    Create order → Add item → Check stock

Current object state:
    order priority
    quantity
    stock shortage
    promised-date slack
    supplier state
    warehouse workload

Object graph:
    Order → Items → SKU → Supplier
    Order → Warehouse
    Order → Shared carrier/resource network

Process context:
    decision gateway
    current process version
    warehouse/site
    calendar regime

Treatment query:
    expedite vs standard

Outcome:
    late delivery within seven days
```

The output can include:

* average treatment effect;
* conditional effects by product, warehouse, and shortage level;
* spillover effect on other orders;
* mediation through replenishment and picking time;
* support and overlap diagnostics;
* hidden-confounding sensitivity;
* recommended decision time;
* process-model annotations.

---

# 7. How process mining supplies structure to the causal model

Process mining can contribute more than feature engineering.

## 7.1 Control-flow constraints

A discovered process model can exclude causally impossible directions based on temporal ordering.

For example:

```text
Create order
     ↓
Check stock
     ↓
Choose fulfilment policy
     ↓
Pick and dispatch
```

A later delivery outcome cannot cause an earlier policy decision within the same process instance. Such constraints reduce the causal-search space.

However, the model should remain cautious about:

* logging delays;
* simultaneous events;
* incorrect timestamps;
* backward effects through repeated decisions;
* cross-case effects;
* latent common causes.

---

## 7.2 Decision-point identification

Gateways and branching points identify places at which different actions are possible.

For example:

```text
                         ┌─ Standard replenishment ─┐
Check inventory ─ choice                           ├─ Dispatch
                         └─ Expedite replenishment ─┘
```

This provides a natural candidate intervention:

[
A=
\begin{cases}
0,&\text{standard replenishment},\
1,&\text{expedited replenishment}.
\end{cases}
]

The process model also helps determine eligibility: only cases reaching that gateway are members of the relevant risk set.

---

## 7.3 Conformance information

Alignments between observed traces and a normative model can produce features such as:

* skipped activity;
* unexpected activity;
* repeated activity;
* late activity;
* incorrect sequence;
* unauthorized resource;
* unfulfilled temporal constraint.

The causal model can then ask:

> Does this deviation cause an adverse outcome, or is it merely a symptom of an already difficult case?

This is stronger than correlating alignment cost with process duration.

A repeated activity may appear strongly associated with delays simply because difficult cases require more work. The foundation model can use the pre-deviation history and explicit treatment timing to distinguish, as far as assumptions permit, between:

[
\text{rework}\rightarrow\text{delay}
]

and

[
\text{latent difficulty}
\rightarrow
\begin{cases}
\text{rework},\
\text{delay}.
\end{cases}
]

---

## 7.4 Process variants as environments or regimes

Trace variants, process versions, sites, or calendar periods can be used as environments:

[
E\in
{
\text{variant 1},
\text{variant 2},
\text{site A},
\text{site B},
\text{policy version 3}
}.
]

The foundation model can study whether:

* a causal effect is invariant across variants;
* the same intervention works differently after a policy update;
* a relation appears only in one warehouse;
* a process change altered the treatment assignment but not the outcome mechanism;
* one variant reflects a genuinely different causal regime.

Variants should not automatically be interpreted as causal groups. They become candidate environments for invariance, heterogeneity, and transportability analysis.

---

## 7.5 Performance annotations as outcomes and mediators

Process mining can compute:

* waiting time;
* service time;
* sojourn time;
* remaining time;
* queue length;
* work in progress;
* synchronization delay;
* throughput time.

These become variables in the causal episode.

For example:

[
\text{extra quality check}
\rightarrow
\text{reduced rework}
\rightarrow
\text{shorter total lead time}.
]

The model can separate:

* direct effect of the quality check on total lead time;
* indirect effect mediated through rework;
* possible opposing effect caused by the additional processing time.

---

# 8. How the causal model helps understand the process

## 8.1 A causal overlay on the discovered process model

The first output can be an ordinary process model augmented with causal annotations.

For each relevant activity or decision, the model could display:

* causal effect on a KPI;
* direct and total effects;
* affected population;
* effect heterogeneity;
* confidence or posterior interval;
* identification status;
* support status.

For example:

```text
Check stock
     ↓
[Expedite order]
     │
     ├── Direct effect on late delivery:  −0.08
     ├── Indirect effect via faster replenishment: −0.12
     ├── Congestion spillover: +0.05
     └── Net estimated effect: −0.15
```

The overlay should visually distinguish:

* observed directly-follows edges;
* conditional associations;
* causal dependencies;
* possible latent confounding;
* uncertain or unoriented relations.

Research on causal execution dependencies similarly argues for supplementing conventional discovered process models with a causal perspective rather than replacing process discovery altogether. ([Springer Link][4])

---

## 8.2 Root-cause analysis

Traditional process analysis may show that delayed cases:

* contain more rework;
* involve a certain team;
* follow a particular variant;
* wait longer before approval.

The causal model can instead ask:

[
P(Y\mid do(X=x))
]

rather than only

[
P(Y\mid X=x).
]

This enables questions such as:

* Would moving the approval earlier actually reduce lead time?
* Would assigning an additional resource reduce waiting time?
* Would removing a manual review increase error rates?
* Is the observed supplier effect explained by product mix?
* Does process variant (V) cause delay, or does it serve difficult cases?

Existing causal process-mining research has explored structural-equation-based root-cause analysis and causal feature recommendation precisely because ordinary correlations and feature importance do not establish intervention effects. ([Springer Link][5])

---

## 8.3 Case-level counterfactual explanations

For a specific delayed case, the system can answer:

> What feasible earlier decision could have changed the outcome?

For case (i), it may estimate

[
Y_i(a=0)
\quad\text{and}\quad
Y_i(a=1).
]

A process-aware counterfactual should respect:

* activity order;
* eligibility conditions;
* resource feasibility;
* object relationships;
* process constraints;
* immutable case properties.

For example:

```text
Observed trace:
Create order → Check stock → Standard replenishment
→ Wait 30 h → Pick → Dispatch → Late delivery

Counterfactual:
Create order → Check stock → Expedite replenishment
→ Wait 9 h → Pick → Dispatch → On-time delivery
```

The result must be presented as assumption-dependent, especially for an individual case. Case-level counterfactual reasoning over event-log features has already been proposed as a way to move from process diagnosis toward process intervention analysis. ([arXiv][6])

---

## 8.4 Heterogeneous intervention effects along the process

An intervention may work only:

* at an early decision point;
* before congestion exceeds a threshold;
* for certain products;
* at particular sites;
* for specific process variants;
* for cases with a particular history.

The model can estimate

[
\tau(H_{i,d})
=============

\mathbb E
\left[
Y_{i,d}(1)-Y_{i,d}(0)
\mid H_{i,d}
\right].
]

This produces a process-state-dependent treatment-effect map:

| Process state                             |            Estimated effect of expediting |
| ----------------------------------------- | ----------------------------------------: |
| Low congestion, large promised-date slack |                             Close to zero |
| Low congestion, small slack               |                       Strongly beneficial |
| High congestion, small slack              |                     Moderately beneficial |
| Extreme carrier saturation                |                     Harmful net spillover |
| Supplier already dispatched               | Unsupported or no meaningful intervention |

This is more informative than grouping cases only by completed trace variant.

---

## 8.5 Understanding when to intervene

Because event logs contain timestamps and prefixes, the effect of an intervention can be estimated at different points in process execution:

[
\tau_d(h)
=========

\mathbb E
\left[
Y^{do(A_d=1)}
-------------

Y^{do(A_d=0)}
\mid H_d=h
\right].
]

The system may find that:

* an intervention is beneficial immediately after an early warning;
* the same action has almost no effect after a later milestone;
* late intervention consumes resources without changing the outcome;
* treatment is only worthwhile when expected benefit exceeds cost.

Causal prescriptive-monitoring research has already emphasized that treatment effects and net gains can depend on when an action is taken during a running case. ([Springer Link][7])

---

## 8.6 Detecting causal process changes

Process-mining logs often span several:

* process versions;
* policy changes;
* software releases;
* organizational restructurings;
* seasonal periods;
* disruptions.

The model can distinguish several kinds of change:

[
\Delta P(A\mid H)
]

change in treatment or routing policy,

[
\Delta P(Y\mid A,H)
]

change in the outcome mechanism,

[
\Delta P(H)
]

change in case mix,

and

[
\Delta W
]

change in resource or object-interaction structure.

This is significantly more informative than stating that trace frequencies or throughput times changed.

For example:

```text
Observed:
Late-delivery rate increased after April.

Possible explanation 1:
More difficult orders arrived.

Possible explanation 2:
Expediting became less effective.

Possible explanation 3:
Planners changed the criteria for expediting.

Possible explanation 4:
Carrier-network congestion increased.
```

The regime and temporal components of the foundation model can separately evaluate these explanations.

---

## 8.7 Cross-case effects and resource contention

Traditional case-centric analysis often assumes independent cases. Operational processes frequently violate this assumption.

Examples include:

* one expedited order consumes capacity needed by other orders;
* one urgent patient delays treatment of others;
* one large batch changes the waiting time of subsequent cases;
* assigning an expert to one case makes the expert unavailable elsewhere;
* several orders share one shipment;
* supplier disruption affects many products simultaneously.

Object-centric event data provide the relations needed to construct such interference structures. Graph-based object-centric work already shows that interactions between objects contain information that is lost under flattened single-case representations. ([Springer Link][3])

The causal model can produce:

* direct effects on the treated case;
* spillover effects on connected cases;
* total system effects;
* capacity-dependent effects;
* policy effects under alternative treatment saturation levels.

An action that benefits one case may be harmful at process-system level:

[
\text{individual benefit}>0,
\qquad
\text{system-wide net benefit}<0.
]

This distinction is particularly important for inventory, logistics, healthcare, and service processes.

---

## 8.8 Discovering latent operational states

Some important process states are not directly recorded:

* supplier reliability;
* latent demand regime;
* organizational stress;
* case complexity;
* machine degradation;
* employee experience;
* congestion state;
* customer urgency.

They may nevertheless influence many observed event patterns.

The latent-variable component can use:

* activity frequencies;
* waiting times;
* rework patterns;
* resource changes;
* object interactions;
* sensor values;
* textual event attributes;

to infer a posterior over latent process states.

For example:

[
Z_t=\text{latent congestion state}
]

may explain simultaneous changes in:

* queue length;
* picking delay;
* resource reassignment;
* partial shipments;
* expedite frequency;
* late delivery.

The latent state is then included in the posterior causal worlds rather than converted into one certain clustering.

---

## 8.9 Understanding deviations rather than merely detecting them

Conformance checking can identify that a trace deviates from a normative model. The causal foundation model can determine whether the deviation is:

* harmful;
* beneficial;
* inconsequential;
* a response to an earlier problem;
* a mediator of another cause.

For example:

```text
Deviation: manual approval was skipped.
Association: skipped approval cases finish faster.
Causal analysis:
    Direct effect on duration: beneficial.
    Effect on compliance failure: harmful.
    Net utility depends on compliance cost.
```

Another deviation may appear harmful but actually be a symptom:

```text
Deviation: repeated customer contact.
Association: strong relation with long lead time.
Causal interpretation:
    complex cases cause both repeated contact and long lead time;
    removing contact would not solve the delay.
```

This changes the role of conformance checking from detecting differences to understanding which differences should be changed.

---

## 8.10 Process redesign and policy comparison

Once an intervention distribution is estimated, the system can compare process policies such as:

* always expedite;
* never expedite;
* expedite only when promised-date slack is below a threshold;
* allocate an expert to cases with high expected treatment benefit;
* route cases by predicted causal benefit rather than predicted risk;
* increase capacity at one activity;
* eliminate or add a quality-control step.

For policy (\pi), the target becomes

[
V(\pi)
======

\mathbb E
\left[
Y^{do(A=\pi(H))}
\right].
]

The model can compare policy value, cost, resource use, and spillover effects.

The critical distinction is:

[
\text{high predicted risk}
\neq
\text{high expected benefit from intervention}.
]

A case may be highly likely to fail but impossible to help with the available intervention. Conversely, a medium-risk case may have a large preventable component.

---

# 9. Mapping the ZIP experiments to process-mining questions

The six experiment families in the package can be interpreted as different views of one rich event-log setting.

| Package experiment                | Process-mining interpretation                                            | Event-data transformation                                   | Process insight                                                 |
| --------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------------------------------------------------------- |
| Observed causal regimes           | Different process contexts or variants have different mechanisms         | Prefix or case features plus candidate splitting variables  | Where and for whom does the process behave differently?         |
| Network interference              | Cases interact through resources, queues, suppliers, or shared objects   | Object-interaction or resource-sharing graph                | Does helping one case affect other cases?                       |
| Outcome-relevant latent groups    | Observed process features measure hidden operational states              | Indicators from event patterns, performance, and attributes | Which hidden factors explain outcomes?                          |
| Relationships among latent groups | Hidden process states interact                                           | Latent-state trajectories or group scores                   | How do congestion, reliability, complexity, and quality relate? |
| Contemporaneous temporal changes  | Process mechanism changes over calendar time                             | Time-windowed event or state data                           | When did the process start behaving differently?                |
| Lagged VARX changes               | Earlier process variables affect later performance differently over time | Lagged state sequences and event histories                  | Which delayed relationships changed, and what changed?          |

The package’s tasks therefore need not be viewed as unrelated experiments. Process event data provide the common substrate from which all six causal analysis problems can be generated.

For instance, the same procure-to-pay OCEL may provide:

* case-level attributes for regime discovery;
* shared-supplier relations for interference;
* event-pattern indicators for latent supplier reliability;
* a graph among reliability, congestion, and quality factors;
* calendar sequences for change detection;
* lagged inventory and demand variables for VARX analysis.

This is the precise sense in which process mining acts as the **glue**.

---

# 10. Outputs should be mapped back to process-mining artifacts

The foundation model should not return only an abstract adjacency matrix or a numerical ATE. Its outputs should be translated into familiar process views.

## 10.1 Causal process map

A DFG, Petri net, BPMN model, or object-centric process model annotated with:

* causal execution dependencies;
* treatment effects;
* mediation paths;
* uncertainty;
* hidden-confounding warnings.

## 10.2 Decision-point effect map

For every process decision:

```text
Decision: expedite order
Eligible cases: 8,420
Supported comparison: 6,980
Estimated effect: −8.3 percentage points late delivery
Best-performing context: low carrier saturation
Harmful context: extreme warehouse congestion
Identification: conditional on measured-confounding assumption
```

## 10.3 Counterfactual trace view

Show the observed trace and a feasible alternative trace, highlighting the changed decision and predicted downstream effects.

## 10.4 Regime timeline

Display:

* detected change interval;
* affected activities or mechanisms;
* process version;
* change type;
* uncertainty.

## 10.5 Object-interference view

Show which objects, resources, or cases generate spillovers and whether their estimated effects are positive or negative.

## 10.6 Identification and sensitivity report

For each causal conclusion, state:

* required assumptions;
* support status;
* possible unmeasured confounding;
* exposure-map uncertainty;
* alternative estimates;
* sensitivity bounds;
* abstention reason.

---

# 11. End-to-end processing algorithm

A process-aware causal pipeline can be summarized as follows.

### Step 1: Ingest event data

Load a traditional event log, OCEL, or relational event database.

### Step 2: Construct process semantics

Identify:

* activities;
* objects;
* resources;
* timestamps;
* lifecycles;
* process versions;
* object and resource relationships.

### Step 3: Discover or import process structure

Obtain:

* process model;
* decision points;
* variants;
* concurrency;
* alignments;
* deviations;
* performance annotations.

### Step 4: Formalize the causal query

Specify:

* causal unit;
* decision point;
* treatment alternatives;
* outcome and horizon;
* target population;
* possible interference;
* assumptions.

### Step 5: Construct eligible decision situations

For each eligible unit and time, extract only the history available before the decision.

### Step 6: Build synchronized model views

Generate:

* event-token prefixes;
* decision-state table;
* event–object graph;
* resource-interaction graph;
* process-model tokens;
* environment and regime tokens;
* query tokens.

### Step 7: Infer posterior causal worlds

Estimate alternative:

* causal graphs;
* process regimes;
* latent states;
* exposure mappings;
* treatment mechanisms.

### Step 8: Run identification and support checks

Determine whether the requested effect is:

* identified;
* partially identified;
* unsupported;
* assumption-dependent;
* not identifiable.

### Step 9: Estimate effects or bounds

Use the direct interventional decoder and compiled classical estimator.

### Step 10: Project results back onto the process

Return causal process maps, decision recommendations, counterfactual traces, regime timelines, and sensitivity reports.

---

# 12. Important limitations

## Event order is not causal order

A directly-follows edge

[
A>_L B
]

means that (B) was observed after (A) in relevant traces. It does not by itself establish

[
A\rightarrow B.
]

This is one of the central motivations for combining process mining with causal discovery and inference. ([arXiv][8])

## Flattening can create spurious relationships

Selecting one case notion in a multi-object process can duplicate events or create artificial loops and back-jumps. Object-centric inputs should therefore be preferred where multiple objects interact. ([OCEL 2.0][2])

## Human decisions are usually confounded

Operators direct difficult cases toward special treatment. The event history must capture the information available to the operator, and the model must still acknowledge the possibility of unrecorded judgment.

## Post-treatment events must not become baseline features

A feature calculated from the complete trace may contain information generated after the treatment. All process features must be computed relative to the decision time.

## A discovered process model is not causal ground truth

It can constrain temporal possibilities and organize the analysis, but causal orientation still depends on design information and assumptions.

## Some activities are not manipulable

“Case becomes complex” or “defect detected” is not automatically a meaningful treatment. The intervention must correspond to a realizable process action or policy.

## Cross-case dependence must be represented

Cases that share resources, objects, queues, or capacity cannot always be treated as independent observations.

## Counterfactuals remain assumption-dependent

A realistic alternative trace must satisfy process feasibility, but feasibility alone does not establish causal identification.

---

# Conclusion

The interconnection between process mining and the proposed causal foundation model is deeper than using process-derived features in a tabular causal estimator.

Process mining provides a structured representation of:

* **what happened**, through events;
* **when it happened**, through timestamps;
* **to what it happened**, through cases and objects;
* **who or what performed it**, through resources;
* **how executions interacted**, through object and resource relations;
* **where decisions occurred**, through gateways and process states;
* **how behavior changed**, through variants and temporal regimes.

The foundation model transforms this information into decision-centered causal episodes and adds reasoning about:

* what caused an outcome;
* what would happen under an intervention;
* when and for whom an action is effective;
* how an action affects other cases;
* which process changes altered causal mechanisms;
* when the available event data cannot support the requested causal claim.

The central conceptual distinction is:

[
\boxed{
\text{process model}
====================

\text{behavioral structure}
}
]

whereas

[
\boxed{
\text{causal model}
===================

\text{interventional structure under explicit assumptions}.
}
]

The two models should be linked but not conflated. Process mining supplies temporal order, eligibility, decision points, object relations, variants, and interpretable visualizations. The causal foundation model uses that structure to infer, estimate, test, or decline causal conclusions. Its results are then returned to the process view as causal annotations, counterfactual traces, intervention-effect maps, regime explanations, and system-level policy evaluations.

In this architecture, event data are the “glue” because one object-centric event log can simultaneously support regime discovery, latent-state inference, temporal causal analysis, resource-interference modeling, treatment-effect estimation, counterfactual reasoning, and prescriptive process improvement.

[1]: https://link.springer.com/chapter/10.1007/978-3-031-08848-3_1 "Process Mining: A 360 Degree Overview | Springer Nature Link"
[2]: https://www.ocel-standard.org/ "OCEL 2.0 - Object-Centric Event Log 2.0"
[3]: https://link.springer.com/article/10.1007/s41060-023-00428-2 "Graph-based feature extraction on object-centric event logs | International Journal of Data Science and Analytics | Springer Nature Link"
[4]: https://link.springer.com/article/10.1007/s13218-024-00883-4 "The WHY in Business Processes: Discovery of Causal Execution Dependencies | KI - Künstliche Intelligenz | Springer Nature Link"
[5]: https://link.springer.com/article/10.1007/s13748-022-00282-6?utm_source=chatgpt.com "Feature recommendation for structural equation model ..."
[6]: https://arxiv.org/abs/2102.13490 "[2102.13490] Case Level Counterfactual Reasoning in Process Mining"
[7]: https://link.springer.com/chapter/10.1007/978-3-031-34560-9_22 "Learning When to Treat Business Processes: Prescriptive Process Monitoring with Causal Inference and Reinforcement Learning | Springer Nature Link"
[8]: https://arxiv.org/abs/2202.08314 "[2202.08314] Causal Process Mining from Relational Databases with Domain Knowledge"
