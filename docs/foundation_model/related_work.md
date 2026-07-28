# Related work: a causal foundation model for inventory management grounded in process event data

Based on the uploaded package, the strongest positioning is **not merely a foundation model for causal-graph discovery**. It is a **query-conditioned foundation model for causal reasoning over operational process worlds**. The package’s experiments cover observed regimes, interference, latent response groups, hierarchical latent structures, contemporaneous changes, and lagged temporal effects. They also show why predictive improvement alone is insufficient: misspecification may produce false regimes, hidden confounding may yield apparently strong detection with invalid coverage, and poor overlap may require abstention rather than an effect estimate. These findings are documented in the [benchmark report](sandbox:/mnt/data/foundation_causality_work/foundation-causality0-main/REPORT.md) and motivate the identification- and support-aware design in the [initial architecture](sandbox:/mnt/data/foundation_causality_work/foundation-causality0-main/docs/foundation_model/initial_architecture.md).

## 1. Structural causal modeling, effect identification, and causal discovery

Structural causal models and the potential-outcomes framework provide the conceptual basis for distinguishing three fundamentally different tasks:

1. predicting an outcome under the observed process;
2. estimating the effect of changing an action or policy;
3. reasoning about what would have happened to a particular process execution under an alternative action.

This distinction is essential for inventory applications. Predicting tomorrow’s demand is not the same as estimating what demand would have been under a different price, replenishment rule, promotion, or service level. Likewise, predicting a stockout does not identify whether expediting a shipment would prevent it. These causal quantities require assumptions concerning confounding, consistency, positivity, interference, and the temporal ordering of variables—not only a well-performing predictor \cite{pearl2009causality,hernan2020whatif}. ([Cambridge University Press][1])

Causal-discovery research addresses a related but different problem: learning aspects of the causal structure from observational, interventional, heterogeneous, or temporal data. Representative approaches include continuous optimization for DAG learning, invariant prediction across environments, discovery from heterogeneous or nonstationary data, and causal analysis of nonlinear time series \cite{peters2016invariant,zheng2018notears,huang2020heterogeneous,runge2019pcmci}. Interference theory extends the usual independent-unit setting by allowing the treatment of one unit to affect another \cite{hudgens2008interference}. These strands map directly to the ZIP experiments on regimes, temporal changes, and spillovers. ([NeurIPS Proceedings][2])

The main architectural implication is that the proposed model should maintain a separation between:

* a posterior over plausible causal structures or “causal worlds”;
* an identification analysis for the submitted query;
* an estimator that is only invoked when the query is identified or partially identified;
* uncertainty, sensitivity, support, and abstention outputs.

A high-confidence predicted graph should not automatically imply a high-confidence causal-effect estimate.

## 2. Causal foundation models

Recent work has begun to amortize causal inference across many synthetic data-generating processes. This is the closest methodological precedent for the proposed model.

### 2.1 Foundation models for causal-effect estimation

CausalFM constructs prior-data fitted networks from priors over structural causal models. Its framework covers causal settings such as back-door, front-door, and instrumental-variable identification, while its principal empirical instantiation estimates conditional average treatment effects. Do-PFN learns to predict interventional outcomes from observational datasets generated from a broad synthetic causal prior. CausalPFN similarly amortizes average and heterogeneous treatment-effect estimation, but its training tasks satisfy ignorability. All three replace repeated estimator selection and fitting with in-context inference by a pretrained transformer \cite{ma2025causalfm,robertson2025dopfn,balazadeh2025causalpfn}. ([arXiv][3])

These models establish that causal estimation can be amortized, but they also illustrate an important limitation: the assumptions encoded in the synthetic prior remain causal assumptions. Pretraining does not make an observational query identifiable when the required assumptions or data support are absent.

### 2.2 Foundation models for causal discovery

Arrow performs zero-shot causal discovery by decomposing a DAG into an undirected skeleton and a topological order. TabCausal predicts directed edges from tabular data, optionally using intervention indicators, while CDFM formulates causal discovery across heterogeneous and initially unknown causal mechanisms \cite{thompson2026arrow,li2026tabcausal,qiao2026cdfm}. Work on partial graphs additionally shows how prior structural knowledge can be supplied to a causal foundation model instead of forcing it to infer everything from data \cite{reuter2026partialgraphs}. ([arXiv][4])

These methods primarily treat a dataset as a table whose columns are causal variables and whose rows are observational units. A process event log is structurally different: it contains irregular event sequences, changing object populations, multiple related object types, timestamps, quantities, and actions selected adaptively from the preceding history.

### 2.3 Temporal, longitudinal, and sensitivity-aware causal foundation models

The most relevant extensions for process data are emerging rapidly. CausalTimePrior generates paired observational and interventional temporal SCM data with nonlinear autoregressive mechanisms, regime changes, and time-varying interventions. CausalLongPFN conditions on longitudinal histories and proposed future treatment sequences to predict counterfactual outcomes under treatment–confounder feedback. A complementary PFN approach amortizes causal sensitivity analysis, returning bounds as the assumed strength of unobserved confounding changes \cite{thumm2026causaltimeprior,zare2026causallongpfn,javurek2026sensitivitypfn}. ([arXiv][5])

These works are particularly important for the proposal, but they do not yet provide a general representation of operational events, process control flow, object relations, inventory movements, or cross-object interference.

## 3. Inventory management: from stochastic models to data-driven policies

Traditional inventory models formulate ordering as sequential decision-making under uncertain demand, lead times, holding costs, shortage costs, and operational constraints \cite{zipkin2000inventory}. Data-driven inventory research increasingly learns decisions from contextual information. The Big Data Newsvendor directly maps demand-related features to order quantities; predictive-to-prescriptive analytics estimates decisions conditional on covariates; and Smart Predict-then-Optimize trains predictors according to the downstream decision loss rather than prediction error alone \cite{ban2019bigdata,bertsimas2020prescriptive,elmachtoub2022spo}. ([PubsOnLine][6])

Recent learning-based work also applies deep reinforcement learning and transformers to inventory policies. For example, the inventory-management transformer automates order timing and order quantity decisions. Such methods demonstrate the value of reusable sequence models for inventory control, but their objective is normally expected operational performance under the learned or simulated environment—not identification of the causal effect of a process intervention \cite{liu2026inventorytransformer}. ([PubsOnLine][7])

Two inventory-specific data problems are especially relevant:

* **Demand censoring:** when inventory reaches zero, observed sales may be lower than latent demand. Consequently, the absence of a sale is not necessarily evidence of no demand.
* **Inventory-record inaccuracy:** the information-system stock level may differ from physical inventory.

Both problems can induce misleading training labels and confounding between recorded process state, ordering decisions, and outcomes \cite{besbes2013censoring,ding2024censored,dehoratius2008inaccuracy}. ([PubsOnLine][8])

## 4. Causal inference in operations and inventory management

Empirical operations-management research has explicitly imported causal methods to address endogeneity and selection bias. Ho et al. discuss how tools from empirical economics can support causal conclusions in operations settings \cite{ho2017causalom}. More recent work combines causal ideas with censored demand, including offline feature-based pricing under censored observations \cite{tang2025causalpricing}. ([PubsOnLine][9])

It is useful to distinguish this literature from older work using the expression **causal demand forecasting**. In parts of the forecasting literature, “causal” means that demand is predicted using explanatory variables such as prices, promotions, or weather. Such models may be useful for safety-stock planning, but an explanatory forecasting model is not necessarily an identified intervention model \cite{beutel2012causalforecasting}.

The closest direct bridge is *From Object-Centric Event Data to Causal Inventory Insights*. It reconstructs item–location trajectories from object-centric event data, derives indicators related to demand, supply, batching, and buffering, defines understock and overstock as time-in-state outcomes, and estimates a theory-constrained structural equation model \cite{berti2026causalinventory}. Importantly, its interpretation remains dependent on the supplied causal structure and assumptions; event data alone do not establish causality. ([TechRxiv][10])

This paper is therefore an important precursor, but a foundation model would extend it by learning across many inventory systems, causal structures, process variants, and intervention queries rather than estimating one fixed SEM for one extracted dataset.

## 5. Process event data as the “glue”

### 5.1 Why object-centric process data are needed

Conventional case-centric process mining assumes that every event belongs to one case, such as one order. Inventory processes violate this assumption. A receipt event may relate to multiple items, products, purchase-order lines, shipments, storage locations, and suppliers. A transfer affects both an origin and a destination. A supplier delay may influence many item–location units simultaneously.

Object-centric process mining allows one event to relate to multiple typed objects and therefore provides a more faithful representation of such processes \cite{vanderAalst2023ocpm,koren2024ocel2}. Research on object-centric Petri nets and quantity-aware object-centric process mining further addresses intertwined object lifecycles and material quantities rather than only object identities \cite{graves2024quantities}. ([MDPI][11])

Kretzschmann et al. integrate order-to-cash and purchase-to-pay data in an object-centric inventory model and enrich the event data with inventory metrics \cite{kretzschmann2025retail}. Together with the causal-inventory SEM, this work shows how process event data can connect operational execution, inventory states, and analytical models. ([Springer][12])

### 5.2 How event data instantiate causal concepts

| Causal concept        | Process-event-data realization                        | Inventory example                                                |
| --------------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| Unit of analysis      | Object or typed object tuple                          | Item–location, SKU–store, order line                             |
| Treatment or action   | Event, policy decision, or event attribute            | Expedite shipment, change supplier, raise reorder point          |
| Treatment time        | Timestamp of the decision or policy activation        | Time at which an emergency order is approved                     |
| Pretreatment history  | Event prefix and state reconstructed before treatment | Inventory position, recent demand, backlog, open orders          |
| Outcome               | Future event, state interval, or accumulated cost     | Stockout duration, service level, holding cost                   |
| Mediator              | Subsequent state or event sequence                    | Lead time, batching, allocation decision                         |
| Environment           | Process variant, site, period, or policy regime       | Store, warehouse, region, pre/post policy period                 |
| Interference graph    | Object relations and shared resources                 | Shared supplier, warehouse, transport lane, substitute product   |
| Censoring mechanism   | State-dependent observability of events               | Lost demand when stock is zero                                   |
| Support or positivity | Frequency of comparable action histories              | Whether both expedited and normal orders occur in similar states |

Process event data thus provide three forms of glue:

* **Semantic glue:** events connect causal variables to concrete operational actions and objects.
* **Temporal glue:** timestamps and event prefixes determine which variables can legitimately precede a treatment and prevent post-treatment leakage.
* **Relational glue:** object links reveal potential interference, shared resources, and hierarchical dependencies.
* **Decision glue:** estimated effects can be translated back into operational actions, process changes, and inventory policies.

However, an event log is **not itself a causal model**. Directly-follows relations express observed ordering, not intervention semantics. Recent work on causal execution dependencies shows that a conventionally discovered process model may disagree with a causal process view, especially in the presence of parallelism, confounding, or differing activity durations \cite{fournier2025why}. ([Springer][13])

## 6. Causal process mining and prescriptive process monitoring

Several process-mining contributions already combine event data with causal reasoning:

* Narendra et al. combine process logs, a BPMN model, and SCM assumptions to answer counterfactual process-improvement questions \cite{narendra2019counterfactual}.
* Qafari and van der Aalst use structural equation models for process root-cause analysis and develop case-level counterfactual reasoning \cite{qafari2020rootcause,qafari2021counterfactual}.
* Bozorgi et al. discover treatment-effect-based causal rules from event logs \cite{bozorgi2020causalrules}.
* Shoush and Dumas use causal inference for resource-constrained prescriptive process monitoring \cite{shoush2022prescriptive}.
* Leemans and Tax study causal dependencies between control-flow decisions and process outcomes \cite{leemans2022controlflow}.
* Fournier et al. distinguish conventional process ordering from causal execution dependencies \cite{fournier2025why}. ([Springer][14])

These methods are generally designed for one event log, one treatment definition, one outcome, or one process model. They do not yet provide a pretrained causal reasoner that transfers across organizations, object schemas, process variants, and causal queries.

A complementary line is foundation modeling for process mining. The in-context foundation model for predictive process monitoring is trained across heterogeneous event logs and adapts to a new log using contextual examples, supporting next-activity and remaining-time prediction \cite{berti2026processfoundation}. It demonstrates the feasibility of event-log-native cross-process pretraining. Its tasks, however, are predictive rather than causal, and its principal representation is case-centric rather than object-centric. ([IEEE Xplore][15])

## 7. Research gap and proposed positioning

The literature leaves a clear four-way gap:

| Research family                 | Main contribution                                | Missing component                                      |
| ------------------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| Causal foundation models        | Reusable causal discovery or effect estimation   | Native process, object, and inventory semantics        |
| Data-driven inventory models    | Strong forecasting and policy optimization       | Explicit identification and counterfactual validity    |
| Causal process mining           | Intervention-aware analysis of event logs        | Cross-log pretraining and broad task amortization      |
| Object-centric inventory mining | Faithful multi-object operational representation | General causal-world inference and reliability control |

The proposed model can therefore be positioned as:

> **An object-centric, query-conditioned causal foundation model pretrained over causal process worlds and adapted in context to heterogeneous operational event data.**

Its inputs would combine an object-centric event episode, a causal query, optional process or domain constraints, and information about treatments, environments, temporal scope, and interference. Its outputs should include:

1. a posterior over plausible causal structures, regimes, latent variables, and exposure mappings;
2. an identification or partial-identification result for the query;
3. an estimated causal effect, counterfactual distribution, or policy value when justified;
4. overlap, extrapolation, sensitivity, and calibration diagnostics;
5. an explicit abstention result when the available event data do not support the requested contrast.

Pretraining should use a synthetic and semi-synthetic **SCM-to-process-log generator**. The generator would first sample an inventory/process causal world and policy, then simulate demand, orders, receipts, transfers, stock states, resource competition, and censoring, and finally emit an OCEL-like log. This makes it possible to supply the model with both realistic event observations and otherwise unavailable ground-truth interventions and counterfactuals.

The uploaded benchmark families are well aligned with this design:

* regime experiments become warehouse, product-family, or policy-environment variation;
* interference becomes shared suppliers, capacity, transport, or product substitution;
* latent groups become unobserved SKU or location classes;
* hierarchical latent graphs become item–location–warehouse–region structures;
* temporal splits become policy changes and contemporaneous shocks;
* lagged VARX structures become delayed replenishment and demand effects.

### Paper-ready gap paragraph

> Existing causal foundation models amortize causal discovery or effect estimation over synthetically generated tabular problems, while inventory foundation and decision models primarily optimize predictions or policies without explicitly determining whether intervention effects are identified. Causal process-mining methods introduce intervention and counterfactual reasoning into event-log analysis, but they are generally tailored to individual logs, treatments, and outcomes. Object-centric process mining offers the missing representational layer: it preserves the temporal, relational, and quantitative interactions among orders, items, locations, shipments, suppliers, and resources. We therefore propose an object-centric causal foundation model that treats process event data as the interface between reusable causal reasoning and operational inventory decision-making, while explicitly representing structural uncertainty, identification, support, interference, temporal change, and the option to abstain.

## Proposed BibTeX bibliography

I assembled a **validated 42-entry BibTeX file** covering:

* causal modeling, discovery, temporal causality, and interference;
* causal effect, discovery, longitudinal, and sensitivity-analysis foundation models;
* classical, data-driven, censored-demand, and transformer-based inventory research;
* object-centric process mining and inventory modeling;
* causal process mining, counterfactual process analysis, and predictive process foundation models.

[Download the complete BibTeX bibliography](sandbox:/mnt/data/foundation_causality_related_work.bib)

The most central citation keys for the main narrative are:

```text
Causal foundations:
pearl2009causality
hernan2020whatif
peters2016invariant
huang2020heterogeneous
runge2019pcmci
hudgens2008interference

Causal foundation models:
ma2025causalfm
robertson2025dopfn
balazadeh2025causalpfn
thompson2026arrow
li2026tabcausal
qiao2026cdfm
reuter2026partialgraphs
thumm2026causaltimeprior
zare2026causallongpfn
javurek2026sensitivitypfn

Inventory and operations:
ban2019bigdata
bertsimas2020prescriptive
elmachtoub2022spo
besbes2013censoring
ding2024censored
ho2017causalom
tang2025causalpricing
liu2026inventorytransformer
dehoratius2008inaccuracy

Process-event-data glue:
vanderAalst2023ocpm
koren2024ocel2
graves2024quantities
kretzschmann2025retail
berti2026causalinventory

Causal and foundation-model process mining:
narendra2019counterfactual
qafari2020rootcause
qafari2021counterfactual
bozorgi2020causalrules
shoush2022prescriptive
leemans2022controlflow
fournier2025why
berti2026processfoundation
```

[1]: https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B?utm_source=chatgpt.com "Causality"
[2]: https://proceedings.neurips.cc/paper/2018/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html "https://proceedings.neurips.cc/paper/2018/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html"
[3]: https://arxiv.org/abs/2506.10914 "https://arxiv.org/abs/2506.10914"
[4]: https://arxiv.org/abs/2605.07204 "https://arxiv.org/abs/2605.07204"
[5]: https://arxiv.org/html/2603.11090v2 "https://arxiv.org/html/2603.11090v2"
[6]: https://pubsonline.informs.org/doi/10.1287/opre.2018.1757 "https://pubsonline.informs.org/doi/10.1287/opre.2018.1757"
[7]: https://pubsonline.informs.org/doi/10.1287/serv.2024.0236 "https://pubsonline.informs.org/doi/10.1287/serv.2024.0236"
[8]: https://pubsonline.informs.org/doi/10.1287/mnsc.1120.1654 "https://pubsonline.informs.org/doi/10.1287/mnsc.1120.1654"
[9]: https://pubsonline.informs.org/doi/10.1287/msom.2017.0659 "https://pubsonline.informs.org/doi/10.1287/msom.2017.0659"
[10]: https://www.techrxiv.org/doi/10.36227/techrxiv.177155632.23266403 "https://www.techrxiv.org/doi/10.36227/techrxiv.177155632.23266403"
[11]: https://www.mdpi.com/2227-7390/11/12/2691 "https://www.mdpi.com/2227-7390/11/12/2691"
[12]: https://link.springer.com/chapter/10.1007/978-3-031-94193-1_14 "https://link.springer.com/chapter/10.1007/978-3-031-94193-1_14"
[13]: https://link.springer.com/article/10.1007/s13218-024-00883-4 "https://link.springer.com/article/10.1007/s13218-024-00883-4"
[14]: https://link.springer.com/chapter/10.1007/978-3-030-26643-1_6 "https://link.springer.com/chapter/10.1007/978-3-030-26643-1_6"
[15]: https://ieeexplore.ieee.org/document/11366646/ "https://ieeexplore.ieee.org/document/11366646/"
