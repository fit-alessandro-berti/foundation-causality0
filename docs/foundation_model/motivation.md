## Overall assessment

Training a causality-specific foundation model **could mitigate several of the main obstacles that prevent causal methods from working reliably in real-life applications**, but only under a careful definition of what such a model should do.

It should not be conceived as a larger predictive model that always returns a causal graph or a point estimate. A more defensible goal is an **amortized, query-conditioned causal inference system** that:

* adapts its inductive bias to the data-generating mechanism;
* integrates tabular, temporal, network, and latent-variable information;
* propagates uncertainty across the full analysis pipeline;
* diagnoses overlap, confounding, misspecification, and out-of-distribution inputs;
* distinguishes association, Granger predictability, and intervention-level causality;
* abstains or returns partial-identification bounds when a point effect is not identified.

Recent work provides preliminary evidence that this is technically possible. CausalFM and Do-PFN pretrain transformer models on synthetic structural causal models to perform in-context causal-effect estimation, while Arrow and CDFM pursue zero-shot causal discovery across heterogeneous graph and mechanism families. However, much of this evidence is still synthetic or semi-synthetic, and the newest causal-discovery foundation models are recent preprints rather than evidence of dependable deployment in uncontrolled real-world settings. ([arXiv][1])

## What the ZIP package establishes

The ZIP does **not** contain a trained causal foundation model. It contains six executable synthetic benchmark pipelines covering:

1. observed causal regimes;
2. network interference and spillovers;
3. outcome-relevant latent groups;
4. relationships between latent groups;
5. contemporaneous temporal changes;
6. lagged VARX temporal changes.

The recorded experiment consists of 88 scenarios evaluated with five seeds, resulting in 440 discovery/evaluation pairs. The authors appropriately describe these as engineering experiments rather than publication-scale Monte Carlo studies. The results are nevertheless highly useful because they reveal what a causal foundation model would need to learn and, equally importantly, when it should refuse to make a strong causal claim.

All numerical results discussed below come from the package’s [benchmark report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md).

## Weaknesses exposed by the experiments and how a foundation model could address them

| Real-world weakness                                                      | Evidence in the ZIP experiments                                                                                                                                                                                                                                                                                                                                                                          | How a causal foundation model could mitigate it                                                                                                                                                                                                                                                                                                                                           | What it cannot eliminate                                                                                                                                                                            |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sensitivity to model class and method selection**                      | A global nonlinear mechanism was correctly left unsplit when the nonlinear basis was specified, but a misspecified linear model created a false regime in **5/5 runs**, while reducing test loss from **0.950 to 0.303**. In the interference benchmark, the simple linear estimator beat the flexible estimator for linear spillovers, whereas the flexible estimator was better for threshold effects. | Pretrain over diverse linear, nonlinear, threshold, interaction, and heteroskedastic mechanisms. Use a mixture-of-mechanism experts or Bayesian model averaging rather than committing to one estimator. Include a head that distinguishes genuine mechanism changes from a single globally nonlinear mechanism.                                                                          | The model can still fail when the real mechanism is absent from or very unlikely under its training prior.                                                                                          |
| **Low power for weak, sparse, or heterogeneous effects**                 | Weak regime changes had zero detection. “Only some outcomes change” had power **0.40**. Sparse spillover outcomes had power **0.40**, and mixed continuous/binary/count outcomes had outcome F1 **0.300**. Multiple and weak temporal breaks were completely missed.                                                                                                                                     | Share statistical strength through a common representation while retaining outcome-specific likelihoods and sparse hierarchical heads. Pretrain heavily on near-null, weak-effect, sparse-effect, and mixed-outcome episodes. Return posterior probabilities over effects rather than thresholded binary decisions.                                                                       | No model can reliably recover an effect that is too weak relative to the sample size and noise. It should express low information through broad uncertainty or abstention.                          |
| **Prediction is easier than structural or causal recovery**              | In the high-dimensional latent experiment, prediction remained close to oracle while group ARI was only **0.387** and support stability **0.495**. Under weak measurement, support F1 was **0.984**, but latent-score correlation fell to **0.709**. The predictive gain from adding network exposure was only **0.011**, even where causal spillover testing was meaningful.                            | Make predictive loss auxiliary rather than primary. Train directly on interventional distributions, causal effects, equivalence classes, latent structure, invariance, policy regret, calibration, and query consistency. Test compatibility across different subsets of variables and environments.                                                                                      | A predictive foundation model, even an extremely accurate one, does not automatically become a causal model.                                                                                        |
| **Hidden confounding and non-identification**                            | Under hidden confounding, the interference detector had apparent power **1.00**, but mean bias was **0.716** and interval coverage was **0.000**. This was the most severe false-confidence result in the package.                                                                                                                                                                                       | Train a separate identification and reliability subsystem. It should recognize when a query requires unconfoundedness, an instrument, a front-door variable, negative controls, interventions, or additional domain assumptions. When point identification is unavailable, return sensitivity curves, partial-identification bounds, or an equivalence class instead of a point estimate. | Hidden confounding cannot generally be “learned away” from the same observational distribution. Two causal models can produce the same observed data while implying different intervention effects. |
| **Lack of overlap or positivity**                                        | In the poor-exposure-overlap scenario, the target contrast was unsupported and end-to-end power was zero. This abstention was correct.                                                                                                                                                                                                                                                                   | Add an explicit support/overlap head that estimates whether the requested intervention is represented in the data. The model can reformulate the query for a supported target population, propose a modified intervention, or recommend additional data collection.                                                                                                                       | It cannot infer counterfactual outcomes in regions with no data support merely by being pretrained on other datasets.                                                                               |
| **Misspecified interference or exposure mappings**                       | Misspecified exposure produced ASE RMSE **0.220** and coverage **0.650**. Noisy or incomplete relation matrices also reduced power and accuracy.                                                                                                                                                                                                                                                         | Treat the network and exposure mapping as uncertain latent inputs rather than fixed truth. Use graph encoders, compare several plausible exposure mappings, condition on relational metadata, and average over mapping uncertainty. Return sensitivity bounds over plausible maps.                                                                                                        | If important ties or higher-order spillovers are entirely unobserved, the effect may remain only partially identified.                                                                              |
| **Error propagation through modular pipelines**                          | In the latent-graph benchmark, cross-loading graph F1 fell from **1.000** with true latent scores to **0.827** after score reconstruction and **0.829** after estimated grouping. Under weak measurement, F1 fell from **0.982 → 0.956 → 0.820**.                                                                                                                                                        | Jointly model measurement, latent grouping, graph structure, and the final causal query. Pass posterior distributions or soft assignments downstream instead of committing to a single estimated grouping. Train the system to account for measurement error and cross-loadings.                                                                                                          | An end-to-end model can hide where an error originated. Level-wise diagnostic probes like Levels A/B/C should therefore be retained.                                                                |
| **Complex temporal nonstationarity and ambiguous causal interpretation** | Multiple and weak breaks had zero detection. Gradual drift was detected in every run but localized near the nominal midpoint only **1/5** times. An intercept-only VARX change was falsely attributed to a lagged (B)-change in **2/5** runs.                                                                                                                                                            | Use multi-scale temporal or event-log encoders and structured heads for abrupt, gradual, recurrent, and multiple changes. Predict change intervals and change types—intercept, slope, noise, covariate process, autoregression, or treatment mechanism—rather than only a breakpoint.                                                                                                     | A change in (P(Y_t\mid X_t)) or a Granger-predictive relation is not automatically an intervention-level causal relation.                                                                           |
| **Unreliable calibration and absence of real causal ground truth**       | Each main no-spillover control rejected in **1/5** runs. The temporal procedures were conservative enough to miss important alternatives. Five seeds and 19 or 29 calibration draws are too few for precise error-rate conclusions.                                                                                                                                                                      | Combine learned posterior uncertainty with external calibration: null-rich simulation, permutation tests, cluster bootstrap, conformal or selective-risk procedures where applicable, and self-compatibility across variable subsets. Evaluate abstention and error detection as first-class outcomes.                                                                                    | A neural confidence score is not automatically a valid confidence interval, and real causal ground truth is usually unavailable.                                                                    |

External empirical work reinforces the first issue: causal-structure algorithms can change substantially with algorithm and hyperparameter choice, while tuning is especially difficult because the true graph is unavailable on real data. 

## 1. The main benefit would be amortized model selection, not merely greater model capacity

Many current causal workflows require the analyst to choose, before seeing the answer:

* a causal graph family;
* a linear or nonlinear structural equation model;
* an adjustment strategy;
* an exposure mapping;
* a lag order;
* a breakpoint penalty;
* a latent-variable representation;
* one of several effect estimators.

The package shows why this is dangerous. The same flexible model that helps with a threshold spillover is inferior to a simple linear estimator in the clean linear setting. Similarly, increasing predictive flexibility in the regime benchmark does not automatically protect against structural misspecification: a misspecified pooled model can use a spurious split to approximate a nonlinear response and achieve an apparently excellent validation loss.

A foundation model could mitigate this by learning a distribution over mechanisms and estimators. Rather than internally representing “the causal method,” it could represent a family such as

[
p(M,\theta,\mathcal G\mid D,Q,K),
]

where (D) is the data, (Q) is the causal query, (K) denotes domain constraints, (M) is the mechanism class, and (\mathcal G) is a graph or other causal structure. Predictions would average over plausible (M) and (\mathcal G), or the system could route the episode to specialized experts.

This is one reason prior-data fitted networks are relevant: they amortize inference over a distribution of synthetic problems, making a new dataset an in-context inference problem rather than requiring complete refitting and model selection from scratch. CausalFM explicitly formulates this in terms of priors over structural causal models and demonstrates it for back-door, front-door, and instrumental-variable settings. ([arXiv][1])

However, the package’s misspecified-nonlinearity experiment also provides a warning for foundation models: **a model with more capacity can find a more convincing wrong explanation**. The pretraining prior must therefore contain both:

* true mechanism changes; and
* stationary but complex mechanisms that can superficially resemble changes.

Otherwise, the model may simply learn a more sophisticated version of the same false-split behavior.

## 2. The model should be trained on causal targets, not predominantly predictive targets

Several experiments show that predictive performance is an inadequate proxy for causal quality:

* latent-group prediction can be nearly oracle while the grouping is unstable;
* a support estimate can look excellent while the recovered latent scores are poor;
* a false regime split can produce a large predictive gain;
* adding network exposure can have little predictive value while still mattering for a causal contrast.

A causality-specific foundation model should therefore be trained with several distinct objectives:

1. **Interventional objective:** predict (p(Y\mid do(A=a),X=x)), policy effects, or explicitly defined average effects.
2. **Structural objective:** recover graph skeletons, equivalence classes, latent groups, change mechanisms, and exposure structures.
3. **Query-consistency objective:** answers to related causal queries should satisfy algebraic and graphical constraints.
4. **Invariance and compatibility objective:** inferred mechanisms should remain compatible across environments and appropriate subsets of variables.
5. **Calibration and abstention objective:** unsupported or non-identified queries should receive low confidence, wide bounds, or an abstention decision.

Self-compatibility is especially useful when real causal ground truth is absent. It tests whether causal models inferred from different subsets of variables can be jointly compatible; incompatibility can falsify either the assumptions or the finite-sample output even when the true graph is unavailable. 

Predictive reconstruction can remain an auxiliary objective, but it should not dominate the representation.

## 3. Identification awareness is more important than another powerful estimator

The hidden-confounding experiment is the strongest result in the ZIP for motivating a causal foundation model—but also the strongest argument against a naive one.

The method detected an effect in every run, yet the estimated effect was severely biased and its interval never covered the truth. A model evaluated only by detection power would regard this as success. A scientifically appropriate causal system must regard it as failure.

For such a system, the output should separate at least three questions:

1. **Is there a statistical signal associated with treatment or exposure?**
2. **Is the requested causal quantity identified under the supplied graph and assumptions?**
3. **If it is identified, can it be estimated precisely from this sample?**

These are different tasks. The first can be easy while the second is impossible.

A reliability head can be useful, but it must not be represented as a formal identification theorem. Recent temporal causal PFN work trains heads for null probability, confounding, identifiability, mediation, and regime classification, but explicitly cautions that learned identifiability scores are proxies based on the training prior: identifiability is a property of the causal model and assumptions, not something recoverable from data patterns alone. ([arXiv][2])

A better architecture would combine learned diagnostics with symbolic or algorithmic identification:

* apply graph-based identification when a graph or partial graph is available;
* report a CPDAG or PAG when directions are observationally ambiguous;
* request an instrument or experimental environment where necessary;
* run negative-control and invariance checks;
* compute sensitivity or partial-identification bounds;
* abstain from a point estimate when assumptions cannot be justified.

Foundation-model training can make sensitivity analysis fast and reusable. Recent PFN work demonstrates amortized prediction of causal sensitivity bounds. Crucially, that work also makes the appropriate conceptual point: when a causal query is only partially identified, its exact location inside the valid interval cannot be learned from the observational data; the learnable quantities are the bounds themselves. ([arXiv][3])

Thus, the desired improvement in the ZIP’s hidden-confounding scenario is not necessarily a lower-RMSE point estimate. A more appropriate success criterion is:

> The model flags non-identification and returns bounds with valid coverage, or requests additional assumptions or data.

## 4. Support-aware inference should preserve the good abstention behavior

The poor-overlap interference experiment is one of the more encouraging results in the package: the target exposure contrast was unsupported, and the pipeline did not silently extrapolate.

A causal foundation model should preserve and generalize this behavior. Before estimating an effect, it should evaluate support at the level of the **specific query**, not merely ask whether both treatment values occur somewhere in the dataset. For network effects, this includes support for combinations such as own treatment (A=a) and neighborhood exposure (G=g).

When support is insufficient, the model could:

* redefine the effect for the common-support population;
* propose a smaller or stochastic intervention;
* indicate which clusters, treatment levels, or network exposures are missing;
* suggest an experimental allocation that would identify the target;
* produce an extrapolation result only when explicitly requested, clearly separating it from a data-supported causal estimate.

Pretraining can teach the model to recognize overlap patterns, but it cannot create information about outcomes under interventions that never occur in comparable observations.

## 5. Network interference requires uncertainty over the exposure representation

Most practical interference analyses begin by choosing an exposure mapping, for example the proportion of treated neighbours. That mapping is rarely known with certainty. It can omit distant neighbours, treat all edges equally when influence is weighted, or ignore higher-order spillovers.

The package demonstrates all three problems through misspecified, noisy, and incomplete relation matrices. A foundation model could improve this by treating exposure construction as part of inference rather than fixed preprocessing. A relational encoder could receive:

* the observed network;
* edge types and weights;
* spatial or organizational distances;
* partial domain constraints;
* candidate exposure definitions;
* uncertainty about missing links.

It could then average effects over plausible mappings or return a sensitivity surface. Recent work on causal inference under misspecified network exposure mappings derives sharp bounds for direct and spillover effects, showing how partial identification can replace unjustified certainty when the exposure representation is uncertain. ([arXiv][4])

This approach would be particularly valuable in real settings such as workplaces, hospitals, social networks, supply chains, and process systems, where the recorded network is rarely identical to the true influence network.

## 6. Joint training could reduce latent-variable error propagation

The three-level graph experiment has a useful design:

* Level A evaluates graph recovery using true latent scores;
* Level B introduces score reconstruction;
* Level C also introduces estimated grouping.

This decomposition shows where performance is lost. A conventional pipeline makes hard decisions at each stage:

[
X \rightarrow \widehat Z
\rightarrow \widehat{\text{groups}}
\rightarrow \widehat{\mathcal G}
\rightarrow \widehat{\text{effect}}.
]

Once a variable is assigned to the wrong latent group, the graph procedure generally treats that assignment as certain.

A foundation model could instead maintain a joint posterior:

[
p(Z,\text{groups},\mathcal G,\theta_Q\mid X,Y,Q),
]

so uncertainty about measurements and group membership is propagated into graph and effect uncertainty. Soft cross-loadings could be represented directly, and the model could learn from many measurement systems that different observed variables may be imperfect indicators of the same latent mechanism.

Nevertheless, the package’s A/B/C evaluation should not be discarded. A monolithic model could improve final performance while becoming harder to diagnose. The foundation model should therefore expose intermediate outputs corresponding to:

* measurement quality;
* group-membership uncertainty;
* latent-score reliability;
* graph stability;
* final effect uncertainty.

## 7. Temporal foundation models could help, provided the causal semantics remain explicit

The temporal experiments expose three distinct problems:

1. conservative methods lose power for weak and multiple changes;
2. abrupt-break metrics are inappropriate for gradual transitions;
3. a detected predictive change can be attributed to the wrong mechanism.

A temporal causal foundation model could be pretrained on a much richer collection of trajectories:

* no change;
* one or several abrupt changes;
* close breakpoints;
* gradual drift;
* recurrent regimes;
* coefficient, intercept, variance, covariate, and lag-order changes;
* latent confounding;
* missing and irregular observations;
* changing treatment assignment policies.

A multi-scale sequence or event-log encoder could output a posterior over transition windows rather than a single change point. A structured change-type head could distinguish:

[
\Delta A,\quad \Delta B,\quad
\Delta \text{intercept},\quad
\Delta \sigma^2,\quad
\Delta P(X),\quad
\text{gradual drift}.
]

This would directly target the package’s false attribution of an intercept change to (B).

The model must, however, label its conclusion correctly. The first temporal benchmark concerns changes in a conditional distribution, while the VARX benchmark concerns lagged predictive or Granger relationships. Neither alone proves an intervention-level effect. A causal foundation model should never translate “(X_{t-1}) improves prediction of (Y_t)” into “intervening on (X) changes (Y)” without the additional causal assumptions required for that statement.

## 8. A suitable training design

A practical model should operate on a **causal episode**, not just a matrix. One possible episode representation is

[
E=(D,S,R,K,Q),
]

where:

* (D) is the observed dataset;
* (S) describes variable roles, types, units, and missingness;
* (R) describes time, network, cluster, treatment-assignment, or process structure;
* (K) contains domain constraints or partial causal knowledge;
* (Q) is the exact causal query.

The output should include:

[
\bigl[
p(\theta_Q\mid E),
p(\mathcal G\mid E),
p(M\mid E),
\text{support},
\text{identification status},
\text{sensitivity bounds},
\text{OOD score}
\bigr].
]

### Pretraining distribution

The existing 440 benchmark episodes are far too few to train a foundation model, but they form a valuable initial curriculum. Their generators could be generalized to produce an effectively unbounded stream of episodes varying:

* sample size and dimensionality;
* graph density and topology;
* linear, nonlinear, discontinuous, and interaction mechanisms;
* mixed outcomes;
* observed and hidden confounding;
* instruments and front-door variables;
* selection and missingness;
* measurement error and cross-loadings;
* networks and exposure mappings;
* abrupt and gradual temporal changes;
* null, near-null, and strong effects;
* supported and unsupported causal queries.

An important principle is that a causal foundation model does not eliminate assumptions—it **moves many of them into the pretraining distribution**. The prior over mechanisms must therefore be explicit, versioned, and stress-tested. Recent causal foundation-model work similarly emphasizes the indispensable role of causal priors for generalization and identifiability. ([arXiv][5])

### Synthetic-to-real transfer

The model should not be trained solely on simple random SCMs. Purely synthetic benchmarks often omit the irregularities and dependencies that make real causal discovery difficult. Research on causal discovery without ground truth and on semi-synthetic industrial generators documents this gap and motivates combining domain knowledge, real covariate distributions, and partially known causal structures. 

A robust training mixture would combine:

* fully synthetic SCMs, where complete ground truth is available;
* semi-synthetic systems based on real covariate and process distributions;
* randomized and quasi-experimental datasets;
* real observational datasets with partial domain constraints;
* adversarial assumption-stress episodes.

Entire mechanism families—not merely random seeds—should be held out for evaluation. For example, all threshold interference mechanisms or all gradual-drift mechanisms could be excluded from training to test genuine out-of-distribution transfer.

## 9. How the existing package could evaluate the foundation model

No result in the ZIP directly proves that a foundation model would outperform the present pipelines. The correct next step is to turn its scenarios into explicit research hypotheses.

### Misspecification hypothesis

Train without one family of global nonlinear functions and test whether the model avoids the 5/5 false regime splits while preserving high power for genuine regimes. Evaluate both:

* structural false-positive rate;
* predictive improvement.

A model that obtains lower loss by inventing a regime has failed the causal task.

### Identification hypothesis

In the hidden-confounding interference scenario, score the following separately:

* effect RMSE;
* interval or bound coverage;
* probability of an “identified” declaration;
* abstention accuracy;
* sensitivity-bound validity.

A confidently biased point estimate should score worse than a correct non-identification decision.

### Support hypothesis

On the poor-overlap scenario, evaluate whether the model:

* refuses the unsupported contrast;
* correctly identifies the unsupported region;
* proposes a supported alternative estimand;
* avoids borrowing unsupported answers from its synthetic pretraining prior.

### Latent-pipeline hypothesis

Compare:

1. the current modular pipeline;
2. a foundation model with hard intermediate decisions;
3. a joint model with uncertainty propagation.

Continue reporting the Level A/B/C metrics so that an apparent final improvement cannot conceal degraded measurement or grouping.

### Network-mapping hypothesis

Hold out exposure mappings during training and evaluate:

* mapping-posterior calibration;
* ASE coverage;
* sensitivity-bound width;
* effect accuracy under noisy and incomplete networks.

This would test whether the model has learned transferable network causal reasoning rather than memorized a particular exposure formula.

### Temporal hypothesis

The central tests should be:

* several breakpoints;
* weak breakpoints;
* close breakpoints;
* gradual drift;
* intercept-versus-(B) attribution;
* contemporaneous-versus-lagged relationships.

For gradual drift, score overlap with the true transition interval, not only distance from an arbitrary midpoint.

### Calibration hypothesis

The package itself recommends at least 200 final seeds and 199 or more calibration repetitions. The foundation-model comparison should follow that recommendation. Evaluation should report:

* type-I error and power;
* structural F1, ARI, and stability;
* effect bias, RMSE, and coverage;
* calibration curves;
* selective risk as a function of abstention;
* policy regret;
* performance under unseen mechanism families.

A learned posterior should also be compared with externally calibrated versions using permutation or bootstrap procedures. A foundation model may greatly reduce computation by providing a strong initial posterior, while a conventional calibration layer provides the final error-control guarantee.

## What a causal foundation model would not solve

Even a very large and well-trained model cannot by itself resolve:

* hidden confounding without additional assumptions, proxies, instruments, or interventions;
* absent overlap for the requested intervention;
* causal direction inside an observational equivalence class;
* an incorrectly formulated or scientifically meaningless estimand;
* interference through an entirely unobserved network;
* external validity from one population or policy environment to another;
* synthetic-to-real prior mismatch;
* formal confidence coverage merely through neural uncertainty estimates.

The foundation model can help **recognize, communicate, and manage** these limitations. It cannot make them disappear.

## Conclusion

The experiments in the ZIP support a strong but qualified case for a causality-specific foundation model.

Its main value would not be that it is universally more flexible than existing estimators. Indeed, the package shows that flexibility can worsen causal conclusions: a nonlinear misspecification created convincing false regimes, and a flexible spillover estimator lost to a simple linear estimator in the correctly linear setting.

The more compelling advantages are:

1. **amortized selection and combination of causal mechanisms and estimators;**
2. **shared learning across weak, sparse, mixed, network, latent, and temporal settings;**
3. **joint propagation of measurement and structural uncertainty;**
4. **explicit identification, overlap, and misspecification diagnostics;**
5. **fast sensitivity analysis, partial identification, and calibrated abstention;**
6. **clear separation of predictive, associational, Granger, and interventional claims.**

The hidden-confounding experiment should be the defining acceptance test. A foundation model that detects the effect with power 1.00 but returns a biased point estimate with zero coverage has not mitigated the weakness—it has merely automated false confidence. A genuinely causal foundation model should instead recognize that the effect is not identified under the available information, state which assumptions or additional data are needed, and return appropriate bounds or abstain.

### Package files examined

* [Complete benchmark report](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/REPORT.md)
* [Observed-regime benchmark design](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/causal_model_determination.md)
* [Causal-interference benchmark design](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/causal_interference-detection.md)
* [Latent-variable benchmark design](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/latent_variable_determination.md)
* [Latent-graph benchmark design](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/latent_variable_determination2.md)
* [Contemporaneous temporal-split benchmark](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection.md)
* [Lagged VARX temporal benchmark](sandbox:/mnt/data/foundation_causality_full/foundation-causality0-main/docs/data/temporal_split_detection_method.md)

[1]: https://arxiv.org/abs/2506.10914 "[2506.10914] Foundation Models for Causal Inference via Prior-Data Fitted Networks"
[2]: https://arxiv.org/html/2606.20889v1 "Temporal Causal Prior-Data Fitted Networks for Panel Data with Learned Reliability Signals"
[3]: https://arxiv.org/html/2605.10590v1 "Amortizing Causal Sensitivity Analysis via Prior Data-Fitted Networks"
[4]: https://arxiv.org/abs/2602.03459 "[2602.03459] Causal Inference on Networks under Misspecified Exposure Mappings: A Partial Identification Framework"
[5]: https://arxiv.org/abs/2607.11508 "[2607.11508] CDFM: Towards a General-Purpose Causal Discovery Foundation Model"
