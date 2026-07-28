## Main conclusion

After inspecting the Phase-I implementation and its saved predictions, I would **not increase the model size first**. The present result is constrained mainly by the way the neural model is connected to the estimators, several task–objective mismatches, and one important uncertainty bug.

Recent causal PFN work supports the basic idea of pretraining on synthetic SCMs and predicting interventional quantities in context. The weak point is therefore not the foundation-model premise itself, but the estimator and decoder architecture wrapped around the backbone. ([arXiv][1])

The highest-priority findings are:

| Finding in the current implementation                                                             | Consequence                                                        |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| The final estimate is always a 50/50 average of the direct and compiled paths                     | A weak direct decoder degrades an already strong compiled estimate |
| The differentiable “linear anchor” is not query-correct                                           | The direct decoder starts from a badly biased estimate             |
| The router is trained to recognize mechanism labels rather than select the lowest-risk estimator  | It routes to flexible estimators even when they perform worse      |
| The eight world weights are effectively uniform in every scenario                                 | The causal-world posterior is not being used                       |
| The particle variance formula is incorrect                                                        | The direct uncertainty loss and raw intervals are distorted        |
| The regime head jointly compares “none” against every variable and does not predict the threshold | It learns a conservative no-split shortcut                         |

The first four changes below are especially likely to improve results.

---

## 1. Replace the fixed hybrid with a safe, learned residual stack

The current code uses:

```python
compiled = (1 - p_flexible) * linear + p_flexible * flexible
hybrid = 0.5 * direct + 0.5 * compiled
```

in `cwfm/experiment.py:90–93`.

Because the hybrid weight is exactly 0.5, the compiled estimate can be reconstructed from the saved results:

[
\widehat\tau_{\text{compiled}}
==============================

## 2\widehat\tau_{\text{hybrid}}

\widehat\tau_{\text{direct}}.
]

Across the 180 supported and identified effect episodes:

| Estimator reconstructed from the current results |       MAE |
| ------------------------------------------------ | --------: |
| Direct neural path                               |     0.424 |
| Current 50/50 hybrid                             |     0.271 |
| **Compiled path alone**                          | **0.169** |

Thus, in this run, the fixed average increased MAE by approximately **61% relative to the compiled path**, or equivalently removing the harmful direct contribution would reduce MAE by about **38% relative to the current hybrid**.

This is a post-hoc diagnostic, so 0.169 should not be presented as a new confirmatory benchmark result. Nevertheless, it identifies the clearest architectural problem.

### Recommended formulation

Make the compiled estimator the default and let the neural model predict a gated correction:

[
\widehat\tau
============

\widehat\tau_{\mathrm{compiled}}
+
\alpha_\theta(\mathcal E,q)
\left(
\widehat\tau_{\mathrm{direct}}
------------------------------

\widehat\tau_{\mathrm{compiled}}
\right),
\qquad
0\leq\alpha_\theta\leq1.
]

Better still, have the neural decoder directly predict a correction:

[
\widehat\tau
============

\widehat\tau_{\mathrm{compiled}}
+
\alpha_\theta(\mathcal E,q),
\widehat\delta_\theta(\mathcal E,q).
]

The gate should receive:

* estimated out-of-fold risks of the candidate estimators;
* direct–compiled disagreement;
* overlap and effective sample size;
* mechanism probabilities;
* posterior or ensemble dispersion;
* task and formal query;
* prior-mismatch score.

Initialize the gate bias so that (\alpha\approx0). This gives the system a **do-no-harm initialization**: before the neural correction has learned anything useful, the model reproduces the classical estimator rather than averaging it with noise.

During synthetic pretraining, the ideal gate target can be computed from oracle estimand errors. On real or semi-synthetic data, it can use cross-fitted pseudo-risk. The aggregation rule and all thresholds should be frozen using a separate validation seed range before final evaluation.

---

## 2. Replace the present anchor with a query-correct orthogonal estimator layer

The current anchor in `cwfm/model.py:187–217` has several problems:

1. It has no intercept.
2. For network interference, it selects exposure (G) as the focal variable but omits own treatment (A).
3. It omits the (A\times G) interaction.
4. It multiplies the exposure coefficient by an unconditional constant 0.5.
5. It is not generated from the formal intervention contrast.
6. It is a plain plug-in regression rather than an orthogonal or doubly robust estimate.

Evaluating only the anchor formulas on the existing effect episodes gives:

| Formula                        | Pooled MAE |
| ------------------------------ | ---------: |
| Present differentiable anchor  |      0.381 |
| Query-correct linear estimator |      0.169 |

This does not mean that replacing the anchor automatically makes the complete direct network obtain 0.169 after retraining. It does show that the current residual decoder is being asked to correct a needlessly poor starting point.

### Static ATE branch

The static branch should produce nuisance estimates

[
\widehat\mu_0(x),\quad
\widehat\mu_1(x),\quad
\widehat e(x)=P(A=1\mid X=x),
]

and then use a differentiable one-step/AIPW layer:

[
\widehat\tau_{\mathrm{DR}}
==========================

\frac1n\sum_i
\left[
\widehat\mu_1(X_i)-\widehat\mu_0(X_i)
+
\frac{A_i(Y_i-\widehat\mu_1(X_i))}
{\widehat e(X_i)}
-----------------

\frac{(1-A_i)(Y_i-\widehat\mu_0(X_i))}
{1-\widehat e(X_i)}
\right].
]

For randomized episodes, the known or declared assignment probability should be used rather than estimated unnecessarily.

### Network-interference branch

The network branch should estimate a response surface

[
\widehat\mu(a,g,x)
==================

\mathbb E[Y\mid A=a,G=g,X=x]
]

and evaluate exactly the contrast specified in the query. Its candidate basis should include at least:

[
A,\quad G,\quad A,G,\quad
\text{spline}(G),\quad
\mathbf 1(G>c),\quad
A,\mathbf 1(G>c).
]

Cross-fitting must hold out whole clusters rather than individual nodes.

### Neural role

The transformer should predict nuisance functions and a small residual correction around the orthogonal estimate, rather than learning an unconstrained scalar effect from scratch. Orthogonal scores and cross-fitting are designed specifically to prevent regularization and nuisance-prediction errors from entering the causal target at first order. Dragonnet’s targeted regularization follows a related principle: train the representation and nuisance heads in a way aligned with the downstream causal estimand. ([arXiv][2])

This change is likely to restore strong performance in simple linear and randomized scenarios while retaining room for learned corrections under nonlinear mechanisms.

---

## 3. Convert the mechanism MoE into a risk-supervised estimator MoE

At present, the router is trained using labels such as:

```python
route_flexible = float(scenario == "nonlinear")
```

or

```python
route_flexible = float(scenario == "threshold")
```

in `cwfm/data.py`.

That asks the model:

> “Was this episode generated by a nonlinear family?”

But the relevant question is:

> “Which estimator has the lowest expected error for this estimand at this sample size, support level and signal strength?”

Those questions are not equivalent. In the current evaluation:

* static nonlinear: linear MAE (0.218), AIPW MAE (0.242);
* network threshold: linear MAE (0.311), random-forest MAE (0.351).

Thus, the designated “flexible” estimator is not actually the best estimator in either of the two scenarios labelled flexible.

The router confirms that it is not learning episode-specific selection:

| Scenarios         | Mean flexible probability |
| ----------------- | ------------------------: |
| Static linear     |                    0.4300 |
| Static nonlinear  |                    0.4306 |
| Network linear    |                    0.3363 |
| Network threshold |                    0.3363 |
| Network null      |                    0.3364 |

It is largely identifying the task rather than the mechanism or expected estimator risk.

### Recommended estimator expert library

Use different candidate libraries by task.

For static ATE:

1. linear/GLM nuisance expert;
2. additive spline expert;
3. interaction and heterogeneous-effect expert;
4. tree or forest expert;
5. neural nuisance expert;
6. randomized-design difference-in-means expert.

For network interference:

1. linear (A,G,A\times G) expert;
2. smooth spline dose–response expert;
3. piecewise/threshold expert;
4. graph-neural response expert;
5. null-shrinkage expert.

A generic quantile-threshold scan on the current threshold episodes produced an exploratory MAE of about **0.150**, compared with 0.311 for the linear estimator. This is post-hoc and should not be treated as a benchmark result, but it indicates that an explicit piecewise expert is far more promising than expecting a generic random forest or latent MLP expert to discover the complete exposure-response structure from 96 observations.

### Train the router using risk

For synthetic episodes, construct a soft oracle target:

[
\pi_m^\star
===========

\frac{\exp[-L_m/T]}
{\sum_j\exp[-L_j/T]},
\qquad
L_m=
\ell(\widehat\tau_m,\tau).
]

Then train with both:

[
\ell!\left(\sum_m\pi_m\widehat\tau_m,\tau\right)
+
\lambda
\operatorname{KL}(\pi^\star|\pi_\theta).
]

For real-data adaptation, use cross-validated doubly robust pseudo-risk rather than mechanism labels. Selective-machine-learning work explicitly develops cross-validated pseudo-risk criteria for choosing among nuisance learners according to the resulting causal functional, rather than according to ordinary predictive loss. ([arXiv][3])

An oracle that selects the better of just the current linear and flexible estimators episode by episode has MAE about **0.130**. That is not achievable directly because it uses the hidden truth, but it quantifies substantial headroom for a competent router.

### Couple the two routers

The generic hidden-state MoE in `SparseMechanismMoE` and the final `flexible_route_head` are currently separate. The experts that specialize the representation should also be the experts producing or parameterizing the candidate estimators. Otherwise, “expert specialization” in the backbone has no direct relationship to estimator selection.

Add:

* load-balancing loss;
* minimum expert utilization;
* routing stability across row bootstrap/permutation perturbations;
* consistency between mechanism and estimator routing;
* optional router freezing after an initial specialization phase.

Expert Choice and StableMoE both address undertrained experts, load imbalance and unstable assignments, although their routing methods would need adaptation from token routing to episode-level causal experts. ([arXiv][4])

---

## 4. Repair the causal-world posterior and mixture objective

The reported world entropy is essentially

[
2.07944 = \log 8
]

for every scenario. Consequently, all eight particles receive nearly uniform weight regardless of the observed data. The “posterior over causal worlds” is currently behaving like eight equally weighted residual heads, not a posterior that concentrates on plausible worlds.

### Fix the variance bug first

In `cwfm/model.py:240–243`, the code computes the second moment of the **residual particles**:

[
\sum_s w_s(\sigma_s^2+\mu_s^2),
]

but subtracts the squared **full effect mean**:

[
(\text{anchor}+\sum_s w_s\mu_s)^2.
]

The correct residual variance is:

[
V_R
===

## \sum_s w_s(\sigma_s^2+\mu_s^2)

\left(\sum_s w_s\mu_s\right)^2.
]

Equivalently, define full component means (m_s=a+\mu_s) and use:

[
V
=

## \sum_s w_s(\sigma_s^2+m_s^2)

\left(\sum_s w_sm_s\right)^2.
]

The present expression can become negative solely because the anchor is nonzero and is then silently clamped. That damages scale learning and the Gaussian NLL gradient.

### Use the exact mixture likelihood

The current training loss first moment-matches all particles to one Gaussian. That removes much of the incentive to learn genuinely different worlds. Instead use:

[
\mathcal L_{\mathrm{mix}}
=========================

-\log
\sum_s
w_s
\mathcal N
\left(
\tau;
a+\mu_s,
\sigma_s^2
\right).
]

### Make particles structural, not merely scalar

Each particle should decode:

* graph or partial graph;
* mechanism family;
* identification strategy;
* regime structure;
* exposure mapping;
* nuisance functions;
* corresponding causal estimate.

The particle’s effect must be computed **through its decoded world**, rather than predicted by an unrelated linear head. This couples structural uncertainty to estimand uncertainty.

Because the simulator knows the generating graph and mechanism, particle specialization can be supervised using:

* graph likelihood;
* mechanism likelihood;
* exposure-mapping likelihood;
* posterior responsibility targets;
* permutation-invariant matching between particles and true latent worlds;
* diversity of decoded structures rather than merely maximizing the standard deviation of scalar effects.

The current diversity term,

```python
-output["particle_means"].std(-1).mean()
```

encourages numerical separation even when that separation has no causal meaning.

A pragmatic alternative is to temporarily use a **single residual head**. Eight uniform worlds with a malformed variance objective are more complex but not more informative than one well-trained estimator.

---

## 5. Use a shared encoder with task-specific causal decoders

The current generic effect head is trained on all identified and supported episodes, including regime episodes, in `cwfm/train.py:27–34`. Thus the same scalar head is asked to represent:

* a static ATE;
* a network exposure contrast;
* a regime-change magnitude.

These quantities have different statistical structures and should not share the final decoder.

A better organization is:

| Shared component           | Task-specific component           |
| -------------------------- | --------------------------------- |
| typed cell/schema encoder  | static nuisance and ATE decoder   |
| variable/sample attention  | network exposure-response decoder |
| query and design encoder   | regime structure decoder          |
| general mechanism features | task-specific uncertainty head    |
| partial graph encoder      | task-specific estimator library   |

Use small task adapters or low-rank task-specific projections in the upper encoder blocks. The lower representation can remain shared.

This should reduce negative transfer without abandoning the foundation-model objective.

---

## 6. Redesign the regime branch as detection followed by localization

The present regime decoder outputs one logit for every valid variable plus a no-split logit. It also masks only padding, not role-ineligible variables, in `cwfm/model.py:245–249`.

Its handcrafted instability statistic:

* tests candidate variables only at threshold zero;
* uses one predictor–outcome product;
* does not estimate the true threshold;
* does not represent change type;
* is then added to a flat multiclass classifier.

This makes “no split” an easy global shortcut, explaining:

* strong-split exact accuracy: 0.067;
* weak-split exact accuracy: 0;
* stationary nonlinear accuracy: 0.967;
* null accuracy: 1.000.

### Recommended hierarchical decoder

Use:

[
p(\text{any split}\mid\mathcal E)
]

followed, conditionally, by:

[
p(j,c,t,\Delta\mid \text{split},\mathcal E),
]

where:

* (j): regime variable;
* (c): change type, such as slope/intercept/variance;
* (t): threshold or interval;
* (\Delta): change magnitude.

Construct candidate tokens for variable–threshold pairs:

[
z_{j,k}
=======

\text{evidence for split on variable }j
\text{ at quantile }q_k.
]

Each token can contain:

* left/right sample sizes;
* standardized coefficient difference;
* residual variance change;
* likelihood or BIC improvement;
* cross-fitted predictive improvement;
* threshold quantile;
* uncertainty of the difference.

Then use a set decoder:

```text
global split gate
        ↓
variable × threshold candidate attention
        ↓
change-type and magnitude decoder
        ↓
STOP / additional split
```

Important details:

* mask the output to eligible regime variables;
* train split/no-split separately from localization;
* use conditional class balancing for variable localization;
* include matched hard negatives: stationary nonlinear mechanism versus true slope instability;
* vary thresholds continuously during pretraining;
* report split detection, variable localization and threshold error separately.

This should substantially improve strong-split power while retaining the current low false-positive rate. Recovery of the weakest (0.18)–(0.38) changes at (n=96) is less certain; some weak episodes may simply contain insufficient information.

---

## 7. Add a query-conditioned statistical-summary stream

The model currently asks three small attention blocks to rediscover basic regression and testing statistics from raw rows. Its covariance feature is an uncentered (X^\top X/n), and source-pair information is largely collapsed by summation.

A more sample-efficient architecture would have two parallel inputs:

```text
raw observation tokens ─────────────┐
                                    ├─ cross-attention → shared encoder
statistical summary tokens ─────────┘
```

Useful summary tokens include:

* centered means, variances and quantiles;
* centered pairwise covariance and correlation;
* treatment-group conditional moments;
* treated/control counts and effective sample size;
* propensity and overlap quantiles;
* outcome residual moments;
* (A\times X), (G\times A) and spline/hinge cross-moments;
* candidate regime-instability statistics;
* cluster saturation and exposure distributions;
* graph degree and connected-component summaries.

These are not intended to replace raw-data attention. They provide the model with statistically meaningful low-variance features while retaining the raw route for mechanisms not represented by the summaries.

### Use real query tokens

The current query consists only of task and design embeddings. Add explicit tokens for:

* treatment variable;
* outcome variable;
* intervention values;
* exposure contrast;
* own-treatment value or distribution;
* target population;
* estimand;
* time horizon;
* identification strategy.

The intervention values should cross-attend to variable and summary tokens. Adding the same task vector to every cell is much weaker than query-token attention.

---

## 8. Replace the current network message with a hierarchical cluster-aware encoder

In the existing axial block, all variables for a row are averaged into one row summary, adjacency propagation is applied, and the same relation message is broadcast back to every variable. This loses the distinction between:

* baseline covariates;
* own treatment;
* neighbor exposure;
* outcome;
* graph-derived characteristics.

Use three levels instead:

1. **Unit encoder**
   [
   h_i=f(X_i,A_i,\text{degree}_i,\ldots).
   ]

2. **Graph/exposure encoder**
   [
   \widetilde h_i=
   \operatorname{GAT}
   \left(h_i,{h_j:W_{ij}=1}\right).
   ]

3. **Cluster encoder**
   [
   c_k=
   \operatorname{Pool}
   {\widetilde h_i:i\in k}.
   ]

The causal query attends jointly to unit and cluster tokens. Independent clusters should be blocked from exchanging outcome information during nuisance estimation, and cross-fitting should hold out clusters.

Where partial process, organizational or causal graphs are known, inject them as learnable attention biases rather than only adding post-attention graph messages. Recent CFM experiments found attention-bias conditioning particularly effective for using complete or partial graph information. ([arXiv][5])

---

## 9. Make formal identification deterministic and uncertainty adaptive

The current safety heads receive design tokens explicitly saying “nonidentified” or “poor support.” Their nearly perfect separation therefore demonstrates metadata compliance rather than discovery of hidden confounding.

Use two separate components:

1. **Formal identification engine:** deterministic or rule-based from the declared graph/design/assumptions.
2. **Empirical diagnostics:** learned overlap, prior mismatch, negative controls and sensitivity analysis.

This preserves perfect compliance while removing a trivial classification task from the neural representation loss. As the model expands to back-door, IV, front-door and longitudinal settings, use identification-strategy-specific priors or adapters selected by the formal query. A recent CausalFM formulation explicitly argues that identifiability assumptions must be represented in the PFN prior or conditioning information rather than mixing incompatible causal settings indiscriminately. ([arXiv][6])

For uncertainty, after fixing the mixture variance, predict a heteroskedastic scale and conformalize normalized residuals:

[
r_i=
\frac{|\tau_i-\widehat\tau_i|}
{\widehat\sigma_i}.
]

Calibration can be conditioned or stratified by:

* task;
* selected estimator family;
* support level;
* sample size;
* network versus i.i.d. design;
* direct–compiled disagreement.

Conformalized quantile regression is one established way to obtain intervals whose lengths adapt to heteroskedasticity rather than using one absolute-error radius for an entire task. ([arXiv][7])

This is likely to improve scenario-level coverage and interval width, although it will not directly improve point MAE.

---

## Recommended revised architecture

```text
FORMAL QUERY
treatment, outcome, contrast, estimand, population,
identification strategy, time/exposure specification
                         │
                         ▼
              FORMAL IDENTIFICATION ENGINE
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│ MULTI-STREAM EPISODE ENCODER                              │
│                                                           │
│ raw observation stream                                    │
│ statistical-summary stream                               │
│ variable/schema stream                                    │
│ graph/cluster/process stream                              │
│ partial-knowledge attention biases                        │
└───────────────────────────────────────────────────────────┘
                         │
                  task-specific adapter
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
 static nuisance    network response    regime candidate
    experts             experts             decoder
 μ0, μ1, e          μ(a,g,x), P(A,G)    split/j/t/type
          │              │              │
          └──── cross-fitted estimators ┘
                         │
              RISK-SUPERVISED ESTIMATOR MoE
                         │
                  compiled anchor
                         │
        gated neural/world-conditioned residual
                         │
      adaptive uncertainty, calibration, abstention
```

---

## Ablation order I would use

| Model  | Change from current implementation                                                   | Main expected signal                                                                   |
| ------ | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **A0** | Current checkpoint                                                                   | Reference                                                                              |
| **A1** | Correct variance, correct anchor, compiled-only or validation-learned residual stack | Effect MAE should approach the current compiled-path level                             |
| **A2** | Add risk-supervised linear/spline/threshold/network experts                          | Improvement in nonlinear and threshold cells                                           |
| **A3** | Add task-specific decoders and hierarchical regime head                              | Higher strong-split power without a large stationary-nonlinear false-positive increase |
| **A4** | Add summary tokens and hierarchical network encoder                                  | Better small-(n) efficiency and network threshold performance                          |
| **A5** | Replace scalar particles with structurally coupled worlds and exact mixture NLL      | Better direct-path likelihood, world utilization and uncertainty                       |
| **A6** | Scale hidden dimension, blocks and pretraining episodes                              | Test whether capacity is still limiting after architectural corrections                |

For the next confirmatory run, the blending and routing rules should be selected on a new validation range, followed by a completely untouched evaluation range. Besides ordinary MAE and coverage, I would report:

* router regret relative to the best candidate estimator;
* frequency with which the neural residual improves versus harms the compiled anchor;
* world-weight entropy and structural disagreement;
* split detection, localization and threshold error separately;
* coverage conditional on mechanism and support;
* interval width;
* performance as (n), signal strength and prior mismatch vary.

The strongest immediate version would therefore be **a query-correct, cross-fitted compiled estimator with a zero-initialized neural residual, followed by a risk-supervised estimator MoE**. That redesign directly addresses the largest measured losses and is much more likely to improve the next experiment than simply moving from the current 96-dimensional backbone to a larger transformer.

[1]: https://arxiv.org/abs/2506.07918 "[2506.07918] CausalPFN: Amortized Causal Effect Estimation via In-Context Learning"
[2]: https://arxiv.org/abs/1608.00060 "https://arxiv.org/abs/1608.00060"
[3]: https://arxiv.org/pdf/1911.02029 "https://arxiv.org/pdf/1911.02029"
[4]: https://arxiv.org/abs/2202.09368 "https://arxiv.org/abs/2202.09368"
[5]: https://arxiv.org/abs/2602.14972 "https://arxiv.org/abs/2602.14972"
[6]: https://arxiv.org/html/2506.10914v3 "Foundation Models for Causal Inference via Prior-Data Fitted Networks"
[7]: https://arxiv.org/abs/1905.03222 "https://arxiv.org/abs/1905.03222"
