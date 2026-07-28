# Guardrails against overfitting in the revised CWFM

Because training episodes are generated online, ordinary memorization of a finite training set is not the main danger. The more important risks are:

| Risk                                | Example in this project                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Simulator-prior overfitting**     | Excellent results on the exact SCM families in `data.py`, but failure on a new nonlinear function, graph generator, sample size, or process model |
| **Shortcut learning**               | Inferring identification or support directly from the design token instead of examining the causal assumptions and empirical data                 |
| **Estimator-selection overfitting** | The router learns which expert looks best in-sample rather than which expert generalizes                                                          |
| **Validation overfitting**          | Repeatedly modifying the architecture after inspecting the same evaluation scenarios                                                              |
| **Component overfitting**           | The residual decoder, router, regime localizer, or world particles become more confident without improving causal risk                            |
| **Negative transfer**               | Improvements on static ATE come at the expense of network interference or regime detection                                                        |

A decreasing training loss alone will not distinguish these cases.

## 1. Use five strictly separated episode streams

The current validation seeds are disjoint from the training seeds, which is useful, but both sets come from the same generator code, parameter ranges, fixed dimensions, and mechanism families. That mainly tests whether the network memorizes individual samples. It does not test whether it has learned a transferable causal procedure.

I recommend the following separation:

| Stream                   | Content                                                                                                   | Permitted use                                                |
| ------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Training stream**      | Random online episodes from training SCM families                                                         | Gradient updates                                             |
| **ID development bank**  | Fixed episodes from the same SCM families, with disjoint latent SCMs and seeds                            | Early stopping and checkpoint selection                      |
| **OOD development bank** | Fixed episodes containing held-out mechanism families, graph families, parameter ranges, and combinations | Robustness constraint during model selection                 |
| **Calibration bank**     | Fresh episodes never used for weights or checkpoint selection                                             | Interval calibration, abstention thresholds, gate thresholds |
| **Final test bank**      | Locked scenarios and seeds                                                                                | One-time reporting only                                      |

The important unit of separation is the **SCM template or environment**, not merely the random seed. This follows the same principle as domain-shift evaluation: holding out examples from a familiar environment can substantially underestimate failure on genuinely new environments. ([Proceedings of Machine Learning Research][1])

### Suggested OOD partitions

For static ATE, hold out entire outcome and treatment mechanism families. For example, train on linear, sine, quadratic, and tanh functions, while validating on hinges, piecewise polynomials, multiplicative interactions, and saturation functions. Also vary:

* (n), instead of always using (n=96);
* (p), instead of fixed task-specific dimensionality;
* treatment prevalence;
* overlap strength;
* outcome noise and heteroskedasticity;
* treatment-effect heterogeneity;
* irrelevant and proxy variables;
* missingness mechanisms.

For interference, hold out:

* graph generators such as block, Erdős–Rényi, small-world, preferential-attachment, and degree-corrected block models;
* cluster sizes and graph densities;
* exposure mappings;
* thresholds other than 0.5;
* nonlinear and non-monotone spillover functions;
* interactions between own treatment and exposure.

For regime detection, hold out:

* threshold ranges or quantiles;
* change types, such as slope, intercept, variance, and nonlinear-form changes;
* multiple candidate regime variables;
* multiple splits;
* stationary nonlinear hard negatives;
* combinations of weak signal and small subgroup size.

When process event data is introduced, the equivalent guardrail is to hold out entire **process-model families or organizations**, rather than simply holding out traces generated by a process model also seen in training.

Every generated episode should carry metadata such as:

```text
generator_version
template_id
mechanism_family
parameter_bucket
graph_family
sample_size
variable_count
signal_strength
noise_level
overlap_bucket
```

These fields should be used for evaluation grouping, but should not automatically be provided as model inputs.

---

## 2. Randomize away simulator shortcuts

The simulator should make irrelevant regularities difficult to exploit.

At training time, randomly vary row order, role-preserving column order, sample size, variable count, variable scale, noise scale, graph node order, and the number of irrelevant variables. Include paired episodes in which the statistical or causal content is unchanged but presentation differs.

Useful paired tests include:

[
\widehat\tau(D,q)
\approx
\widehat\tau(\operatorname{permuteRows}(D),q),
]

and, after correctly remapping the query,

[
\widehat\tau(D,q)
\approx
\widehat\tau(\operatorname{permuteVariables}(D),\operatorname{permute}(q)).
]

Also create two independently sampled datasets from the same latent SCM. A model that has learned the causal procedure should give estimates consistent up to sampling error; a model exploiting accidental sample patterns may not.

### Add a metadata-only shortcut baseline

Train a very small diagnostic model using only:

* task;
* design;
* (n);
* (p);
* missingness rate;
* generator metadata that the main model receives.

It should not predict the effect accurately. If it does, the generator contains leakage or an unintended shortcut.

For formal identification, the strongest solution remains to move deterministic identification logic outside the learned model. Then log whether the learned empirical diagnostics agree with the formal engine rather than training the network to rediscover a label already encoded in the design token.

---

# Component-specific training guardrails

## 3. Protect the compiled estimator from a harmful neural residual

For the revised residual architecture,

[
\widehat\tau_{\mathrm{final}}
=============================

\widehat\tau_{\mathrm{compiled}}
+
\alpha_\theta \widehat\delta_\theta,
]

use the following safeguards.

### Zero-initialize the correction

Initialize both the residual output and the gate so that:

[
\widehat\delta_\theta\approx0,
\qquad
\alpha_\theta\approx0.
]

The initial model should therefore reproduce the compiled estimator.

### Stop gradients through the compiled estimate

Unless there is a strong reason to jointly optimize a differentiable nuisance layer, use:

```python
compiled_for_residual = compiled.detach()
final = compiled_for_residual + alpha * residual
```

This prevents the neural correction objective from degrading the statistically motivated anchor.

### Add a no-harm auxiliary loss

Because the true effect is available during synthetic pretraining, directly penalize corrections that make the estimate worse:

[
\mathcal L_{\mathrm{no\ harm}}
==============================

\frac{1}{B}
\sum_i
\max
\left[
0,;
\ell(\widehat\tau_i,\tau_i)
---------------------------

## \ell(\widehat\tau_{\mathrm{compiled},i},\tau_i)

\epsilon
\right].
]

Here, (\epsilon) is a small tolerance for sampling noise. This should be an auxiliary loss, not the only objective; otherwise, the model may simply keep the gate closed.

On real data, where (\tau_i) is unavailable, replace oracle error with a cross-fitted orthogonal pseudo-risk.

### Penalize unsupported corrections

Normalize the correction by the uncertainty of the compiled estimator:

[
r_i =
\frac{\alpha_i\delta_i}
{\widehat{\mathrm{se}}_{\mathrm{compiled},i}+\varepsilon}.
]

A soft penalty on large (r_i) discourages the network from making a three-standard-error correction without strong evidence. I would initially use a soft penalty rather than permanent hard clipping because some genuinely misspecified anchors require large corrections.

### Use a staged schedule

A safer training sequence is:

1. Train nuisance models, compiled estimators, and task encoders while the residual gate is fixed at zero.
2. Train estimator experts and the risk router while most of the backbone is frozen.
3. Train the residual and gate with the no-harm objective.
4. Jointly fine-tune at a lower learning rate.
5. Unfreeze the router only if its validation regret still improves.

This makes it easier to identify which component introduced a regression.

---

## 4. Ensure all nuisance and routing targets are out-of-fold

For ATE nuisance models, predictions used by the orthogonal score should be produced out-of-fold. For interference, hold out complete clusters or independent graph components rather than individual nodes. For regime detection, threshold selection and threshold evaluation should also use different folds where feasible.

Cross-fitting is especially important here because flexible nuisance learners can fit outcomes well while inducing biased causal estimates when evaluated on the same observations. Orthogonal scores and cross-fitting are explicitly designed to reduce the effect of nuisance regularization bias and overfitting on the causal target. ([arXiv][2])

The estimator router must also receive out-of-fold risk labels. Do not define the best expert using each expert’s in-sample error:

[
m_i^\star
\neq
\arg\min_m
\ell(\widehat\tau_{m,\mathrm{in\ sample}},\tau_i).
]

Instead use an independently evaluated or cross-fitted estimate:

[
m_i^\star
=========

\arg\min_m
\ell(\widehat\tau_{m,\mathrm{OOF}},\tau_i).
]

Otherwise, the router will systematically prefer the most flexible expert.

---

## 5. Guard against MoE router collapse and instability

For the estimator and representation MoEs, add:

* a short soft-routing warm-up before top-(k) routing;
* a low-weight load-balancing objective;
* a router-logit (z)-loss;
* expert dropout or occasional routing noise during training;
* a minimum amount of training data per expert;
* optional freezing or distillation of the router after routing stabilizes;
* risk-based supervision in addition to mechanism supervision.

Sparse MoEs are known to exhibit load imbalance, undertrained experts, and unstable routing. Switch Transformer and ST-MoE introduced load-balancing and router-stabilization techniques, including router (z)-loss; StableMoE specifically targets changing expert assignments during training. ([arXiv][3])

Do not force all experts to be exactly uniform indefinitely. Early balance is useful, but later specialization is desirable. The objective should prevent dead experts without suppressing meaningful specialization.

A practical schedule is:

```text
early training: strong balance, soft routing
middle training: weaker balance, top-k routing
late training: stable or frozen router, expert specialization
```

---

## 6. Do not regularize world particles by scalar disagreement alone

Remove the current incentive that simply maximizes the standard deviation of particle effect predictions. It can generate artificial disagreement without representing different causal worlds.

Instead:

* train with the exact mixture likelihood;
* supervise structural attributes when simulator truth is available;
* encourage diversity in graph, mechanism, exposure mapping, or identification assumptions;
* use particle dropout so no single particle is always required;
* compare against a one-particle model;
* increase the particle count only when held-out mixture NLL, calibration, or OOD error improves.

A useful guardrail is to require each additional particle configuration to outperform (K=1) and a small (K), such as (K=2), on held-out data. More particles should not be accepted merely because the training likelihood improves.

---

## 7. Guard the regime branch against both overfitting and the no-split shortcut

For a hierarchical regime decoder:

[
p(\text{split}),\qquad
p(j,t,c,\Delta\mid \text{split}),
]

apply different regularization to detection and localization.

For split detection, use matched null and stationary-nonlinear episodes and explicitly constrain the false-positive rate. For localization, train only on split episodes and balance variables and threshold ranges. Enforce minimum subgroup sizes for candidate thresholds and evaluate thresholds on held-out observations.

Do not select a checkpoint using overall structure accuracy, because a model that always predicts “no split” can achieve deceptively high accuracy when null episodes are common. Use:

* split-detection AUROC and AUPRC;
* true-positive rate at a fixed false-positive rate;
* null false-positive rate;
* localization accuracy conditional on a true split;
* threshold error conditional on the correct variable;
* separate results for weak and strong splits.

---

## 8. Use ordinary neural regularization only after the structural guardrails

The existing model already uses AdamW, weight decay, dropout, and gradient clipping. Those are useful, but increasing dropout alone will not solve simulator-prior overfitting or an in-sample estimator router.

After introducing the stronger safeguards, tune:

* dropout;
* stochastic depth between encoder blocks;
* weight decay;
* expert dropout;
* residual-gate regularization;
* early stopping;
* exponential moving average of model weights;
* reduced learning rates for pretrained or statistically initialized components.

Capacity should be increased only if both ID and OOD validation curves indicate underfitting.

---

# Improvements to training logging

## 9. Deficiencies in the current logger

In the present `cwfm/train.py`:

* validation runs only after training finishes;
* the logger records training values only every 50 steps;
* `point_loss` is calculated but not written to `parts`;
* `route_loss` is calculated but not written;
* `world_diversity` is calculated but not written;
* the gradient norm returned by `clip_grad_norm_` is discarded;
* validation reports means of batch-level losses, not raw-scale causal metrics;
* metrics are not broken down by task or scenario;
* no estimator is compared against the compiled anchor during training;
* there is no intermediate best-checkpoint selection;
* there are no OOD, worst-group, calibration, routing-collapse, or invariance metrics.

Consequently, the total training objective may decrease while the final causal estimate, router, or regime decoder becomes worse.

## 10. Log raw and weighted versions of every loss

For each objective, write both its unweighted value and its contribution to the total loss:

```text
loss/effect_nll_raw
loss/effect_nll_weighted
loss/effect_point_raw
loss/effect_point_weighted
loss/no_harm_raw
loss/no_harm_weighted
loss/router_risk_raw
loss/router_balance_raw
loss/router_z_raw
loss/world_mixture_nll_raw
loss/world_structure_raw
loss/regime_detection_raw
loss/regime_localization_raw
loss/graph_raw
loss/total
```

This reveals whether one task dominates simply because its numerical loss scale is larger.

Also log the denominator for every loss:

```text
count/estimable
count/identified
count/supported
count/regime
count/split
count/non_split
count/task_ate
count/task_interference
count/task_regime
```

This is important because a random batch can contain very few examples for a particular head. A low regime loss computed from one example is not comparable to a loss computed from ten examples.

---

## 11. Core optimization statistics

At every ordinary logging interval, record:

| Metric                               | Why it matters                                                             |
| ------------------------------------ | -------------------------------------------------------------------------- |
| Learning rate                        | Associates instability or plateaus with the schedule                       |
| Gradient norm before clipping        | Detects explosions and silent dependence on clipping                       |
| Gradient norm after clipping         | Verifies that clipping is behaving as intended                             |
| Fraction of steps clipped            | Persistent clipping can indicate an unsuitable learning rate or loss scale |
| Parameter norm                       | Detects uncontrolled weight growth                                         |
| Update-to-parameter norm ratio       | Shows whether optimizer steps are too aggressive or negligible             |
| Router-gradient norm                 | Detects a frozen or excessively dominant router                            |
| Residual-head gradient norm          | Detects whether the correction head is actually learning                   |
| Non-finite activation/gradient count | Immediate numerical-failure indicator                                      |
| Throughput and memory                | Helps identify slow components and accidental computational growth         |
| Target mean, SD, and quantiles       | Detects changes in training mixture or generator behavior                  |

`torch.nn.utils.clip_grad_norm_` returns the pre-clipping total norm, so it can be logged directly.

---

## 12. Log causal performance in raw units

The primary validation metrics should be computed after reversing outcome normalization.

For every task, scenario, template family, and relevant parameter bucket, record:

```text
effect/mae
effect/rmse
effect/signed_bias
effect/median_absolute_error
effect/q90_absolute_error
effect/q95_absolute_error
effect/correlation
```

The mean alone is insufficient. A foundation model should also report:

[
\text{worst-group MAE}
======================

\max_g
\operatorname{MAE}_g,
]

where groups can represent mechanism family, overlap, sample size, noise level, graph type, and signal strength.

A useful dashboard should always show four columns:

```text
train
validation-ID
validation-OOD
validation-OOD worst group
```

The two most informative generalization gaps are:

[
G_{\mathrm{sample}}
===================

## \mathrm{MAE}_{\mathrm{val-ID}}

\mathrm{MAE}_{\mathrm{train}},
]

and

[
G_{\mathrm{prior}}
==================

## \mathrm{MAE}_{\mathrm{val-OOD}}

\mathrm{MAE}_{\mathrm{val-ID}}.
]

The first suggests ordinary fitting problems; the second indicates simulator-prior dependence.

---

## 13. Make “final versus compiled” the central residual diagnostic

For each episode, retain:

[
e_{\mathrm{compiled}}
=====================

|\widehat\tau_{\mathrm{compiled}}-\tau|,
\qquad
e_{\mathrm{final}}
==================

|\widehat\tau_{\mathrm{final}}-\tau|.
]

Then log:

[
\text{mean gain}
================

\mathbb E[e_{\mathrm{compiled}}-e_{\mathrm{final}}],
]

[
\text{win rate}
===============

P(e_{\mathrm{final}}<e_{\mathrm{compiled}}),
]

[
\text{harm rate}
================

P(e_{\mathrm{final}}>e_{\mathrm{compiled}}),
]

and a prespecified catastrophic-harm rate:

[
P(e_{\mathrm{final}}-e_{\mathrm{compiled}}>h),
]

where (h) is fixed before inspecting the results.

Also log:

```text
residual/correction_mean
residual/correction_abs_mean
residual/correction_q90
residual/correction_over_anchor_se_q50
residual/correction_over_anchor_se_q90
gate/alpha_mean
gate/alpha_std
gate/alpha_p10
gate/alpha_p50
gate/alpha_p90
gate/alpha_when_residual_helps
gate/alpha_when_residual_harms
```

On synthetic validation data, an especially revealing metric is:

[
\operatorname{corr}
\left(
\alpha_i,;
e_{\mathrm{compiled},i}
-----------------------

e_{\mathrm{direct},i}
\right).
]

The gate should open more when the alternative is genuinely better. If this correlation is near zero or negative, the gate is not learning useful selection.

---

## 14. Log estimator-router quality directly

For every candidate estimator (m), retain its validation error. Then compute the router’s regret:

[
R_i
===

## \ell(\widehat\tau_{\mathrm{mixture},i},\tau_i)

\min_m
\ell(\widehat\tau_{m,i},\tau_i).
]

Log:

```text
router/mean_regret
router/median_regret
router/q90_regret
router/oracle_expert_top1_accuracy
router/oracle_expert_top2_accuracy
router/soft_target_cross_entropy
router/mixture_error
router/hard_selected_error
router/best_single_expert_error
router/oracle_error
```

Also create an expert-by-scenario matrix containing:

* average router probability;
* selection frequency;
* expert MAE;
* mixture contribution.

This will show whether an expert is genuinely specialized or merely receives traffic.

---

## 15. Log MoE utilization and routing dynamics

For each MoE layer and expert, record:

```text
moe/layer_k/expert_j/load_fraction
moe/layer_k/expert_j/probability_mass
moe/layer_k/expert_j/gradient_norm
moe/layer_k/expert_j/validation_mae
```

At the layer level, record:

[
N_{\mathrm{eff}}
================

\frac{1}{\sum_j p_j^2},
]

together with:

```text
moe/effective_experts
moe/routing_entropy
moe/load_coefficient_of_variation
moe/load_gini
moe/dead_expert_count
moe/top1_top2_margin
moe/overflow_or_drop_fraction
moe/load_balance_loss
moe/router_z_loss
```

Use a fixed probe bank to measure routing fluctuation:

[
\text{flip rate}
================

P\left[
\arg\max_j p_j^{(t)}
\neq
\arg\max_j p_j^{(t-\Delta)}
\right].
]

A high flip rate late in training indicates unstable specialization. A zero flip rate very early may indicate a collapsed or saturated router.

---

## 16. Log nuisance-model quality out-of-fold

For static ATE:

```text
nuisance/mu0_oof_rmse
nuisance/mu1_oof_rmse
nuisance/propensity_log_loss
nuisance/propensity_brier
nuisance/propensity_auc
nuisance/propensity_clip_fraction
nuisance/propensity_p01
nuisance/propensity_p50
nuisance/propensity_p99
nuisance/orthogonal_score_mean
nuisance/orthogonal_score_sd
nuisance/effect_fold_sd
```

For interference:

```text
nuisance/network_response_oof_rmse
nuisance/network_response_rmse_by_exposure_bin
nuisance/cluster_fold_effect_sd
nuisance/exposure_support_p05
nuisance/exposure_support_p95
nuisance/low_support_fraction
```

Nuisance predictive performance and causal performance should both be logged. Improving outcome RMSE while worsening orthogonal effect error is a warning that the nuisance objective is not sufficiently aligned with the estimand.

---

## 17. Log whether world particles are useful

For every validation split, record:

```text
world/mixture_nll
world/effect_mae
world/entropy_mean
world/entropy_std_across_episodes
world/effective_particle_count
world/max_weight_mean
world/max_weight_q90
world/particle_effect_spread
world/active_particle_count
world/pairwise_structure_distance
world/responsibility_accuracy
```

The effective particle count is:

[
K_{\mathrm{eff}}
================

\frac{1}{\sum_s w_s^2}.
]

Two different failures should trigger separate warnings:

1. **Uniform non-use:** (K_{\mathrm{eff}}\approx K) for almost every episode, with almost no variation across very different SCMs.
2. **Posterior collapse:** (K_{\mathrm{eff}}\approx1) for almost every episode, and the same particle always dominates.

Uniform weights are not automatically incorrect, but persistent entropy near (\log K) across heterogeneous, structurally labelled episodes strongly suggests that the posterior is not using the evidence.

---

## 18. Log uncertainty separately from point accuracy

For the direct mixture or final calibrated output, record:

```text
uncertainty/nll
uncertainty/crps
uncertainty/coverage_50
uncertainty/coverage_80
uncertainty/coverage_90
uncertainty/coverage_95
uncertainty/mean_width_50
uncertainty/mean_width_80
uncertainty/mean_width_90
uncertainty/mean_width_95
uncertainty/standardized_residual_mean
uncertainty/standardized_residual_sd
```

Coverage and width must be reported jointly. A model can trivially improve coverage by producing uselessly broad intervals.

Calculate coverage by:

* task;
* mechanism;
* support bucket;
* signal strength;
* sample size;
* selected estimator;
* gate-activation bucket;
* ID versus OOD.

Do not use the calibration bank to choose the model weights or checkpoint. It should only fit the final interval and abstention calibration after checkpoint selection.

---

## 19. Improve regime logging

Separate the regime metrics into three stages.

### Detection

```text
regime/split_auroc
regime/split_auprc
regime/split_brier
regime/split_ece
regime/null_false_positive_rate
regime/strong_split_recall
regime/weak_split_recall
regime/tpr_at_fixed_fpr
```

### Localization

```text
regime/variable_accuracy_given_split
regime/variable_top2_accuracy_given_split
regime/threshold_mae_given_correct_variable
regime/threshold_quantile_mae
regime/change_type_macro_f1
```

### Effect magnitude

```text
regime/delta_mae
regime/delta_signed_bias
regime/delta_coverage
```

Always log the predicted no-split rate alongside the true no-split prevalence. A growing no-split rate accompanied by falling training loss is a direct sign of class-shortcut learning.

---

## 20. Add invariance and robustness probes to every validation cycle

A small fixed probe bank can detect shortcut learning long before aggregate MAE clearly deteriorates.

Recommended probes are:

| Probe                                | Logged statistic                                                        |
| ------------------------------------ | ----------------------------------------------------------------------- |
| Row permutation                      | Absolute change in effect and gate                                      |
| Role-preserving variable permutation | Equivariance error after remapping output                               |
| Affine rescaling                     | Effect change after transforming and correctly untransforming variables |
| Irrelevant-variable insertion        | Change in effect and estimator weights                                  |
| Independent resample from same SCM   | Between-sample estimate variance                                        |
| Bootstrap or cluster bootstrap       | Prediction stability                                                    |
| Query perturbation                   | Sensitivity to intervention values and target estimand                  |
| Metadata removal                     | Difference between full and data-only prediction                        |
| Outcome permutation negative control | False causal signal                                                     |
| Null treatment or null spillover     | Type-I error and gate-activation rate                                   |

Suggested names include:

```text
robustness/row_permutation_delta
robustness/variable_equivariance_error
robustness/affine_transform_delta
robustness/irrelevant_variable_delta
robustness/same_scm_resample_sd
robustness/null_effect_false_positive_rate
robustness/metadata_ablation_delta
```

---

# Checkpoint and early-stopping policy

## 21. Do not select a checkpoint using total validation loss alone

Use a two-stage policy.

### Stage 1: eligibility constraints

A checkpoint is eligible only when it satisfies prespecified safeguards such as:

* no non-finite outputs or gradients;
* final estimator is not materially worse than the compiled estimator on any core task;
* OOD worst-group error does not exceed a fixed tolerance;
* null-regime false-positive rate remains below its target;
* uncertainty coverage remains within a target band;
* no expert is persistently dead;
* the false-answer rate on nonidentified or unsupported episodes remains below its threshold.

The exact tolerances should be declared before the confirmatory runs.

### Stage 2: rank eligible checkpoints

Among eligible checkpoints, choose using a composite such as:

[
S =
\operatorname{MAE}*{\mathrm{ID}}
+
\lambda_1\operatorname{MAE}*{\mathrm{OOD,worst}}
+
\lambda_2\operatorname{router\ regret}
+
\lambda_3\operatorname{calibration\ error}.
]

Alternatively, use a lexicographic policy:

1. minimize OOD worst-group MAE;
2. among statistically indistinguishable checkpoints, minimize ID MAE;
3. among those, minimize interval width and router regret.

The lexicographic approach is easier to interpret than finely tuned arbitrary weights.

Because the same fixed validation episodes are evaluated at each checkpoint, use paired episode-level differences when comparing checkpoints. Save a new best checkpoint only when the estimated improvement is larger than the validation noise or when it materially improves a guardrail without degrading the primary metric.

Early stopping should occur after several validation rounds without an eligible improvement, especially when training loss continues to decrease while ID or OOD metrics worsen.

For the current approximately 1,600-step experiment, validation every 50–100 steps would already be much more informative than one terminal validation. For a larger run, evaluate a small bank every roughly 1–2% of training and a larger OOD bank less frequently.

---

# Concrete changes to `train.py`

## 22. Expand `compute_loss()`

The returned `parts` dictionary should include every objective:

```python
parts = {
    "loss/total": loss.detach(),
    "loss/effect_nll": effect_loss.detach(),
    "loss/effect_point": point_loss.detach(),
    "loss/no_harm": no_harm_loss.detach(),
    "loss/identification": id_loss.detach(),
    "loss/support": support_loss.detach(),
    "loss/mechanism": mechanism_loss.detach(),
    "loss/router_risk": route_risk_loss.detach(),
    "loss/router_balance": route_balance_loss.detach(),
    "loss/router_z": router_z_loss.detach(),
    "loss/world_mixture_nll": mixture_nll.detach(),
    "loss/world_structure": world_structure_loss.detach(),
    "loss/regime_detection": split_loss.detach(),
    "loss/regime_localization": localization_loss.detach(),
    "loss/graph": graph_loss.detach(),
}
```

Return counts separately and protect every conditional mean against an empty mask. For example:

```python
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum()
    if count == 0:
        return x.new_zeros(())
    return x[mask].sum() / count
```

## 23. Capture optimization statistics

```python
loss.backward()

grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

parts["optim/grad_norm_preclip"] = float(grad_norm.detach())
parts["optim/grad_clipped"] = float(grad_norm > 1.0)
parts["optim/learning_rate"] = optimizer.param_groups[0]["lr"]

optimizer.step()
```

Periodically compute parameter and update norms. Doing this every step is unnecessary.

## 24. Replace terminal-only validation with periodic grouped evaluation

The validation function should return per-episode records, not only averaged losses:

```python
{
    "template_id": ...,
    "task": ...,
    "scenario": ...,
    "target_raw": ...,
    "compiled": ...,
    "final": ...,
    "expert_estimates": ...,
    "gate": ...,
    "world_weights": ...,
    "interval": ...,
    "split_probability": ...,
}
```

Aggregate these records afterward by task, scenario, template, sample-size bucket, overlap, signal, and OOD category.

Do not average pre-averaged batch metrics. Accumulate per-example numerators and denominators, because batches can contain different numbers of estimable or regime episodes.

## 25. Use append-only JSONL for metrics

A single growing `history.json` becomes awkward and can be corrupted if training is interrupted. Use:

```text
artifacts/run_id/train_metrics.jsonl
artifacts/run_id/validation_metrics.jsonl
artifacts/run_id/checkpoint_manifest.json
artifacts/run_id/episode_diagnostics.parquet
```

An individual validation record could look like:

```json
{
  "step": 800,
  "split": "validation_ood",
  "task": "network_interference",
  "scenario": "heldout_small_world_threshold",
  "metric": "residual/mean_gain",
  "value": 0.083,
  "count": 128
}
```

The manifest should contain:

* complete configuration;
* all loss weights;
* training and validation seed manifests;
* generator version and hashes;
* source commit or file hashes;
* expert definitions;
* checkpoint-selection rule;
* best checkpoint step and reason;
* package versions;
* number of episodes seen by each task and expert.

---

# Minimum high-value logging set

For the first revised implementation, I would prioritize the following before adding lower-level attention diagnostics:

1. Periodic fixed ID and OOD validation.
2. Raw-scale MAE, bias, q90 error, and worst-group MAE per task.
3. Compiled-versus-final gain, win rate, harm rate, and correction size.
4. Estimator-router regret and expert-by-scenario weights.
5. Expert utilization, dead-expert count, entropy, and routing flip rate.
6. Mixture NLL, particle effective count, weight entropy, coverage, and width.
7. Regime split FPR, strong/weak recall, conditional localization, and threshold error.
8. Gradient norm, clipping fraction, parameter/update norm, and all raw loss terms.
9. Invariance probes for row order, variable order, scaling, and irrelevant variables.
10. A constraint-based checkpoint rule that can automatically reject a lower-training-loss but less reliable model.

The most important single dashboard would plot, over training steps:

```text
train final MAE
ID validation final MAE
OOD validation final MAE
compiled-estimator MAE
final-minus-compiled gain
residual harm rate
router regret
worst-group MAE
world effective particle count
regime null FPR and split recall
```

That plot would immediately reveal whether the revised architecture is genuinely learning useful causal corrections or merely fitting the training simulator more aggressively.

[1]: https://proceedings.mlr.press/v139/koh21a/koh21a.pdf?utm_source=chatgpt.com "A Benchmark of in-the-Wild Distribution Shifts"
[2]: https://arxiv.org/abs/1608.00060?utm_source=chatgpt.com "Double/Debiased Machine Learning for Treatment and Causal Parameters"
[3]: https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
