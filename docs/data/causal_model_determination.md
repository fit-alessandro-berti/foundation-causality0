# Data generation and testing for causal-model determination

Corresponding method: [`../methods/causal_model_determination.md`](../methods/causal_model_determination.md)

## Detection target

The primary method is model-based recursive partitioning (MOB). The benchmark should therefore test whether the procedure can recover observed **regime variables** whose values change the parameters of the conditional outcome model.

For row \(i\), let

\[
X_i=(X_{i1},\ldots,X_{iN}), \qquad Y_i\in\mathbb{R}^{M}.
\]

A synthetic data set should contain a known regime function

\[
r_i=g(X_{i,R})\in\{1,\ldots,K\},
\]

where \(R\subseteq\{1,\ldots,N\}\) is the set of true model-determining features, and a regime-specific model

\[
Y_i=a_{r_i}+B_{r_i}^{\mathsf T}h(X_{i,P})+\varepsilon_i.
\]

Here:

- \(P\) is the set of ordinary predictors used inside each local model;
- \(R\) is the set of variables that determine which local model applies;
- \(h(\cdot)\) is the feature map fitted by the local model;
- \(a_k\) and \(B_k\) are known regime-specific parameters;
- \(\varepsilon_i\) is multivariate noise.

The benchmark must keep \(P\) and \(R\) conceptually separate. A variable can be highly predictive because it belongs to \(P\) without being a regime variable. Conversely, a regime variable can have almost no marginal correlation with an outcome while still changing the slopes of other predictors.

## Recommended output contract for the generator

Each generated instance should return:

```text
X                    S x N feature matrix
Y                    S x M outcome matrix
feature_names        names of all columns in X
true_regime_features set R
true_tree             split variables, operators, thresholds/categories
true_regime_id        length-S leaf/regime label
true_parameters       {regime: intercept and coefficient matrix}
ordinary_predictors   set P
proxy_groups          optional map from true regime variables to correlated proxies
scenario              generator configuration
seed                  random seed
```

Keeping the truth separate from the observed table prevents accidental use of the answer by the fitting code.

## Default synthetic generator

### 1. Generate observed features

A useful default is a correlated Gaussian design:

\[
X_i\sim\mathcal N(0,\Sigma_X),
\qquad
(\Sigma_X)_{jk}=\rho^{|j-k|},
\]

with \(\rho\in\{0,0.3,0.7\}\). Standardize continuous columns after generation.

Use distinct feature roles, for example:

```text
X1 ... Xp             ordinary local-model predictors P
Xq                    true continuous regime variable
Xc                    optional categorical regime variable
Xproxy                 correlated proxy for Xq
remaining columns      irrelevant distractors
```

To create a proxy without making it identical to the truth, use

\[
X_{\text{proxy}}
=\rho_p X_q+\sqrt{1-\rho_p^2}\,U,
\qquad U\sim\mathcal N(0,1).
\]

Record proxy relationships in `proxy_groups`. Exact recovery and proxy-family recovery should be reported separately.

### 2. Define the true regime tree

Start with a one-split stump because this directly matches the simplest recommended implementation:

\[
r_i=
\begin{cases}
1,&X_{iq}\le c,\\
2,&X_{iq}>c.
\end{cases}
\]

Choose \(c\) from a quantile of \(X_q\), rather than a fixed raw value, so that the expected regime sizes are controlled. Useful quantiles are \(0.50\), \(0.30\), and \(0.15\) for balanced, moderately imbalanced, and highly imbalanced tests.

For deeper-tree tests, use a known hierarchy such as

```text
Xq <= c1
|-- Xc == A       -> regime 1
|-- otherwise     -> regime 2
Xq > c1           -> regime 3
```

Do not start the benchmark with a deep random tree. A controlled sequence from zero splits, to one split, to two or three splits makes failures interpretable.

### 3. Generate regime-specific outcome mechanisms

For a linear default,

\[
Y_i=a_{r_i}+B_{r_i}^{\mathsf T}X_{i,P}+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\Sigma_\varepsilon).
\]

For \(M>1\), let \(B_k\in\mathbb R^{|P|\times M}\), and create correlated residuals with

\[
(\Sigma_\varepsilon)_{mm'}
=\sigma_m\sigma_{m'}\rho_Y^{|m-m'|}.
\]

A convenient way to control detection difficulty is

\[
B_2=B_1+\delta D,
\qquad \|D\|_F=1,
\]

where \(\delta\) is the mechanism-change magnitude. Vary \(\delta\) over a grid such as \(0,0.25,0.5,1,2\), interpreted relative to standardized variables.

For the cleanest regime-variable test, exclude \(X_q\) itself from \(X_P\), set equal intercepts across regimes, and change only slopes in \(B_k\). Then the regime variable need not be a strong marginal predictor of \(Y\).

### 4. Create independent discovery and evaluation samples

Generate at least two independent samples from the same mechanism:

- a discovery sample used to choose features, thresholds, tree depth, and significance cutoffs;
- an evaluation sample used only to measure regime assignment, local-parameter recovery, and predictive performance.

For Monte Carlo experiments, repeat this pair over many seeds. Hyperparameters should be selected using discovery samples from a separate set of seeds, not by maximizing the reported test metrics.

## Required benchmark scenarios

The following scenarios distinguish genuine mechanism detection from ordinary prediction or misspecification.

| Scenario | Data-generating truth | What a good method should do |
| --- | --- | --- |
| Null | \(B_1=\cdots=B_K\); no regime split | Return no split at the nominal false-positive rate |
| Clean one split | One true regime feature and one threshold | Select the feature and localize the threshold |
| Weak change | Small \(\delta\) | Show decreasing power without uncontrolled false positives |
| Correlated proxy | A distractor is highly correlated with the true regime feature | Prefer the true feature; report proxy-family success separately |
| Unbalanced regimes | One leaf contains 10–20% of rows | Avoid tiny unsupported leaves and quantify loss of power |
| Multiple outcomes | Only some outcomes change mechanism | Detect the joint change without domination by outcome scale |
| Multiple splits | A shallow known tree | Recover relevant variables and approximately recover the partition |
| Global nonlinearity, no regime | One smooth nonlinear model such as \(Y=X_1^2+\varepsilon\) | Produce no split when the local model includes the nonlinearity |
| Misspecified local model | Same nonlinear truth but only a linear local model | Reveal the expected false-regime failure mode |
| Main-effect distractor | A feature strongly predicts \(Y\) but does not change coefficients | Not select it merely because it predicts the outcome |
| Variance-only change | Conditional mean is stable but residual variance changes | Classify the change correctly or state that the chosen cost detects distributional, not only coefficient, changes |

The global-nonlinearity pair is particularly important: one run uses a correctly specified local basis \(h(X)=[X,X^2]\), and the other deliberately omits \(X^2\). The difference measures sensitivity to model misspecification.

## Testing pipeline

### 1. Fix the estimand and local model

Before fitting, declare what constitutes a model change:

- changes in regression coefficients;
- changes in a known treatment effect;
- changes in a larger structural parameter vector;
- or any improvement in conditional likelihood, including variance changes.

Use the same estimand in the generator, detector, and evaluation code.

### 2. Fit preprocessing only on discovery data

Imputation, encoding, basis expansion, and scaling must be fitted on the discovery sample and applied unchanged to the evaluation sample. Candidate regime variables must not include identifiers, post-outcome attributes, or columns derived from \(Y\).

### 3. Fit a stump before a deeper tree

For every candidate feature and admissible split:

1. fit the pooled local model;
2. fit the two local models induced by the split;
3. calculate the in-sample instability statistic and, preferably, cross-fitted or held-out loss reduction;
4. select the best feature/threshold pair;
5. run a permutation or parametric-bootstrap test that repeats the **entire split search**.

Only continue recursively when the current split passes the predeclared significance and minimum-leaf-size criteria.

### 4. Estimate uncertainty

Bootstrap the complete procedure, including preprocessing, feature search, threshold search, and pruning. For each candidate feature, retain:

- selection frequency at the root;
- selection frequency anywhere in the tree;
- distribution of selected thresholds;
- distribution of the number of leaves.

A conditional bootstrap that refits only the local coefficients while holding the selected tree fixed is not sufficient for selection uncertainty.

### 5. Evaluate on untouched data

Apply the learned tree to the independent evaluation sample. Refit local models on a discovery-only estimation subset, or use honest sample splitting so that the rows used to choose the split are not also the sole rows used to estimate local parameters.

Compare at least:

- the learned partitioned model;
- one pooled model with the same local-model class;
- an ordinary predictive tree or forest as a diagnostic baseline;
- the oracle partition using the true regime labels.

The ordinary predictive model is not expected to recover regime variables, but it shows whether a method is confusing strong main effects with parameter instability.

## Matching estimated and true structures

### One-split case

A split-feature hit requires the exact true feature. Also report a relaxed proxy-family hit when the selected feature belongs to a declared correlated proxy group.

Normalize continuous threshold error as

\[
e_c=\frac{|\widehat c-c|}{\operatorname{IQR}(X_q)}.
\]

Threshold error is meaningful only when the correct split feature is selected.

### Multiple-leaf case

Leaf labels are arbitrary. Match estimated leaves to true regimes using maximum-overlap assignment, then compute partition and parameter metrics. Partition metrics such as adjusted Rand index do not require explicit label matching.

For comparing estimated split-variable sets with \(R\), use set-based precision, recall, and F1. Count each feature once even when it appears at several nodes, and separately report node-level recovery when repeated use matters.

## Metrics to print

### Detection and selection

```text
split_detected                 0/1
n_true_splits                  integer
n_detected_splits              integer
selected_root_feature          name or NONE
root_feature_exact_hit         0/1
root_feature_proxy_family_hit  0/1
true_feature_rank              rank by instability statistic
permutation_or_bootstrap_p     calibrated p-value
```

Across repeated data sets, print:

- null false-positive rate;
- alternative detection power;
- feature-set precision, recall, F1, and false discovery rate;
- root-feature accuracy;
- mean absolute split-count error.

### Split localization and partition quality

- normalized threshold absolute error;
- threshold confidence-interval coverage;
- adjusted Rand index (ARI) between true and estimated regime labels;
- normalized mutual information (NMI);
- leaf purity and minimum leaf size;
- proportion of evaluation rows assigned to the correct regime after optimal label matching.

### Local-model recovery

After matching regimes, report

\[
\operatorname{relative\ parameter\ error}
=
\frac{\sum_k\|\widehat B_k-B_k\|_F}
     {\sum_k\|B_k\|_F+\epsilon},
\]

as well as intercept error and the cosine similarity of each vectorized coefficient matrix. If the fitted local model is intentionally different from the generator, omit parameter recovery and rely on predictive loss.

### Generalization

For every outcome, print test RMSE or log-loss, then a standardized joint score. Also print

\[
\Delta L_{\text{pooled}}
=L_{\text{pooled}}-L_{\text{partitioned}}
\]

and the fraction of the oracle improvement recovered:

\[
\frac{L_{\text{pooled}}-L_{\text{partitioned}}}
     {L_{\text{pooled}}-L_{\text{oracle}}}.
\]

Clip or mark this ratio when the oracle denominator is approximately zero.

### Stability

- bootstrap selection frequency for every feature;
- Jaccard similarity of selected feature sets across resamples;
- median and interval of the selected threshold;
- probability of recovering the correct number of leaves.

Report means, standard deviations, medians, and 5th/95th percentiles over simulation seeds. A single successful synthetic example is not evidence of reliable detection.

## Minimal benchmark pseudocode

```python
for scenario in scenarios:
    tuning_runs, evaluation_runs = make_independent_seed_sets()

    detector_cfg = tune_detector_on_simulations(
        tuning_runs,
        objective="high regime-feature F1 subject to controlled null FPR",
    )

    records = []
    for seed in evaluation_runs:
        discovery, evaluation, truth = generate_pair(scenario, seed)

        fitted = fit_mob_pipeline(discovery.X, discovery.Y, detector_cfg)
        bootstrap = bootstrap_complete_pipeline(discovery, detector_cfg)

        pred = fitted.predict(evaluation.X)
        estimated_regime = fitted.apply(evaluation.X)

        records.append(
            evaluate(
                fitted=fitted,
                bootstrap=bootstrap,
                prediction=pred,
                estimated_regime=estimated_regime,
                evaluation=evaluation,
                truth=truth,
            )
        )

    print_aggregate_metrics(records)
```

The tuning objective should not use only predictive RMSE. A useful objective is to maximize regime-feature F1 or split-detection power while constraining the null false-positive rate to the chosen level.

## Optional treatment-effect variant

When the corresponding method is implemented as a causal tree for a known treatment \(T\), generate

\[
Y_i=\mu(X_i)+T_i\tau(r_i)+\varepsilon_i,
\]

with randomized treatment for the clean benchmark and confounded treatment plus correctly measured confounders for a harder benchmark. In addition to the partition metrics above, print:

- CATE RMSE against the true \(\tau(r)\);
- treatment-effect sign accuracy;
- coverage of leaf-specific effect intervals;
- policy value or regret on an independent test sample;
- false subgroup-discovery rate under constant \(\tau\).

Keep this treatment-effect benchmark separate from the primary coefficient-instability benchmark because the estimands differ.

## Suggested compact report

```text
scenario=one_split_balanced  seeds=200  S=1000  N=20  M=3
null_FPR=0.047  detection_power=0.910
root_feature_accuracy=0.865  proxy_family_accuracy=0.930
feature_precision=0.882  feature_recall=0.865  feature_F1=0.873
threshold_MAE_IQR=0.083  threshold_CI_coverage=0.941
regime_ARI=0.824  split_count_MAE=0.150
relative_parameter_error=0.117
joint_test_loss_pooled=1.000  joint_test_loss_partitioned=0.781
oracle_improvement_recovered=0.86
root_bootstrap_frequency_true_feature=0.89
```

The report should always place null calibration beside power. A detector that finds the true split frequently but also splits most null data sets is not a high-quality detector.
