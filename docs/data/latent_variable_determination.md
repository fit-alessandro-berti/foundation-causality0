# Data generation and testing for latent-variable determination

Corresponding method: [`../methods/latent_variable_determination.md`](../methods/latent_variable_determination.md)

## Detection target

The recommended method, sparse multi-response PLS (sparse PLS2), is intended to recover **outcome-relevant sparse feature groups and component scores**. It is not a test of causal influence.

Let a synthetic data set contain \(S\) rows, \(N\) observed features, \(M\) outcomes, and \(K_{\text{true}}\) latent variables:

\[
Z\in\mathbb R^{S\times K_{\text{true}}},
\qquad
X=Z\Lambda^{\mathsf T}+E_X,
\qquad
Y=Z\Gamma^{\mathsf T}+E_Y.
\]

The benchmark truth is defined by:

- the loading support of \(\Lambda\), which specifies the observed-feature groups;
- the outcome coefficient support of \(\Gamma\), which specifies which latent variables are outcome-relevant;
- the true latent scores \(Z\), known only to the evaluator.

The primary recovery target should be the groups associated with latent variables that have at least one nonzero column in \(\Gamma\). Correlated feature blocks that do not predict any outcome are useful negative controls.

## Recommended output contract for the generator

```text
X                       S x N observed-feature matrix
Y                       S x M outcome matrix
Z_true                  S x K_true latent-score matrix
Lambda_true             N x K_true feature loading matrix
Gamma_true              M x K_true outcome loading/coefficient matrix
true_group[j]           latent group for feature j, or NOISE
relevant_latents        latent variables with nonzero outcome association
relevant_feature_set    union of their loading supports
cross_loading_features  optional set
feature_names
outcome_names
scenario
seed
```

Use a distinct value such as `NOISE` for unassigned distractor features. Do not force them into a true group.

## Default synthetic generator

### 1. Choose the latent structure

Select:

- \(K_{\text{true}}\), for example 2, 4, or 8;
- group sizes \(d_k\), either equal or deliberately unequal;
- number of pure noise features \(N_0\);
- number of outcome-relevant latent variables \(K_Y\le K_{\text{true}}\).

Sample the latent rows as

\[
Z_i\sim\mathcal N(0,\Sigma_Z).
\]

For a simple case, use \(\Sigma_Z=I\). To test component separation, use an equicorrelated or sparse covariance matrix with pairwise correlations such as 0.3, 0.6, and 0.85. Ensure that \(\Sigma_Z\) is positive definite.

### 2. Construct sparse feature loadings

For every feature \(j\) assigned to group \(k\), set

\[
\Lambda_{jk}=s_j\lambda_j,
\qquad
s_j\in\{-1,+1\},
\]

with \(\lambda_j\) sampled, for example, from \([0.6,1.0]\). Set all other loadings in that row to zero for the clean simple-structure benchmark.

Generate observed features as

\[
X_{ij}=\Lambda_{j,g(j)}Z_{ik}+\sigma_{X,j}U_{ij},
\qquad U_{ij}\sim\mathcal N(0,1).
\]

Control measurement quality through a target reliability or signal-to-noise ratio. Add \(N_0\) noise features independently of \(Z\), or as correlated nuisance blocks that have no path to \(Y\).

For harder scenarios, allow a known fraction of features to cross-load:

\[
X_j=\lambda_{j1}Z_{g_1(j)}+\lambda_{j2}Z_{g_2(j)}+e_j.
\]

Record both supports. Since a strict hard partition is then not the complete truth, evaluate both support recovery and a declared primary-group assignment.

### 3. Generate multiple outcomes

Use

\[
Y=Z\Gamma^{\mathsf T}+E_Y,
\qquad
E_{Y,i}\sim\mathcal N(0,\Sigma_Y).
\]

Give every main-benchmark latent variable at least one nonzero outcome coefficient, while varying the coefficient magnitude to control relevance. Include separate nuisance-latent scenarios in which one or more latent factors strongly structure \(X\) but have \(\Gamma_{mk}=0\) for every outcome.

For multiple outcomes, vary:

- whether each latent affects one or several outcomes;
- residual correlation between outcomes;
- raw outcome scale before preprocessing;
- opposing coefficient signs across outcomes.

A scale-stress scenario should multiply one outcome by a large constant. Correct scaling inside the pipeline should prevent that outcome from determining all components.

### 4. Split before preprocessing

Generate independent training and test rows from the same \(\Lambda\), \(\Gamma\), and \(\Sigma_Z\). Fit imputation and standardization on training rows only. The true \(Z\) for test rows is retained only for evaluation.

For hyperparameter benchmarking, use three independent levels:

1. training folds for fitting candidate models;
2. validation folds for selecting component count and sparsity;
3. untouched test rows for final recovery and prediction metrics.

Nested cross-validation is required when the same observed data set is used for both tuning and performance estimation.

## Required benchmark scenarios

| Scenario | Generator change | Expected behavior |
| --- | --- | --- |
| Clean balanced groups | Independent latents, equal group sizes, strong loadings | Recover \(K\), supports, and scores accurately |
| Weak measurement | Larger \(E_X\) | Graceful decline in group and score recovery |
| Weak outcome relevance | Smaller entries of \(\Gamma\) | Lower selection power for those groups |
| Outcome-irrelevant latent block | A factor structures \(X\) but has \(\Gamma_{\cdot k}=0\) | Sparse PLS may leave it unselected; this is correct for an outcome-relevant target |
| Correlated latents | Large off-diagonal entries of \(\Sigma_Z\) | Avoid merging all groups or unstable swapping |
| Unequal group sizes | Small and large loading supports | Quantify bias toward large groups |
| Correlated nuisance features | Noise block correlated internally but not with \(Y\) | Avoid selecting it merely because it is a strong factor of \(X\) |
| Cross-loadings | Some features load on two factors | Recover supports or mark assignment uncertainty |
| High-dimensional | \(N>S\) | Maintain controlled false selections through sparsity |
| Heterogeneous outcome scales | Outcomes differ greatly in raw variance | Produce similar results after correct scaling |
| Missing data | MCAR and covariate-dependent missingness | Avoid preprocessing leakage; report degradation |
| Complete null | \(Y\perp Z\) and \(Y\perp X\) | Select no stable outcome-relevant groups at the chosen error rate |

Run each scenario over grids of \(S\), \(N\), \(M\), loading strength, outcome strength, and latent correlation. Report results by scenario rather than only one pooled average.

## Testing pipeline

### 1. Nested model selection

Within each outer training split:

1. impute and scale \(X\) and \(Y\) using inner-training rows;
2. fit sparse PLS2 over candidate component counts \(K\) and sparsity levels;
3. choose hyperparameters using a predeclared multivariate validation loss;
4. refit on the complete outer-training data;
5. evaluate on the outer test data.

A useful standardized validation loss is

\[
L=\frac{1}{M}\sum_{m=1}^{M}
\frac{\sum_i(Y_{im}-\widehat Y_{im})^2}
     {\sum_i(Y_{im}-\overline Y_m)^2}.
\]

Use the one-standard-error rule when a sparser model has validation performance indistinguishable from the minimum-loss model.

### 2. Convert sparse loadings into reported groups

For estimated loading matrix \(\widehat W\), assign feature \(j\) only when

\[
\max_k|\widehat w_{jk}|>\tau_w.
\]

Then use

\[
\widehat g(j)=\arg\max_k|\widehat w_{jk}|.
\]

Features below \(\tau_w\) remain unassigned. Tune \(\tau_w\) inside the training procedure, or predeclare it. Never choose it using true test labels.

Retain the original soft loading matrix for score construction and prediction. Hard assignments are for interpretation and group-recovery metrics.

### 3. Align estimated components before evaluation

Component order and sign are arbitrary. On synthetic data, match estimated components to true latent variables using a maximum-weight assignment. Suitable matching weights are:

- absolute correlation between estimated and true test scores;
- Jaccard overlap between estimated and true loading supports;
- absolute cosine similarity between loading vectors.

Choose one primary rule before running the benchmark. After assignment, flip each estimated component sign so that its score correlation with the matched true component is positive.

Components left unmatched count as extras; true components left unmatched count as misses.

### 4. Bootstrap the complete procedure

For each training set, bootstrap rows and repeat:

- preprocessing;
- selection of \(K\) and sparsity, when computationally feasible;
- sparse PLS fitting;
- component alignment;
- hard feature assignment.

At minimum, if retuning inside every bootstrap is too expensive, explicitly label the result as **conditional stability given the selected hyperparameters**. It is less complete than full-pipeline stability.

### 5. Compare with informative baselines

Include:

- PCA or factor analysis on \(X\) only;
- dense PLS2;
- sparse regression fitted independently to each outcome;
- an oracle model using true \(Z\).

The unsupervised baseline tests whether supervision by \(Y\) improves recovery of outcome-relevant groups. The oracle gives the best attainable predictive performance when the true latent scores are observed.

## Metrics to print

### Number of components

```text
K_true
K_selected
K_absolute_error = abs(K_selected - K_true_relevant)
K_exact_recovery  = 0/1
```

When some true latent variables are outcome-irrelevant, state whether the target is all factors or only outcome-relevant factors. For sparse PLS, the default target should be \(K_{\text{true,relevant}}\).

### Feature support recovery

Let \(S_{\text{true}}\) be the union of outcome-relevant loading supports and \(S_{\text{est}}\) the union of selected features. Print:

- support precision, recall, and F1;
- false discovery rate;
- false negative rate;
- number and fraction of noise features selected;
- proportion of weak-loading features left unassigned.

Also print these metrics per matched component and macro-average them so that a large group does not hide failure on a small group.

### Group-assignment recovery

On features belonging to true relevant groups, report:

- adjusted Rand index (ARI);
- normalized mutual information (NMI);
- cluster purity;
- macro group Jaccard after optimal component matching;
- assignment coverage, the fraction of true relevant features not left unassigned;
- assignment consistency across bootstrap samples.

Compute a second ARI that includes `NOISE` as a class. This exposes methods that obtain high within-signal ARI by assigning many distractors to signal groups.

### Loading and score recovery

After component matching and sign alignment, print:

\[
\operatorname{loading\ cosine}_k
=
\frac{\widehat w_k^{\mathsf T}\lambda_k}
     {\|\widehat w_k\|_2\|\lambda_k\|_2},
\]

restricted to a common feature coordinate system.

Also report:

- correlation between \(\widehat z_k\) and \(z_k\) on test rows;
- test \(R^2\) for predicting each true latent score;
- principal-angle or subspace error when individual components are not identifiable but their span is;
- loading-support sign accuracy for nonzero matched loadings.

The subspace metric is important when two true latent variables are nearly collinear and can rotate without materially changing prediction.

### Outcome prediction

For every outcome, print:

- RMSE and normalized RMSE;
- \(R^2\) or predictive \(Q^2\);
- calibration slope/intercept for continuous predictions when useful;
- log-loss and AUROC for binary outcomes, if the implementation supports them.

Also print the joint standardized loss used for model selection, but never use prediction alone as evidence of correct grouping.

### Stability

For each feature and component:

- selection frequency;
- modal component assignment;
- conditional assignment probability given selection;
- median absolute loading and interval.

At the model level, print:

- distribution of selected \(K\);
- pairwise Jaccard similarity of selected supports;
- ARI between bootstrap groupings after alignment;
- probability that each true group is recovered with group Jaccard above a predeclared threshold.

### Null calibration

Under \(Y\perp X\), print:

- probability of selecting any component;
- mean number of selected features;
- probability that a feature exceeds the stability threshold;
- apparent test \(R^2\), which should be near or below zero after honest validation.

A method that always returns at least one component must be accompanied by a separate global relevance test or an explicit rule for declaring no stable latent structure.

## Minimal benchmark pseudocode

```python
for scenario in scenarios:
    records = []
    for seed in evaluation_seeds:
        train, test, truth = generate_latent_data(scenario, seed)

        fitted = nested_cv_sparse_pls2(
            train.X,
            train.Y,
            candidate_components=K_grid,
            candidate_sparsity=sparsity_grid,
        )

        estimated_scores = fitted.transform(test.X)
        estimated_groups = hard_assign(
            fitted.loadings,
            threshold=fitted.assignment_threshold,
        )

        matching = match_components(
            estimated_scores=estimated_scores,
            true_scores=truth.Z_test,
            estimated_groups=estimated_groups,
            true_groups=truth.true_group,
        )

        stability = bootstrap_full_pipeline(train, fitted.search_space)

        records.append(
            score_latent_recovery(
                fitted=fitted,
                matching=matching,
                stability=stability,
                test=test,
                truth=truth,
            )
        )

    print_metrics_by_scenario(records)
```

Use enough repetitions to estimate false-selection probabilities and power with useful uncertainty. Include binomial confidence intervals for rates such as exact group recovery and null false-positive probability.

## Suggested compact report

```text
scenario=correlated_latents  seeds=200  S_train=600  S_test=1000
N=80  M=4  K_true_relevant=4
K_selected_median=4  K_exact_recovery=0.74
support_precision=0.91  support_recall=0.86  support_F1=0.88
ARI_signal=0.82  ARI_with_noise=0.77  assignment_coverage=0.89
mean_group_Jaccard=0.84  mean_score_correlation=0.91
mean_loading_cosine=0.87  subspace_error=0.12
joint_test_NRMSE=0.68  oracle_joint_NRMSE=0.61
mean_bootstrap_support_Jaccard=0.79
null_any_component_rate=0.048
```

The final interpretation should say that the recovered components are **outcome-relevant predictive components**. Even excellent recovery in this synthetic factor model does not turn sparse PLS into a causal-identification method on observational data.
