# Data generation and testing for relationships between latent variables

Corresponding method: [`../methods/latent_variable_determination2.md`](../methods/latent_variable_determination2.md)

## Detection target

The recommended default pipeline is

\[
\text{sparse PLS grouping}
\rightarrow
\text{independently reconstructed group scores}
\rightarrow
\text{partial-correlation network}.
\]

Its primary detection target is therefore an **undirected conditional-dependence graph** between outcome-relevant latent groups. A detected edge means that two reconstructed group scores remain associated after conditioning on the other group scores. It does not, by itself, identify a causal direction.

The benchmark should separate two sources of error:

1. **relationship-estimation error**, assuming the true groups or true latent scores are available;
2. **end-to-end error**, after groups and scores have first been estimated from \(X\) and \(Y\).

Without this separation, poor graph recovery cannot be attributed to the network estimator or to the upstream grouping step.

## Recommended output contract for the generator

```text
X                         S x N observed-feature matrix
Y                         S x M outcome matrix
Z_true                    S x K_true latent-score matrix
Lambda_true               N x K_true measurement/loading matrix
Gamma_true                K_true x M outcome coefficient matrix
true_group[j]             true latent group of feature j, or NOISE
Theta_true                K_true x K_true precision matrix, for undirected tests
edge_set_true             unordered latent-variable pairs
partial_corr_true         K_true x K_true matrix
B_dag_true                optional directed coefficient matrix
cpdag_or_pag_truth         optional equivalence-class target
hidden_variables          optional metadata for confounded scenarios
scenario
seed
```

Store both the graph support and edge weights. Graph support is invariant to latent-score sign changes; signed edge weights are not and require score-sign alignment before evaluation.

## Primary undirected data generator

### 1. Choose a sparse latent graph

Select \(K_{\text{true}}\) and an undirected graph \(G=(V,E)\). Include several graph families:

- empty graph;
- chain;
- fork or hub;
- disconnected components;
- sparse Erdős–Rényi graph;
- scale-free-like sparse graph, when degree imbalance is of interest.

To construct a valid Gaussian graphical model, create a symmetric precision matrix \(\Theta\). For every edge \((k,\ell)\in E\), assign

\[
\Theta_{k\ell}=\Theta_{\ell k}\in[-w_{\max},-w_{\min}]\cup[w_{\min},w_{\max}].
\]

Then make the matrix strictly diagonally dominant:

\[
\Theta_{kk}=\delta+\sum_{\ell\ne k}|\Theta_{k\ell}|,
\qquad \delta>0.
\]

This guarantees positive definiteness. Set

\[
\Sigma_Z=\Theta^{-1},
\]

rescale it to unit marginal variances if desired, and sample

\[
Z_i\sim\mathcal N(0,\Sigma_Z).
\]

The true partial correlations are

\[
\rho_{k\ell\mid-\{k,\ell\}}
=-\frac{\Theta_{k\ell}}
{\sqrt{\Theta_{kk}\Theta_{\ell\ell}}}.
\]

Control difficulty by varying the smallest nonzero absolute partial correlation, graph density, maximum degree, \(K/S\), and the condition number of \(\Sigma_Z\).

### 2. Generate observed feature groups

Use a sparse measurement model

\[
X=Z\Lambda^{\mathsf T}+E_X.
\]

For the clean benchmark, every observed feature loads on exactly one latent variable. Use at least three indicators per latent variable when the scores are reconstructed through a factor model; larger groups permit more stable score estimation.

For feature \(j\) in group \(k\), generate

\[
X_{ij}=\lambda_j Z_{ik}+e_{ij},
\qquad
|\lambda_j|\in[\lambda_{\min},\lambda_{\max}].
\]

Add:

- independent noise features;
- internally correlated nuisance blocks unrelated to \(Z\) and \(Y\);
- optional cross-loading indicators;
- unequal group sizes;
- different measurement-error variances by group.

Record the exact loading support and the primary group of every feature.

### 3. Generate outcomes for supervised grouping

Use

\[
Y=Z\Gamma+E_Y.
\]

In the primary end-to-end benchmark, give every latent variable at least one nonzero coefficient in \(\Gamma\). Otherwise sparse PLS can correctly omit a latent group that is not outcome-relevant, making full graph recovery impossible by design.

Add a separate **partial observability** scenario in which some latent variables are not related to \(Y\). In that scenario, evaluate recovery only for the induced graph over the outcome-relevant latent variables, and explicitly report omitted true nodes.

### 4. Generate independent train and test rows

Generate train and test rows from the same \(\Theta\), \(\Lambda\), and \(\Gamma\). Use training rows for grouping, score-construction choices, graphical-lasso tuning, and stability selection. Use test rows for held-out likelihood and score-recovery metrics.

Graph support metrics are evaluated against the known generator truth and aggregated over independent seeds.

## Optional directed generator

When testing SEM, PC/FCI, or LiNGAM extensions, generate a known acyclic structural model:

\[
Z=BZ+\eta,
\qquad
Z=(I-B)^{-1}\eta.
\]

Construct \(B\) from a random topological order, keep it sparse, and reject draws for which coefficients produce numerically unstable covariance. Choose the error distribution to match the method being tested:

- Gaussian independent errors for PC under linear-Gaussian conditional-independence tests;
- non-Gaussian independent errors for LiNGAM;
- an explicit hidden common cause for FCI/RFCI tests.

The primary directed target should match the algorithm's identifiable output:

- DAG only when the assumptions identify a DAG;
- CPDAG for PC or score-equivalent DAG discovery;
- PAG and endpoint marks for FCI/RFCI.

Do not score an algorithm against directions that are not identifiable from its target representation.

## Benchmark levels

Run all important scenarios at three levels.

### Level A: oracle latent scores

Give the network estimator \(Z_{\text{true}}\) directly. This isolates partial-correlation or causal-discovery performance.

### Level B: oracle groups, estimated scores

Give the procedure the correct feature groups but require it to reconstruct each score independently from \(X_{G_k}\). Compare:

- equal standardized weights;
- first principal component per group;
- one-factor scores;
- sparse-PLS weights restricted to the known group.

This isolates measurement and score-estimation error.

### Level C: full end-to-end pipeline

Estimate groups with sparse PLS, reconstruct non-orthogonal scores, and estimate the graph. This is the result most representative of actual use.

Print the metric difference between levels. For example, a large drop from A to B indicates weak measurement; a large drop from B to C indicates unstable grouping.

## Required benchmark scenarios

| Scenario | Main variation | Purpose |
| --- | --- | --- |
| Empty graph | \(\Theta\) diagonal | Measure graph false-positive rate |
| Chain | Sparse path graph | Test removal of indirect pairwise correlations |
| Hub | One high-degree node | Test degree-related selection bias |
| Disconnected graph | Several components | Test spurious bridges |
| Weak edges | Small nonzero partial correlations | Estimate edge-detection power curve |
| Dense-near-limit graph | More edges but still identifiable | Stress graphical-lasso sparsity assumptions |
| Small sample | Large \(K/S\) | Test regularization and stability |
| Ill-conditioned covariance | Highly correlated latent scores | Test numerical stability |
| Weak measurement | Larger \(E_X\) | Quantify attenuation from reconstructed scores |
| Unequal group quality | One group has few/noisy indicators | Detect endpoint-specific reliability problems |
| Cross-loadings | Some indicators measure two latents | Test spurious latent edges caused by measurement overlap |
| Grouping error | Weak or overlapping outcome relevance | Evaluate full-pipeline uncertainty |
| Outcome-irrelevant node | A latent has no path to \(Y\) | Evaluate induced relevant-node graph and node omission |
| Hidden confounding | Directed extension with unobserved common cause | Compare PC against FCI/RFCI behavior |
| Temporal data | Repeated \(Z_t\) from a VAR | Test the separate lagged-network extension, not the cross-sectional default |

## End-to-end testing pipeline

### 1. Learn groups without leaking test rows

Fit sparse PLS and all preprocessing on training rows only. Select component count, sparsity, and assignment threshold by nested validation. Bootstrap group stability.

### 2. Reconstruct non-orthogonal group scores

For each estimated group \(\widehat G_k\), fit the score rule independently:

\[
\widehat z_k=X_{\widehat G_k}\widehat w_k.
\]

Do not use the sequentially deflated sparse-PLS score vectors as the input to the relationship network. Their imposed orthogonality can erase the target correlations.

Fit each score rule on training rows and apply it unchanged to test rows. Standardize reconstructed scores using training means and scales.

### 3. Align estimated groups to truth for evaluation

Use maximum-weight bipartite assignment, preferably with feature-support Jaccard as the primary matching weight. Score correlation can be used as a secondary tie-breaker.

After group matching, align score signs using the correlation between reconstructed and true training scores. The evaluator may use the synthetic truth for sign alignment; the fitting algorithm may not.

### 4. Estimate the partial-correlation network

For small \(K\) and sufficiently large \(S\), estimate the inverse covariance directly with shrinkage if necessary. Otherwise fit graphical lasso over a penalty grid.

Select the penalty without using true edges. Suitable criteria are:

- inner-fold held-out Gaussian negative log-likelihood;
- stability selection with a predeclared edge-frequency threshold;
- an information criterion computed on training data.

When a validation likelihood selects a dense graph, apply a separately justified edge threshold or stability rule and report both weighted and binary graph metrics.

### 5. Bootstrap the entire pipeline

Each bootstrap replicate should redo:

1. preprocessing;
2. sparse-PLS grouping;
3. group matching/alignment to the reference solution;
4. independent score reconstruction;
5. graphical-lasso tuning or fitting;
6. edge extraction.

For every estimated edge, retain its selection frequency and edge-weight distribution. A bootstrap that starts from fixed estimated scores ignores the largest source of uncertainty in the full pipeline.

### 6. Evaluate an untouched test set

Use test rows to calculate:

- held-out Gaussian negative log-likelihood of the reconstructed-score model;
- correlation of reconstructed and true scores;
- covariance and partial-correlation estimation error.

The binary true graph is fixed by the generator, so adjacency metrics can be computed after fitting, but hyperparameters must not be adjusted based on those test metrics.

## Graph matching and scoring rules

### Node matching

Latent labels are arbitrary. Match estimated nodes to true nodes before graph comparison. If the number of estimated and true nodes differs:

- unmatched true nodes count as missed nodes and all their incident edges as false negatives;
- unmatched estimated nodes count as extra nodes and their incident edges as false positives.

Also print graph metrics conditional on correctly matched nodes. The unconditional result measures the full pipeline; the conditional result isolates edge estimation.

### Edge thresholding

Predeclare how a weighted precision estimate becomes an edge. Examples:

- nonzero graphical-lasso coefficient;
- absolute partial correlation above \(\tau_\rho\);
- bootstrap selection frequency above \(\pi_{\text{thr}}\).

For a complete performance curve, vary the threshold and report AUPRC. Use AUROC only as a secondary metric because sparse graphs have many more nonedges than edges.

### Sign handling

The sign of a latent score is arbitrary. Binary edge presence is unaffected. For signed edge evaluation, first align every estimated score sign to its matched true score. Only then compare partial-correlation signs.

## Metrics to print

### Upstream grouping and score quality

Reuse the metrics from `latent_variable_determination.md`:

- selected \(K\) and node-count error;
- feature-support precision, recall, and F1;
- group ARI/NMI and mean group Jaccard;
- reconstructed-score correlation and \(R^2\);
- measurement reliability per group.

These should appear beside graph metrics in the end-to-end report.

### Binary graph recovery

For the unordered adjacency set, print:

- edge precision, recall, and F1;
- false discovery rate and false negative rate;
- specificity;
- Matthews correlation coefficient;
- structural Hamming distance (edge additions/deletions);
- normalized structural Hamming distance, divided by \(K(K-1)/2\);
- AUPRC over edge scores;
- exact graph-recovery rate over simulation seeds.

Under the empty graph, the most important quantities are the probability of any false edge and the mean number of false edges.

### Edge localization and topology

- per-edge detection probability as a function of true \(|\rho|\);
- degree correlation between true and estimated graphs;
- connected-component recovery;
- number of spurious bridges between true disconnected components;
- hub-node identification accuracy, when applicable.

These reveal errors that a single global F1 can hide.

### Weighted network recovery

After node and sign alignment, print

\[
\operatorname{partial\ correlation\ RMSE}
=
\sqrt{\frac{2}{K(K-1)}
\sum_{k<\ell}
(\widehat\rho_{k\ell}-\rho_{k\ell})^2}.
\]

Also report:

- RMSE on true edges only;
- RMSE on true nonedges;
- sign accuracy on detected true edges;
- Spearman correlation between true and estimated absolute edge weights.

### Generalization and stability

- held-out Gaussian negative log-likelihood;
- covariance prediction error on test rows;
- edge-selection frequency;
- median edge weight and bootstrap interval;
- pairwise Jaccard similarity of bootstrap edge sets;
- probability of recovering each true connected component.

### Directed-extension metrics

For PC/GES-type outputs:

- adjacency precision/recall/F1;
- arrowhead precision/recall;
- CPDAG structural Hamming distance.

For FCI/RFCI:

- adjacency metrics;
- endpoint-mark precision/recall by tail, arrowhead, and circle;
- PAG structural Hamming distance under a stated endpoint-scoring convention.

For a fully identifiable DAG estimator:

- directed-edge precision/recall/F1;
- reversal count;
- structural Hamming distance;
- structural intervention distance when the complete true DAG and required assumptions make it meaningful.

Never penalize an undirected or partially directed output for refusing to orient an edge that is not identifiable under the tested method.

## Minimal benchmark pseudocode

```python
for scenario in scenarios:
    records = []
    for seed in evaluation_seeds:
        train, test, truth = generate_latent_graph_data(scenario, seed)

        # Level A: network estimation with oracle scores
        oracle_graph = fit_network(train.Z_true, tuning="inner_nll")

        # Level B: known groups, independently estimated scores
        oracle_group_scores = fit_group_scores(train.X, truth.true_groups)
        level_b_graph = fit_network(oracle_group_scores.train, tuning="inner_nll")

        # Level C: complete recommended pipeline
        groups = fit_sparse_pls_groups_nested_cv(train.X, train.Y)
        scores = fit_independent_group_scores(train.X, groups)
        graph = fit_network(scores.train, tuning="inner_nll")
        stability = bootstrap_complete_pipeline(train)

        matching = match_estimated_nodes_to_truth(groups, truth)

        records.append(
            evaluate_all_levels(
                oracle_graph=oracle_graph,
                level_b_graph=level_b_graph,
                full_graph=graph,
                scores=scores,
                stability=stability,
                matching=matching,
                test=test,
                truth=truth,
            )
        )

    print_metrics_by_level_and_scenario(records)
```

## Suggested compact report

```text
scenario=chain_weak_measurement  seeds=200  S_train=800  K=6  N=48
level_A_edge_F1=0.93  level_B_edge_F1=0.82  level_C_edge_F1=0.71
level_C_edge_precision=0.76  edge_recall=0.67  normalized_SHD=0.11
AUPRC=0.79  any_false_edge_rate_empty_graph=0.046
partial_corr_RMSE=0.10  detected_edge_sign_accuracy=0.94
node_support_F1=0.87  group_ARI=0.81  mean_score_corr=0.86
heldout_Gaussian_NLL=7.42  bootstrap_edge_Jaccard=0.73
spurious_bridges_mean=0.08
```

The primary scientific statement supported by this benchmark is that the procedure recovers a stable network of **conditional associations between reconstructed latent groups**. Directional or causal wording belongs only to the optional directed benchmarks and must be tied to their additional assumptions.
