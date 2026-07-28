# Data generation and testing for temporal conditional-mechanism splits

Corresponding method: [`../methods/temporal_split_detection.md`](../methods/temporal_split_detection.md)

## Detection target

This method searches for time points at which the same-row conditional mechanism

\[
P(Y_t\mid X_t)
\]

changes. It does **not** use \(X_{t-1}\) to explain \(Y_t\); that lagged setting is covered by `temporal_split_detection_method.md`.

Let the ordered table contain \(T\) rows and true breakpoints

\[
0=\tau_0<\tau_1<\cdots<\tau_K<\tau_{K+1}=T.
\]

For \(\tau_{k-1}<t\le\tau_k\), generate

\[
Y_t=a_k+B_k^{\mathsf T}h(X_t)+\varepsilon_t.
\]

The benchmark should maintain at least two truth labels:

- **conditional-mean breakpoints**, where \(a_k\), \(B_k\), or the response function changes;
- **slope/mechanism breakpoints**, where \(B_k\) or the nonconstant part of the response function changes.

This distinction allows the evaluation to separate a change in the feature-to-outcome relationship from an intercept-only or variance-only change.

## Recommended output contract for the generator

```text
X                         T x N ordered feature matrix
Y                         T x M ordered outcome matrix
timestamps                length-T strictly ordered values
true_breakpoints_all      all changes in P(Y|X) represented by the generator
true_breakpoints_mean     changes in conditional mean
true_breakpoints_slope    changes in feature-to-outcome mapping
segment_id_true           length-T segment labels
intercepts_true           one vector per segment
coefficients_true         one N x M matrix per segment
noise_covariances_true    one M x M matrix per segment
x_process_metadata        autocorrelation and distribution-shift information
scenario
seed
```

Use the convention that a breakpoint \(\tau\) lies **after row \(\tau\)**: one segment ends at \(\tau\), and the next starts at \(\tau+1\).

## Default synthetic generator

### 1. Choose segment boundaries

For a single-break benchmark, choose a break fraction \(q\in\{0.25,0.5,0.75\}\) and set

\[
\tau_1=\lfloor qT\rfloor.
\]

For multiple breaks, sample segment lengths subject to a fixed minimum \(L_{\min}\). Record the boundaries before generating any outcomes. Recommended tests include:

- no break;
- one balanced break;
- one boundary break with a short first or last segment;
- two or four well-separated breaks;
- two breaks closer than the method's minimum segment length, as an explicit nonidentifiable stress case.

### 2. Generate the feature process

The simplest design is

\[
X_t\stackrel{iid}{\sim}\mathcal N(0,\Sigma_X).
\]

A more realistic ordered design is a stable vector autoregression:

\[
X_t=\Phi X_{t-1}+u_t,
\qquad
u_t\sim\mathcal N(0,\Sigma_u),
\]

with spectral radius \(\rho(\Phi)<1\). A diagonal \(\Phi=\phi I\), with \(\phi\in\{0,0.4,0.8\}\), is sufficient to vary temporal dependence.

Add a **feature-distribution-shift negative control** by changing the mean or covariance of \(X_t\) at a known time while keeping \(P(Y_t\mid X_t)\) fixed. For example,

\[
X_t\sim
\begin{cases}
\mathcal N(0,\Sigma_1),&t\le\tau_X,\\
\mathcal N(\mu_2,\Sigma_2),&t>\tau_X,
\end{cases}
\]

but use one common \(a\), \(B\), and noise law for all rows. A correct conditional-mechanism detector should not treat \(\tau_X\) as a target breakpoint merely because \(P(X)\) changed.

### 3. Generate piecewise conditional outcomes

For the linear default,

\[
Y_t=a_k+B_k^{\mathsf T}X_t+\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal N(0,\Sigma_{\varepsilon,k}).
\]

Choose a baseline matrix \(B_1\), then create a controlled jump:

\[
B_{k+1}=B_k+\delta_kD_k,
\qquad \|D_k\|_F=1.
\]

Vary \(\delta_k\) to obtain power curves. For sparse changes, modify only a chosen subset of coefficient entries. For dense changes, perturb the full matrix.

For \(M>1\), generate correlated outcome residuals and vary whether a break affects:

- every outcome;
- one outcome only;
- different coefficient subsets in different outcomes;
- opposite signs across outcomes.

### 4. Generate negative-control changes

Use separate scenarios in which the conditional slope remains fixed but another quantity changes:

**Intercept only**

\[
a_{k+1}\ne a_k,\qquad B_{k+1}=B_k.
\]

**Noise variance only**

\[
\Sigma_{\varepsilon,k+1}\ne\Sigma_{\varepsilon,k},
\qquad a_{k+1}=a_k,\quad B_{k+1}=B_k.
\]

**Feature distribution only**

\[
P_k(X)\ne P_{k+1}(X),
\qquad P_k(Y\mid X)=P_{k+1}(Y\mid X).
\]

**One global nonlinear mechanism**

\[
Y_t=X_{t1}^2+\varepsilon_t
\]

with no breakpoint. Run it once with a correctly specified nonlinear basis and once with a deliberately linear detector. The latter quantifies false splits caused by misspecification.

### 5. Add robustness variations

Vary:

- \(T\), \(N\), and \(M\);
- segment length relative to the number of fitted parameters;
- feature autocorrelation and collinearity;
- residual autocorrelation;
- heavy-tailed or contaminated residuals;
- missing values and irregular timestamps;
- gradual drift instead of an abrupt jump.

Gradual drift has no unique true point. Score it separately using an interval of change or by measuring whether detections fall inside the predeclared transition window.

## Required benchmark scenarios

| Scenario | Truth | Desired result |
| --- | --- | --- |
| Complete null | No change in \(P(Y\mid X)\) | No detected breakpoint at the nominal error rate |
| One coefficient break | One change in \(B\) | High power and small localization error |
| Multiple coefficient breaks | Several changes in \(B\) | Correct count and partition |
| Weak coefficient break | Small \(\delta\) | Smooth power curve with controlled null FPR |
| Sparse coefficient break | Few entries of \(B\) change | Detect with ridge/sparse local model when supported |
| Intercept-only break | Only \(a\) changes | Detect as mean break but not slope break |
| Variance-only break | Only \(\Sigma_\varepsilon\) changes | Either no mean break or explicit distributional-change classification |
| Feature shift only | \(P(X)\) changes | No conditional-mechanism breakpoint |
| Global nonlinear truth | No break, nonlinear \(h\) | No break under correct local specification |
| Misspecified local model | Same truth, linear detector | Expose false-break sensitivity |
| Strong temporal dependence | Autocorrelated \(X\) and residuals | Valid block-bootstrap calibration |
| Short segment | Segment near parameter-count limit | Respect minimum segment size; report non-detection rather than unstable fit |
| Gradual drift | Coefficients move over a window | Detection near/in transition window, scored separately |

## Testing pipeline

### 1. Separate simulation calibration from evaluation

Use one collection of generated sequences to choose:

- local model class;
- ridge strength or other regularization;
- PELT/BIC penalty multiplier;
- minimum segment length;
- edge trimming and significance threshold.

Evaluate the fixed choices on independent seeds. Tuning a penalty to maximize breakpoint F1 on the same simulations that are reported produces optimistic results.

### 2. Preprocess without destroying changes

Use one transformation per complete sequence for offline change-point detection. In particular:

- standardize each outcome with one pooled location and scale, not separately inside candidate segments;
- fit feature encodings without using true breakpoints;
- do not interpolate across long missing intervals without recording that operation;
- preserve row order.

For a deployment-oriented online experiment, fit transformations only on the available prefix and evaluate separately from the offline benchmark.

### 3. Define a segment cost

For every admissible interval \([a,b]\), fit

\[
\widehat B_{a:b}
=\arg\min_B
\sum_{t=a}^{b}\|Y_t-a-B^{\mathsf T}X_t\|_2^2
+\lambda\|B\|_F^2.
\]

Use \(\lambda=0\) only when every segment has many more rows than parameters and the design is well conditioned. Keep the regularization rule fixed across candidate segments; segment-specific tuning can make costs incomparable.

Define the segment cost as residual sum of squares, negative log-likelihood, or cross-fitted loss. Record exactly whether the cost is sensitive to variance changes as well as mean changes.

### 4. Detect one or several breakpoints

For one possible split, scan all admissible \(\tau\) and maximize the pooled-versus-split improvement. For multiple splits, run PELT or another exact penalized segmentation routine with the same cost and minimum segment length.

The search output should contain both:

- the unpenalized gain at every candidate point;
- the final penalized breakpoint set.

This makes threshold and penalty behavior auditable.

### 5. Calibrate significance by rerunning the full search

Under the null, fit one pooled model while retaining the observed \(X_t\) sequence. Generate bootstrap outcomes by resampling residual vectors:

- resample the complete \(M\)-dimensional residual vector together;
- use moving-block or stationary bootstrap when residuals are temporally dependent;
- add residuals back to the pooled fitted values;
- rerun preprocessing choices that are data dependent and rerun the complete scan or PELT procedure.

For a one-break scan, compare the observed maximum gain with the bootstrap distribution of the maximum gain. For multiple breaks, calibrate a global statistic such as the improvement of the selected segmentation or the number of selected breaks.

### 6. Refit and classify detected changes

After segmentation, refit each segment model and compare adjacent segments. For every detected breakpoint, print separate evidence for:

- intercept change;
- coefficient change \(B_k\ne B_{k+1}\);
- residual covariance or scale change.

A breakpoint should count as a **slope/mechanism detection** only when the coefficient-change criterion passes. This prevents intercept-only and variance-only controls from being counted as successes for the wrong target.

### 7. Use independent predictive evaluation

Segmentation uses the complete discovery sequence, so in-sample loss improvement is optimistic. For synthetic tests, generate a paired evaluation sequence with:

- the same length and true breakpoint locations;
- the same segment parameters;
- independently sampled \(X\) and residuals.

Apply the estimated boundaries and segment models to that sequence. Alternatively, use blocked cross-fitting inside the discovery sequence, but state the dependence between boundary selection and loss estimation.

## Breakpoint matching rule

Use a one-to-one matching between true and detected breakpoints. Let the tolerance be

\[
w=\max(5,\lceil 0.01T\rceil)
\]

or another predeclared value appropriate to the application.

Construct all true-estimated pairs within distance \(w\), then find the matching that:

1. maximizes the number of matched pairs;
2. among ties, minimizes total absolute localization error.

Do not independently map every true breakpoint to its nearest estimate, because one detected point could then receive credit for several true points.

Report metrics over several tolerance values, such as 0, 1%, 2%, and 5% of \(T\), to show localization sensitivity.

## Metrics to print

### Breakpoint-set recovery

For each truth type (`all`, `mean`, and `slope`), print:

- breakpoint precision, recall, and F1 within tolerance;
- false discovery rate;
- number of true and detected breakpoints;
- absolute breakpoint-count error;
- exact-set recovery rate;
- probability of detecting any break under the null.

Across null simulations, the last quantity is the family-wise false-positive rate of the complete search.

### Localization

For matched breakpoints, print:

- mean and median absolute localization error;
- root mean squared localization error;
- normalized error \(|\widehat\tau-\tau|/T\);
- maximum matched error;
- Hausdorff distance between breakpoint sets;
- bootstrap confidence-interval coverage and mean interval width.

When no valid match exists, localization metrics should be marked missing rather than silently assigned zero.

### Segment partition quality

Compare the true and estimated segment label of every row using:

- adjusted Rand index;
- normalized mutual information;
- row-wise segment accuracy after optimal label matching;
- boundary displacement loss, the number of rows assigned to the wrong temporal segment divided by \(T\).

ARI is invariant to arbitrary segment labels, while boundary displacement is directly interpretable in rows or time units.

### Parameter recovery

After matching estimated and true temporal segments, report:

\[
\operatorname{relative\ coefficient\ error}
=\frac{\sum_k\|\widehat B_k-B_k\|_F}
       {\sum_k\|B_k\|_F+\epsilon}.
\]

Also print:

- intercept RMSE;
- jump-size error \(\|\widehat B_{k+1}-\widehat B_k\|_F-\|B_{k+1}-B_k\|_F\);
- support precision/recall if coefficient changes are sparse;
- outcome-specific coefficient error.

### Predictive quality

On the paired evaluation sequence, print:

- pooled-model joint standardized loss;
- segmented-model joint standardized loss;
- oracle-segmentation loss;
- fraction of oracle loss improvement recovered;
- per-outcome RMSE, log-loss, or deviance.

Good breakpoint recovery should normally improve conditional prediction, but predictive improvement alone cannot establish correct localization.

### Calibration, power, and robustness

Across seeds, print:

- null false-positive rate with a binomial confidence interval;
- detection power by jump magnitude \(\delta\);
- localization error conditional on detection;
- performance by minimum segment length, autocorrelation, and \(N/T\);
- rate of false slope attribution in intercept-only, variance-only, and feature-shift controls.

A useful power plot has \(\delta\) on the horizontal axis and both detection power and median localization error on the vertical summaries.

## Minimal benchmark pseudocode

```python
calibration_sequences = generate_sequences(calibration_seeds, scenarios)
config = tune_segmentation_pipeline(
    calibration_sequences,
    constraints={"null_FPR": 0.05},
    objective="slope_break_F1",
)

records = []
for scenario in scenarios:
    for seed in evaluation_seeds:
        discovery, paired_test, truth = generate_sequence_pair(scenario, seed)

        fitted = detect_breakpoints(
            discovery.X,
            discovery.Y,
            config=config,
        )

        bootstrap = bootstrap_full_search(
            discovery.X,
            discovery.Y,
            fitted=fitted,
            config=config,
            block_residuals=scenario.has_temporal_dependence,
        )

        classified = classify_adjacent_segment_changes(fitted, bootstrap)
        matching = match_breakpoints(
            classified.slope_breakpoints,
            truth.true_breakpoints_slope,
            tolerance=max(5, ceil(0.01 * truth.T)),
        )

        records.append(
            evaluate_temporal_detection(
                fitted=classified,
                matching=matching,
                bootstrap=bootstrap,
                paired_test=paired_test,
                truth=truth,
            )
        )

print_metrics_by_scenario(records)
```

## Suggested compact report

```text
scenario=two_slope_breaks  seeds=250  T=1200  N=12  M=3
true_breaks=2  mean_detected_breaks=2.08  count_MAE=0.24
break_precision_1pct=0.91  break_recall_1pct=0.89  break_F1_1pct=0.90
median_localization_error=4 rows  normalized_Hausdorff=0.009
segment_ARI=0.94  boundary_displacement=0.018
relative_coefficient_error=0.13  jump_size_MAE=0.09
joint_test_loss_pooled=1.00  segmented=0.73  oracle=0.69
oracle_improvement_recovered=0.87
null_any_break_rate=0.048
false_slope_attribution_intercept_only=0.031
false_slope_attribution_feature_shift_only=0.024
```

The final interpretation should describe detected points as changes in the observed conditional mechanism \(P(Y_t\mid X_t)\). A causal-model interpretation requires the additional assumptions stated in the corresponding method document.
