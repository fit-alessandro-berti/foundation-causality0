# Data generation and testing for lagged temporal split detection

Corresponding method: [`../methods/temporal_split_detection_method.md`](../methods/temporal_split_detection_method.md)

## Detection target

The recommended method fits a piecewise VARX(1)-type model

\[
Y_t=c_k+A_kY_{t-1}+B_kX_{t-1}+\varepsilon_t,
\qquad
\tau_{k-1}<t\le\tau_k.
\]

The primary target is the set of temporal breakpoints where the lagged feature coefficient matrix changes:

\[
B_k\ne B_{k+1}.
\]

The benchmark must distinguish these **\(B\)-breaks** from changes in:

- the intercept \(c_k\);
- outcome persistence \(A_k\);
- residual covariance;
- the marginal process generating \(X_t\).

A segmentation algorithm may detect any of these as a generic structural break. The post-segmentation testing stage should retain a breakpoint as a feature-to-outcome mechanism change only when there is evidence that \(B\) changed.

The detected relation is Granger-predictive: lagged \(X\) improves prediction of current \(Y\) beyond lagged \(Y\). It is not automatically an intervention-level causal effect.

## Valid row semantics

Generate rows as consecutive states of one evolving system. Do not create a single sequence by sorting and concatenating independent cases.

For repeated entities, either:

- generate and analyze each trajectory separately;
- fit a panel model that explicitly respects entity boundaries;
- or aggregate the system into fixed calendar windows before creating lags.

Never create \((X_{t-1},Y_t)\) pairs across an entity boundary. The generator should expose a `sequence_id`, even when there is only one trajectory, so that this condition can be asserted in tests.

## Recommended output contract for the generator

```text
X                         T x N ordered feature matrix
Y                         T x M ordered outcome matrix
timestamps                length-T ordered values
sequence_id               length-T entity/trajectory identifier
true_breakpoints_all      all structural breaks in c, A, B, or noise
true_breakpoints_B        breakpoints where B changes
true_breakpoints_A        breakpoints where A changes
true_breakpoints_c        intercept-only or intercept-including changes
true_breakpoints_noise    residual-distribution changes
segment_id_true           length-T segment labels
c_true                    one M-vector per segment
A_true                    one M x M matrix per segment
B_true                    one M x N matrix per segment
Sigma_eps_true            one M x M matrix per segment
B_support_true            nonzero lagged X-to-Y coefficients per segment
X_process_truth           parameters of the exogenous X process
scenario
seed
```

Use the convention that breakpoint \(\tau\) occurs after row \(\tau\), so the next parameter regime begins at \(\tau+1\).

## Default piecewise VARX generator

### 1. Generate a stable exogenous feature process

A simple choice is

\[
X_t=d_k+F_kX_{t-1}+u_t,
\qquad
u_t\sim\mathcal N(0,\Sigma_u),
\]

where \(\rho(F_k)<1\) in every segment. For the clean benchmark, keep \(d_k\), \(F_k\), and \(\Sigma_u\) constant, and use a diagonal

\[
F=\phi I,
\qquad \phi\in\{0,0.4,0.8\}.
\]

This makes \(X_t\) temporally persistent without allowing \(Y\) to feed back into \(X\).

For negative controls, change the mean, persistence, or covariance of \(X_t\) while keeping \(c\), \(A\), \(B\), and the residual law of \(Y\) fixed.

### 2. Generate stable outcome dynamics

Within segment \(k\), generate

\[
Y_t=c_k+A_kY_{t-1}+B_kX_{t-1}+\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal N(0,\Sigma_{\varepsilon,k}).
\]

Require

\[
\rho(A_k)<1.
\]

A practical construction is:

1. draw a sparse matrix \(A_k^{(0)}\);
2. calculate its spectral radius;
3. rescale it to a target radius such as 0.3, 0.6, or 0.85.

For the default clean experiment, hold \(A_k=A\) fixed across all segments and change only \(B_k\).

### 3. Control the size and type of \(B\) changes

Choose a baseline sparse matrix \(B_1\in\mathbb R^{M\times N}\). Create the next segment through

\[
B_{k+1}=B_k+\delta_kD_k,
\qquad
\|D_k\|_F=1.
\]

Use several change types:

- **magnitude change:** retain support but alter coefficient magnitudes;
- **sign reversal:** change selected coefficients from positive to negative;
- **edge addition:** turn selected zero entries on;
- **edge deletion:** turn selected nonzero entries off;
- **mixed sparse change:** modify only a small fraction of entries;
- **dense change:** perturb most entries.

Store both the coefficient matrix and its nonzero support in every segment.

### 4. Use a burn-in period

Initialize \(X_0\) and \(Y_0\), simulate at least 100–500 burn-in rows depending on persistence, and discard them before assigning the reported timestamps. Carry the final state continuously across breakpoints; do not reset the process at each segment.

This prevents arbitrary initialization transients from becoming artificial detections.

### 5. Generate correlated multivariate residuals

For \(M>1\), use a positive-definite residual covariance matrix. Vary:

- independent versus correlated residuals;
- equal versus unequal outcome variances;
- Gaussian versus heavy-tailed innovations;
- residual autocorrelation as a misspecification stress test.

Resample the complete multivariate residual vector in bootstrap procedures so that cross-outcome dependence is preserved.

## Required benchmark scenarios

| Scenario | Parameter change | Expected classification |
| --- | --- | --- |
| Complete null | No parameter changes | No raw break and no \(B\)-break |
| One \(B\)-break | Only \(B\) changes | Detect and retain the breakpoint |
| Several \(B\)-breaks | \(B\) changes at several points | Recover count, locations, and segment-specific matrices |
| Weak \(B\)-break | Small \(\|B_{k+1}-B_k\|_F\) | Controlled power curve |
| \(A\)-only break | Only autoregressive outcome dynamics change | Raw break may be found; must not be labeled a \(B\)-break |
| Intercept-only break | Only \(c\) changes | Raw break may be found; must not be labeled a \(B\)-break |
| Noise-only break | Only \(\Sigma_\varepsilon\) changes | No false \(B\)-attribution |
| \(X\)-process break | Only \(d\), \(F\), or \(\Sigma_u\) changes | No false \(B\)-attribution when the conditional model is stable |
| Mixed \(A+B\) break | Both matrices change | Detect raw and \(B\)-specific change |
| \(B=0\) to nonzero | Lagged feature relevance appears | Detect breakpoint and segment relevance transition |
| Nonzero \(B\) to zero | Lagged feature relevance disappears | Detect breakpoint and segment relevance transition |
| High persistence | \(\rho(A)\) and/or \(\rho(F)\) near 1 | Test blocked validation and bootstrap calibration |
| High-dimensional | \(N+M\) large relative to segment length | Require ridge/sparse fitting and larger minimum segments |
| Wrong lag order | Truth is VARX(2), detector fits VARX(1) | Quantify misspecification-induced breaks or coefficient bias |
| Contemporaneous-only relation | \(Y_t\) depends on \(X_t\), not directly on \(X_{t-1}\) | Interpretation stress test under persistent \(X\), reported separately |
| Close breakpoints | Segment shorter than reliable fit length | Respect minimum segment length and report unresolved changes |

The \(A\)-only, intercept-only, noise-only, and \(X\)-process-only scenarios are essential for measuring the specificity of the post-segmentation \(B\)-change test.

## Simulation design

Vary at least:

- total length \(T\);
- number of features \(N\) and outcomes \(M\);
- number and minimum spacing of breakpoints;
- spectral radii of \(A\) and \(F\);
- sparsity and norm of \(B\);
- jump norm \(\|B_{k+1}-B_k\|_F\);
- residual signal-to-noise ratio;
- residual and feature autocorrelation;
- segment balance.

Use independent seed sets for pipeline calibration and final evaluation. Report results by scenario and signal level rather than averaging incompatible cases into one number.

## Testing pipeline

### 1. Construct the lagged design safely

After sorting by timestamp and checking sequence boundaries, construct

\[
R_t=Y_t,
\qquad
Q_t=[1,Y_{t-1}^{\mathsf T},X_{t-1}^{\mathsf T}]^{\mathsf T}.
\]

Drop the first row of every sequence. Assert programmatically that no lag pair crosses a `sequence_id` boundary.

Use one fixed scaling transformation per discovery sequence, or training-only scaling in a deployment benchmark. Do not standardize separately inside candidate segments because that can erase intercept and variance changes and make segment costs incomparable.

### 2. Define the full segment cost

For candidate interval \([a,b]\), fit

\[
(\widehat c,\widehat A,\widehat B)
=
\arg\min_{c,A,B}
\sum_{t=a}^{b}
\|Y_t-c-AY_{t-1}-BX_{t-1}\|_2^2
+\lambda_A\|A\|_F^2
+\lambda_B\|B\|_F^2.
\]

Use ordinary least squares only when the segment is substantially longer than \(1+M+N\) and the design is well conditioned. Choose regularization strengths on separate calibration sequences or through blocked inner validation. Hold them fixed while comparing candidate segments.

### 3. Detect generic structural breaks

Run a one-break scan, PELT, or Bai–Perron-type procedure using the full segment cost. Enforce a minimum segment length large enough for stable estimation.

Retain the complete raw breakpoint set before classifying the parameter responsible for each change.

### 4. Test whether \(B\) changes at each candidate breakpoint

For two adjacent estimated segments \(L\) and \(R\), compare:

**Restricted adjacent-segment model**

\[
Y_t=
\begin{cases}
c_L+A_LY_{t-1}+BX_{t-1}+\varepsilon_t,&t\in L,\\
c_R+A_RY_{t-1}+BX_{t-1}+\varepsilon_t,&t\in R,
\end{cases}
\]

which permits separate intercepts and \(A\) matrices but one shared \(B\), against:

**Full adjacent-segment model**

\[
Y_t=
\begin{cases}
c_L+A_LY_{t-1}+B_LX_{t-1}+\varepsilon_t,&t\in L,\\
c_R+A_RY_{t-1}+B_RX_{t-1}+\varepsilon_t,&t\in R.
\end{cases}
\]

The full-versus-restricted comparison isolates the hypothesis

\[
H_0:B_L=B_R.
\]

Use a joint Wald/F test only when its assumptions are credible. Otherwise use a residual or wild block bootstrap that refits both models and repeats the comparison.

Correct for testing several detected boundaries, or calibrate the complete select-then-test procedure by simulation/bootstrap.

### 5. Test whether lagged \(X\) is relevant inside each segment

Within each segment, compare:

\[
Y_t=c_k+A_kY_{t-1}+\varepsilon_t
\]

against

\[
Y_t=c_k+A_kY_{t-1}+B_kX_{t-1}+\varepsilon_t.
\]

Measure either a joint statistical test or the reduction in blocked out-of-sample loss. Record whether each segment has detectable nonzero \(B_k\).

A breakpoint can represent a genuine \(B\) change even if one side has \(B=0\). Therefore, the equality test across segments is the primary classification criterion; within-segment relevance indicates the nature of the change.

### 6. Bootstrap the full procedure

A valid bootstrap should repeat:

1. pooled or segmentwise residual generation under the relevant null;
2. the complete breakpoint search;
3. segment refitting;
4. adjacent-segment \(B\)-equality tests;
5. within-segment full-versus-restricted tests;
6. any multiplicity correction and retention rule.

For dependent residuals, use blocks. Keep the observed \(X\) sequence fixed when calibrating whether the conditional lagged outcome mechanism changed.

### 7. Evaluate forecasting on independent data

Generate a paired test trajectory with the same boundaries and parameters but independent innovations. Initialize it with a fresh burn-in. Apply the learned boundaries and segment models.

Compare:

- pooled autoregressive model \(Y_t\sim Y_{t-1}\);
- pooled VARX model \(Y_t\sim Y_{t-1}+X_{t-1}\);
- estimated piecewise VARX model;
- oracle piecewise VARX model using true boundaries;
- oracle-parameter model, as a noise floor.

This separates the value of lagged features from the value of segmentation.

## Breakpoint matching

Use one-to-one optimal matching within a predeclared tolerance

\[
w=\max(5,\lceil0.01T\rceil).
\]

Evaluate two breakpoint sets independently:

1. raw detected breaks against `true_breakpoints_all`;
2. retained \(B\)-breaks against `true_breakpoints_B`.

A raw detection at an \(A\)-only change is not a raw false positive, but it is a false positive if retained as a \(B\)-break.

Report results at multiple tolerances to distinguish near misses from grossly incorrect segmentation.

## Metrics to print

### Raw structural-break detection

- raw breakpoint precision, recall, and F1 against all structural changes;
- raw breakpoint-count error;
- localization MAE, RMSE, and normalized Hausdorff distance;
- temporal-segment ARI and boundary displacement;
- global no-break false-positive rate.

### \(B\)-specific detection

- \(B\)-break precision, recall, and F1;
- exact \(B\)-break-set recovery rate;
- \(B\)-break localization error;
- power as a function of \(\|B_{k+1}-B_k\|_F\);
- probability of detecting a transition from zero to nonzero \(B\);
- probability of detecting a transition from nonzero to zero \(B\).

### False attribution

Print the probability that the pipeline labels a breakpoint as a \(B\)-change in each negative-control scenario:

```text
false_B_attribution_A_only
false_B_attribution_intercept_only
false_B_attribution_noise_only
false_B_attribution_X_process_only
```

These are among the most important specificity metrics for this method.

### Coefficient and support recovery

After matching temporal segments, print

\[
\operatorname{relative\ B\ error}
=\frac{\sum_k\|\widehat B_k-B_k\|_F}
       {\sum_k\|B_k\|_F+\epsilon},
\]

and the analogous relative error for \(A\).

When \(B\) is sparse, treat each nonzero entry \(B_{mn}\) as a directed lagged predictive edge \(X_n(t-1)\to Y_m(t)\) and print:

- coefficient-support precision, recall, F1, and false discovery rate;
- sign accuracy on true detected coefficients;
- coefficient RMSE on true nonzeros;
- segment-specific and macro-averaged metrics.

Also print error in the jump matrix

\[
\Delta B_k=B_{k+1}-B_k.
\]

Support recovery should be omitted or interpreted carefully for ridge estimates unless a predeclared threshold converts coefficients into edges.

### Within-segment relevance detection

For each segment, classify whether \(B_k\) is zero or nonzero. Across simulations, print:

- segment relevance precision, recall, and F1;
- false positive rate when \(B_k=0\);
- power by \(\|B_k\|_F\);
- blocked-validation loss gain from adding \(X_{t-1}\).

### Forecasting quality

On the paired test trajectory, print per outcome and jointly standardized:

- one-step RMSE or negative log-likelihood;
- improvement of pooled VARX over pooled autoregression;
- improvement of estimated piecewise VARX over pooled VARX;
- fraction of oracle segmentation improvement recovered;
- calibration of predictive intervals, when available.

### Stability and uncertainty

- bootstrap selection frequency for every breakpoint neighborhood;
- bootstrap interval for breakpoint location;
- bootstrap probability that each retained break is a \(B\)-change;
- interval coverage for selected entries of \(B\) and \(\Delta B\);
- Jaccard similarity of retained \(B\)-break sets across resamples.

Selection-aware intervals are preferred. Intervals calculated after fixing the discovered boundaries should be labeled conditional on those boundaries.

## Minimal benchmark pseudocode

```python
config = tune_on_independent_simulations(
    calibration_seeds,
    objective="B_break_F1",
    constraints={
        "complete_null_FPR": 0.05,
        "A_only_false_B_attribution": 0.05,
    },
)

records = []
for scenario in scenarios:
    for seed in evaluation_seeds:
        discovery, paired_test, truth = generate_piecewise_varx_pair(
            scenario=scenario,
            seed=seed,
            burn_in=config.burn_in,
        )

        lagged = make_lagged_table(
            discovery.X,
            discovery.Y,
            discovery.sequence_id,
        )

        raw = detect_structural_breaks(lagged, config)
        classified = test_B_changes_at_detected_breaks(lagged, raw, config)
        relevance = test_X_lag_relevance_by_segment(lagged, classified, config)
        bootstrap = bootstrap_complete_select_and_test_pipeline(
            lagged,
            config,
        )

        raw_matching = match_breakpoints(
            raw.breakpoints,
            truth.true_breakpoints_all,
            tolerance=config.tolerance,
        )
        B_matching = match_breakpoints(
            classified.B_breakpoints,
            truth.true_breakpoints_B,
            tolerance=config.tolerance,
        )

        records.append(
            evaluate_varx_detection(
                raw=raw,
                classified=classified,
                relevance=relevance,
                bootstrap=bootstrap,
                raw_matching=raw_matching,
                B_matching=B_matching,
                paired_test=paired_test,
                truth=truth,
            )
        )

print_metrics_by_scenario(records)
```

## Suggested compact report

```text
scenario=mixed_A_and_B_breaks  seeds=250  T=1600  N=10  M=4
raw_break_precision=0.93  raw_break_recall=0.91  raw_break_F1=0.92
B_break_precision=0.90  B_break_recall=0.86  B_break_F1=0.88
B_break_localization_median=5 rows  segment_ARI=0.95
relative_A_error=0.12  relative_B_error=0.15  delta_B_error=0.11
B_support_precision=0.88  B_support_recall=0.81  B_support_F1=0.84
segment_relevance_F1=0.91
piecewise_VARX_test_RMSE=0.74  pooled_VARX_test_RMSE=0.88
oracle_segmentation_improvement_recovered=0.83
complete_null_any_break_rate=0.044
false_B_attribution_A_only=0.036
false_B_attribution_intercept_only=0.028
false_B_attribution_noise_only=0.040
false_B_attribution_X_process_only=0.024
```

The benchmark should report both generic segmentation quality and \(B\)-specific classification quality. High raw breakpoint F1 is insufficient when the scientific claim concerns changes in the lagged feature-to-outcome relationship.
