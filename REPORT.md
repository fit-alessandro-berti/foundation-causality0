# Synthetic causal-structure benchmark report

## CWFM revision addendum — 2026-07-30

The separate CWFM reference experiment has been redesigned and rerun. On its
locked 30-seed supported/identified effect bank, pooled MAE improved from 0.271
to 0.166 with 0.906 coverage for target-90% intervals. Strong regime-split
recovery improved from 0.067 to 0.633 while complete-null false positives
remained zero. The full architecture, training guardrails, calibration,
scenario tables, OOD failures, and reproducibility record are in
[`docs/foundation_model/experimental_results.md`](docs/foundation_model/experimental_results.md).

Evaluation date: 2026-07-28

## Executive summary

The six proposals in `docs/data` are now executable benchmarks. The recorded
run covered 88 scenarios with seeds 0–4: 440 independent
discovery/evaluation pairs, 1,760 persisted files, and 120.5 MiB of compressed
data. Every component completed, every seed produced metrics, and no component
approached the 20-minute limit.

The overall result is mixed in a useful way:

- Strong, clean single-regime and single-break signals were usually recovered
  with good localization and controlled null behavior.
- The causal-interference detector recovered strong spillovers and correctly
  abstained under poor exposure overlap, but rejected in 1/5 runs for each key
  no-spillover control. Hidden confounding produced large bias and zero
  interval coverage, as the identification-stress scenario was intended to
  reveal.
- Correlation, cross-loadings, weak edges, small samples, unequal group sizes,
  and high dimensionality mainly damaged structural recovery before they
  seriously damaged prediction.
- Both temporal pipelines were deliberately conservative. This controlled the
  complete-null and feature-process controls, but it completely missed the
  tested weak and multiple-break settings.
- Model specification matters sharply. The observed-regime detector returned
  no false split with the correct nonlinear basis and split every time when the
  same mechanism was fitted with a misspecified linear basis.
- The lagged VARX classifier mostly separated \(B\)-changes from other changes,
  but falsely attributed an intercept-only break to \(B\) in 2 of 5 runs.

These are engineering results, not publication-sized Monte Carlo estimates.
With only five seeds, a rate of 0 or 1 has an approximate 95% Wilson interval
of [0.00, 0.43] or [0.57, 1.00], respectively.

## What was run

Each generator saves observed arrays separately from evaluator-only truth:

```text
docs/data/generated/<method>/<scenario>/seed_NNNN/
├── discovery.npz
├── evaluation.npz
├── truth.npz
└── metadata.json
```

The corresponding metric directories contain per-seed CSV and JSONL, aggregate
CSV and JSON, a manifest, and run status. Aggregate files report mean, standard
deviation, median, and 5th/95th percentiles for every numeric metric.

The fixed evaluation settings were:

- five final-evaluation seeds per scenario;
- independent discovery and evaluation samples or trajectories;
- discovery-only preprocessing;
- 19 residual permutation/bootstrap repetitions for full-search calibration,
  giving a minimum Monte Carlo p-value of 0.05;
- five bootstrap repetitions for stability summaries;
- a 1% breakpoint-matching tolerance, with additional 0%, 2%, and 5% summaries
  for the contemporaneous temporal detector;
- a hard 1,200-second subprocess timeout per component.
- 29 cluster-bootstrap repetitions for causal-interference uncertainty and 29
  separate no-spillover null bootstraps that preserve direct effects and
  cluster shocks.

The six components completed as follows:

| Component | Scenarios × seeds | Latest runtime | Peak memory | Result |
| --- | ---: | ---: | ---: | --- |
| Observed regime / MOB-like tree | 11 × 5 | 164.6 s | 201.0 MiB | 55/55 |
| Causal spillover / interference | 21 × 5 | 402.9 s | 220.4 MiB | 105/105 |
| Sparse multi-response latent groups | 12 × 5 | 26.4 s | 204.1 MiB | 60/60 |
| Three-level latent graph | 15 × 5 | 66.1 s | 211.5 MiB | 75/75 |
| Contemporaneous temporal splits | 13 × 5 | 81.9 s | 190.4 MiB | 65/65 |
| Lagged VARX temporal splits | 16 × 5 | 109.7 s | 192.6 MiB | 80/80 |

No run failed or timed out. The original complete-run manifest is
`docs/data/metrics/run_manifest.json`; each component's `run_status.json`
records its latest corrected run.

## Observed regime determination

The detector scans the complete feature/threshold search, calibrates the root
with residual permutations, and only then grows a shallow regression tree.

| Scenario | Detection | Root accuracy | Regime ARI | Relative parameter error | Pooled / fitted / oracle loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean one split | 1.00 | 1.00 | 0.947 | 0.100 | 0.354 / 0.309 / 0.309 |
| Correlated proxy | 1.00 | 1.00 | 0.939 | 0.089 | 0.350 / 0.304 / 0.301 |
| Unbalanced regimes | 1.00 | 1.00 | 0.950 | 0.135 | 0.363 / 0.338 / 0.336 |
| Multiple splits | 1.00 | 1.00 | 0.951 | 0.109 | 0.406 / 0.313 / 0.309 |
| Only some outcomes change | 0.40 | 0.40 | 0.363 | 0.131 | 0.359 / 0.348 / 0.344 |
| Weak change | 0.00 | 0.00 | 0.000 | 0.116 | 0.343 / 0.343 / 0.343 |

The complete null, main-effect distractor, variance-only change, and correctly
specified global nonlinearity all had zero retained splits. In contrast, the
misspecified nonlinear model split in 5/5 runs and reduced test loss from 0.950
to 0.303 even though the true regime count was zero. This is the clearest
demonstration that predictive improvement alone cannot establish a genuine
regime.

The strong scenarios recovered 92–99% of the oracle loss improvement. Power
collapsed when only some outcomes changed and was zero for the weak jump used
here. The result argues for an explicit power curve before applying the method
to a weak-signal empirical problem.

## Causal-interference and spillover detection

The interference benchmark uses 24 independent clusters of 20 rows, two-stage
randomized saturation or declared observational assignment, explicit observed
and oracle relation matrices, cluster-fold cross-fitting, and two distinct
cluster bootstraps:

- an ordinary cluster bootstrap for ASE uncertainty and dose-response bands;
- a no-spillover null bootstrap that retains own-treatment effects, covariate
  adjustment, and cluster-level outcome shocks.

The primary contrast is
\(\operatorname{ASE}(a=0;g_1=0.75,g_0=0.25)\). Exposure is recomputed from the
complete allocation and declared mapping. Unsupported targets count as no
detection rather than being silently discarded.

| Scenario | Target supported | Global power / FPR at \(a=0\) | Outcome F1 | ASE RMSE | ASE coverage | Functional detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete null | 1.00 | 0.20 FPR | 1.000 | 0.148 | 0.850 | 0.20 |
| Direct-effect-only null | 1.00 | 0.20 FPR | 1.000 | 0.148 | 0.850 | 0.20 |
| Shared-outcome-shock null | 1.00 | 0.20 FPR | 1.000 | 0.222 | 0.950 | 0.00 |
| Observed-homophily null | 1.00 | 0.20 FPR | 0.800 | 0.099 | 0.950 | 0.20 |
| Linear spillover | 1.00 | 0.80 power | 0.553 | 0.149 | 0.800 | 1.00 |
| Threshold response | 1.00 | 1.00 power | 0.931 | 0.148 | 0.850 | 1.00 |
| Sparse affected outcomes | 1.00 | 0.40 power | 0.667 | 0.150 | 0.850 | 1.00 |
| Mixed continuous/binary/count | 1.00 | 0.80 power | 0.300 | 0.112 | 0.867 | 0.80 |
| Degree heterogeneity | 1.00 | 0.80 power | 0.460 | 0.123 | 0.950 | 1.00 |
| Poor exposure overlap | 0.00 | 0.00 end-to-end power | 0.000 | 0.092 | 0.900 | 0.00 |

The four primary no-spillover controls each rejected once in five runs. That
0.20 point estimate is above the nominal 0.05 target, although five seeds give
it a very wide interval. Outcome-specific family-wise false detection was zero
for the complete, direct-only, and shared-shock controls and 0.20 for observed
homophily. This benchmark therefore does not establish nominal calibration; a
larger run with at least 199 null-bootstrap draws and hundreds of final seeds is
needed.

For linear spillover, the true outcome effects were
\([0.400,-0.325,0.250,0]\), while the mean estimates were
\([0.358,-0.354,0.227,-0.030]\). Global power was 0.80 at \(a=0\) and 1.00 at
\(a=1\), but affected-outcome recall was only 0.467. The flexible cross-fitted
model's ASE RMSE was 0.149; the simple \(A,G,A\times G\) randomized-design
baseline obtained 0.102. The simpler baseline is better for this correctly
linear clean case, while the flexible model is substantially better for the
threshold response (0.146 versus 0.229 RMSE).

The weak-effect grid behaved monotonically:

| Spillover coefficient magnitude | Global power \(a=0\) | Global power \(a=1\) | Outcome F1 |
| ---: | ---: | ---: | ---: |
| 0.15 | 0.40 | 0.60 | 0.213 |
| 0.30 | 0.60 | 0.60 | 0.251 |
| 0.45 | 0.80 | 0.80 | 0.465 |

The interaction experiment separated own-treatment levels: power was 0.20 at
\(a=0\), where the spillover was deliberately weak, and 1.00 at \(a=1\). The
cancellation experiment achieved power 1.00 and 0.60 at the two levels instead
of averaging opposite effects into one null. For the nonmonotone
zero-selected-contrast case, the selected ASE correctly had zero global
rejections at \(a=0\), while the functional dose-response test detected
nonconstancy in 4/5 runs. This validates the distinction between contrast and
functional interference.

The assumption-stress scenarios were appropriately damaging:

| Stress scenario | Global power | ASE RMSE | ASE coverage | Mapping correlation | Key diagnostic |
| --- | ---: | ---: | ---: | ---: | --- |
| Hidden confounding | 1.00 | 0.726 | 0.000 | 1.000 | Detection does not repair identification bias |
| Misspecified exposure | 0.60 | 0.220 | 0.650 | 0.791 | Distance-two sensitivity RMSE 0.185 versus 0.221 for the primary unweighted map |
| Noisy relation matrix | 0.60 | 0.129 | 0.950 | 0.924 | 17.9% observed/true edge disagreement |
| Interference outside observed \(W\) | 0.80 | 0.153 | 0.950 | 0.960 | 11.5% omitted-edge disagreement |

Hidden confounding is the sharpest warning: the detector had high apparent
power while average bias was 0.716 and interval coverage was zero. It should be
read as an identification failure, not estimator noise.

The policy-shift scenario selected the correct low- versus high-saturation
policy in all five runs, with policy-effect RMSE 0.090 and zero observed policy
regret. Predictive gain from adding exposure was generally small (0.011
standardized-loss units for the clean linear case), reinforcing that causal
spillover testing and ordinary predictive utility are different targets.

## Outcome-relevant latent groups

The implementation uses nested three-fold validation with a one-standard-error
selection rule, supervised sparse sequential components, hard assignment only
for reporting, and conditional bootstrap stability.

| Scenario | Exact \(K\) | Support F1 | Group ARI | Score correlation | Test / oracle loss | Support stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean balanced | 1.00 | 0.981 | 0.923 | 0.960 | 0.303 / 0.261 | 0.806 |
| Correlated latents | 0.60 | 0.938 | 0.487 | 0.593 | 0.381 / 0.313 | 0.816 |
| Cross-loadings | 1.00 | 0.903 | 0.566 | 0.866 | 0.318 / 0.265 | 0.806 |
| High-dimensional | 1.00 | 0.707 | 0.387 | 0.924 | 0.295 / 0.265 | 0.495 |
| Unequal groups | 1.00 | 0.778 | 0.520 | 0.950 | 0.314 / 0.259 | 0.652 |
| Weak measurement | 1.00 | 0.984 | 0.724 | 0.709 | 0.466 / 0.261 | 0.905 |
| Weak outcome relevance | 0.60 | 0.954 | 0.647 | 0.857 | 0.830 / 0.805 | 0.837 |

The complete null selected zero components in all five runs and had honest test
loss 1.000. The outcome-irrelevant block was correctly omitted: exact relevant
\(K\) recovery was 1.00 and support F1 was 0.950. Scaling one outcome by 25
produced the same metrics as the clean scenario, confirming that training-only
outcome standardization removed raw-scale dominance.

Support recovery can be misleadingly reassuring. Under weak measurement,
support F1 remained 0.984 while score correlation fell to 0.709 and test loss
rose to 0.466. In the high-dimensional case, prediction stayed close to oracle
while group ARI and stability fell to 0.387 and 0.495. Structural and predictive
metrics should therefore remain side by side.

## Relationships between latent groups

The graph benchmark separates:

- Level A: true latent scores;
- Level B: true groups with independently reconstructed scores;
- Level C: estimated groups, independently reconstructed scores, and a
  held-out-likelihood-tuned graphical lasso.

| Scenario | Level A F1 | Level B F1 | Level C F1 | Level C AUPRC | Level C partial-correlation RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chain | 0.982 | 0.982 | 0.982 | 1.000 | 0.037 |
| Cross-loadings | 1.000 | 0.827 | 0.829 | 0.982 | 0.063 |
| Weak measurement | 0.982 | 0.956 | 0.820 | 0.980 | 0.094 |
| Weak edges | 0.701 | 0.594 | 0.575 | 0.907 | 0.042 |
| Small sample | 0.754 | 0.708 | 0.722 | 0.816 | 0.105 |
| Ill-conditioned covariance | 1.000 | 0.982 | 0.789 | 0.879 | 0.132 |
| Dense near limit | 0.970 | 0.957 | 0.946 | 0.998 | 0.064 |

The level differences locate the failure source. Cross-loadings caused most of
their damage between Levels A and B, while grouping error and weak measurement
caused an additional Level B-to-C loss. The empty graph was exact in Levels A
and B; Level C produced a false edge in 1/5 runs, for an exact graph rate of
0.80 and 0.20 mean false edges.

The temporal latent extension recovered its directed lag support with precision
0.914, recall 0.927, and F1 0.919. The hidden-confounding data are persisted and
stress the undirected pipeline, but no FCI/RFCI implementation was available;
PAG endpoint metrics are consequently outside this recorded primary evaluation.
The primary conclusion remains recovery of conditional associations, not
causal directions.

## Contemporaneous temporal splits

The detector uses a full segment-cost scan, residual full-search calibration,
recursive segmentation, and a separate adjacent-segment coefficient test.
“Localized recall” below requires one-to-one matching within 1% of the
trajectory length.

| Scenario | Any raw break | Localized raw recall | Any slope break | Median/mean localization | Pooled / fitted / oracle loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| One coefficient break | 1.00 | 1.00 | 1.00 | 1.0 rows | 0.319 / 0.257 / 0.257 |
| Sparse coefficient break | 1.00 | 0.80 | 1.00 | 0.0 rows when matched | 0.313 / 0.261 / 0.259 |
| Strong dependence | 1.00 | 0.80 | 1.00 | 1.25 rows when matched | 0.384 / 0.326 / 0.324 |
| Short segment | 0.80 | 0.80 | 0.80 | 5.0 rows when matched | 0.259 / 0.245 / 0.240 |
| Multiple coefficient breaks | 0.00 | 0.00 | 0.00 | missing | 0.320 / 0.320 / 0.276 |
| Weak coefficient break | 0.00 | 0.00 | 0.00 | missing | 0.294 / 0.294 / 0.293 |

The complete null, feature-distribution shift, and correctly specified global
nonlinearity all had zero detections. Intercept-only breaks were found in 5/5
runs and never retained as slope changes. Variance-only changes were not
detected by this conditional-mean cost and were never falsely labeled as slope
changes.

Gradual drift produced a raw and slope detection in every run, but only 1/5 was
within 1% of the designated midpoint. That is why “any detection” is 1.00 while
point-localized recall is 0.20; transition-window scoring is more appropriate
for this scenario.

Bootstrap recovery frequency was 1.00 for the clean, sparse, and intercept
breaks and 0.96 under strong dependence. The clean breakpoint confidence
interval covered the true location in 4/5 runs. The conservative global gate
was stable but too insensitive to the multiple and weak cases.

## Lagged VARX temporal splits

The lagged detector first finds generic changes in
\(Y_t\sim(Y_{t-1},X_{t-1})\), then compares a shared-\(B\) restricted model with
separate-\(B\) adjacent models.

| Scenario | Any raw break | Any retained \(B\)-break | \(B\)-break localization | Pooled / fitted / oracle loss |
| --- | ---: | ---: | ---: | ---: |
| One \(B\)-break | 1.00 | 1.00 | 2.0 rows | 0.516 / 0.423 / 0.422 |
| Mixed \(A+B\) | 1.00 | 1.00 | 1.0 row | 0.578 / 0.423 / 0.423 |
| High persistence | 1.00 | 1.00 | 2.2 rows | 0.300 / 0.258 / 0.258 |
| Zero to nonzero \(B\) | 0.60 | 0.60 | 4.7 rows | 0.742 / 0.686 / 0.678 |
| Nonzero to zero \(B\) | 0.40 | 0.40 | 1.0 row | 0.745 / 0.701 / 0.691 |
| Several \(B\)-breaks | 0.00 | 0.00 | missing | 0.554 / 0.554 / 0.475 |
| Weak \(B\)-break | 0.00 | 0.00 | missing | 0.566 / 0.566 / 0.559 |
| High-dimensional | 0.00 | 0.00 | missing | 0.362 / 0.362 / 0.326 |
| Close breakpoints | 0.00 | 0.00 | missing | 0.586 / 0.586 / 0.561 |

For retained changes, mean relative jump-\(B\) error was 0.230 for the clean
case, 0.202 for mixed \(A+B\), and 0.272 under high persistence. The clean
retained break reappeared in 92% of bootstrap fits; mixed and high-persistence
breaks reappeared in every bootstrap fit.

The complete null, \(X\)-process break, noise-only break, wrong-lag-order
stationary process, and contemporaneous-only relationship produced no raw or
\(B\)-specific break. The contemporaneous control had no segment-level
lag-relevance calls, but its thresholded coefficient-support F1 was 0 against
the zero direct-lag truth. Persistence can therefore create apparent lag
coefficients even when the select-and-test stage declines to call them
relevant.

The \(A\)-only change produced a raw break in 4/5 runs but no retained
\(B\)-break in the fitted samples. The intercept-only change produced a raw
break in every run and a false \(B\)-attribution in 2/5. Its bootstrap
false-\(B\) frequency was 0.80, so intercept/\(B\) separation is the clearest
specificity problem in this implementation.

## Overall interpretation

Five conclusions are consistent across the six benchmarks:

1. Null calibration remains a central tradeoff. The temporal pipelines were
   conservative enough to lose multiple and weak changes, while the
   interference pipeline rejected in 1/5 runs for every primary null control.
2. Prediction is easier than structural recovery. High-dimensional latent
   grouping and weak-edge graphs can predict well while recovering unstable or
   incorrect structures.
3. Measurement and grouping error propagate downstream. The three graph levels
   make this visible and should remain part of any future evaluation.
4. Scientific wording must follow the estimand. The first temporal benchmark
   detects changes in \(P(Y_t\mid X_t)\); VARX detects Granger-predictive
   changes; the latent graph detects conditional associations. None alone
   establishes intervention-level causality.
5. Causal-interference claims are especially assumption-sensitive. Correct
   mapping and overlap produced useful estimates, but hidden confounding
   retained apparent power while destroying bias and coverage.

## Limitations and next evaluation

- Five seeds are sufficient to exercise and compare the pipelines, but not to
  estimate 5% error rates. A serious study should run at least 200 seeds.
- Nineteen calibration repetitions give only the coarsest possible 0.05
  Monte Carlo decision. Use 199–999 repetitions for final inference.
- The interference null test used 29 bootstrap draws, so its p-values have
  1/30 resolution. Its 0.20 null rejection estimates and subnominal interval
  coverage require a much larger calibration and final-seed study.
- Sparse-PLS and graph stability are conditional on selected hyperparameters;
  they do not retune the full nested search in every bootstrap.
- The temporal implementation is a calibrated recursive scan rather than a
  full PELT penalty study. Its complete failure on the tested multiple-break
  cases should be addressed before deployment.
- The optional treatment-effect benchmark and PAG/FCI endpoint scoring were
  not part of this primary recorded run.
- Gradual drift should be summarized against its stored transition interval,
  not only against a nominal midpoint.

The most valuable next run is therefore not a blind increase in seeds. First
retune temporal penalties on independent calibration seeds to recover
multiple-break power, and independently calibrate the interference null
bootstrap while constraining all four negative controls. Then rerun 200+ final
seeds with 199+ bootstrap repetitions.
