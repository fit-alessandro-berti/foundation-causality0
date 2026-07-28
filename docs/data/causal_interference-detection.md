# Data generation and testing for causal-interference detection

Corresponding method: [`../methods/causal_interference-detection.md`](../methods/causal_interference-detection.md)

## Detection target

The method targets a **causal spillover effect**: whether changing the treatment of rows related to row \(i\) changes row \(i\)'s outcome while its own treatment is held fixed.

Let there be \(S\) rows, \(N\) observed pre-treatment features, and \(M\) outcomes. For row \(i\), define

\[
X_i\in\mathbb R^N,
\qquad
A_i\in\{0,1\},
\qquad
Y_i=(Y_i^{(1)},\ldots,Y_i^{(M)}),
\]

and let \(W\in\mathbb R_+^{S\times S}\) be the known row-to-row relation matrix, with \(W_{ii}=0\). The true exposure is

\[
G_i=g(A_{-i},W_i),
\]

with the weighted treated-neighbor fraction as the default:

\[
G_i=
\frac{\sum_{j\ne i}W_{ij}A_j}
     {\sum_{j\ne i}W_{ij}}.
\]

The primary target for outcome \(m\) is

\[
\operatorname{ASE}_m(a;g_1,g_0)
=
\mu_m(a,g_1)-\mu_m(a,g_0),
\qquad
\mu_m(a,g)=\mathbb E[Y_i^{(m)}(a,g)].
\]

The benchmark must distinguish spillover from:

- a direct effect of \(A_i\) on \(Y_i\);
- correlated outcomes among related rows;
- homophilous or clustered treatment assignment;
- shared observed or unobserved causes;
- predictive improvement obtained by adding \(G_i\);
- misspecification of \(W\) or of the exposure mapping \(g(\cdot)\).

A nonzero coefficient or correlation between observed \(G_i\) and \(Y_i\) is not, by itself, a successful causal-interference detection.

## Identification conditions represented by the benchmark

The clean benchmark should encode the assumptions under which the causal quantities are intended to be identified:

1. \(W\), \(X\), and the assignment strata are determined before \(A\), and \(A\) is determined before \(Y\).
2. The potential outcome \(Y_i(a,g)\) depends on other rows' treatments only through the declared exposure mapping \(g(A_{-i},W_i)\).
3. The assignment is randomized, or all common causes of the joint treatment/exposure and the outcome are observed and adjusted for.
4. The target combinations \((a,g_0)\) and \((a,g_1)\) have adequate support.
5. There is no relevant interference through relations omitted from \(W\).

Violation scenarios should be included, but reported as assumption-stress tests rather than silently mixed with the correctly identified benchmark.

## Recommended output contract for the generator

Each generated instance should return:

```text
X                         S x N pre-treatment feature matrix
W_true                    true sparse relation matrix, with zero diagonal
W_observed                relation matrix supplied to the estimator
A                         length-S treatment vector
G_true                    exposure computed with W_true and the true mapping
G_candidates              exposures under all prespecified sensitivity mappings
Y                         S x M observed outcome matrix
cluster_id                independent interference cluster, when applicable
degree                    weighted and unweighted degree of every row
assignment_design         Bernoulli, randomized saturation, or observational
assignment_metadata       probabilities, strata, and a sampler for valid reassignments
true_exposure_mapping     complete definition of g(A_-i, W_i)
true_mean_function        callable or parameters for E[Y_i(a,g) | X_i]
target_levels             a, g0, g1, and the supported exposure grid
true_estimands             mu, ASE, SASE, ADE, INT, and optional policy effects
affected_outcomes         truth for the selected ASE and for functional interference
outcome_types             continuous, binary, count, ordinal, or survival
observed_confounders      variables available to the estimator
hidden_confounders        truth used only in violation scenarios
scenario                  complete generator configuration
seed                      random seed
```

Keep the oracle functions, true effects, hidden confounders, and \(W_{\text{true}}\) inaccessible to the fitting code. The estimator should receive only the fields that would be observed in the intended setting.

## Default synthetic generator

### 1. Generate the relation matrix and pre-treatment features

Start with **partial interference**, because independent relation clusters make the assignment design and uncertainty calculation auditable.

Create \(C\) clusters with 15--30 rows each and no edges between clusters. Within each cluster, use either:

- a complete unweighted graph;
- a sparse Erdős--Rényi graph with controlled expected degree;
- or positive edge weights sampled independently and then fixed.

Set \(W_{ii}=0\), retain the original weights for reporting, and row-normalize only when calculating an exposure. Ensure that the default graph has no isolated rows.

Generate pre-treatment features from

\[
X_i=\Gamma H_{c(i)}+\nu_i,
\qquad
H_c\sim\mathcal N(0,I),
\qquad
\nu_i\sim\mathcal N(0,\Sigma_X),
\]

where \(H_c\) creates optional within-cluster similarity. Set \(\Gamma=0\) in the simplest randomized benchmark and increase it in homophily scenarios.

After the implementation works for independent clusters, add general-network tests using stochastic-block, random-geometric, or degree-heterogeneous graphs. Record connected components and communities so that graph-aware resampling can be evaluated.

### 2. Generate treatment using a known design

The recommended primary assignment is a two-stage randomized saturation design. For cluster \(c\), first draw

\[
P_c\in\{0.2,0.5,0.8\}
\]

with known probabilities, and then draw

\[
A_i\mid P_{c(i)}\sim\operatorname{Bernoulli}(P_{c(i)}).
\]

This creates useful variation in neighbor exposure while preserving a known assignment mechanism. Include \(P_c\) in the assignment metadata and in any nuisance model that conditions on the design.

Also test independent Bernoulli assignment,

\[
A_i\sim\operatorname{Bernoulli}(p),
\]

for several values of \(p\). With large degrees, this design can concentrate \(G_i\) tightly around \(p\); that is an overlap stress test rather than an estimator failure.

For an observational extension, generate

\[
\Pr(A_i=1\mid X_i,\bar X_i,H_{c(i)})
=
\operatorname{logit}^{-1}
\left(
\eta_0+\eta_X^{\mathsf T}X_i
+\eta_N^{\mathsf T}\bar X_i
+\eta_H^{\mathsf T}H_{c(i)}
\right),
\]

where \(\bar X_i\) summarizes pre-treatment features of related rows. In the identified observational scenario, expose every variable entering this probability to the estimator. In the hidden-confounding scenario, withhold \(H_c\).

Clip individual probabilities away from zero and one in the identified observational benchmark. Store the unclipped and clipped probabilities so that the strength of the positivity problem is known.

### 3. Define the true exposure mapping

Use the weighted treated-neighbor fraction as the default:

\[
G_i^{(1)}=
\frac{\sum_{j\ne i}W_{ij}A_j}
     {\sum_{j\ne i}W_{ij}}.
\]

Add controlled alternatives:

```text
G1  unweighted fraction of directly related rows treated
G2  relation-strength-weighted treated fraction
G3  weighted exposure through paths of length at most two
G4  threshold exposure 1(G1 >= c)
G5  maximum or top-k-neighbor treatment exposure
```

The true mapping must be fixed before outcomes are generated. Candidate sensitivity mappings should also be declared before fitting; choosing the mapping with the smallest p-value invalidates the nominal error rate.

For isolated rows in a robustness scenario, either exclude them from the spillover estimand or set \(G_i=0\) and include an isolate indicator. The choice must be made in the generator and reproduced by the estimator.

### 4. Generate potential-outcome mean functions

For outcome \(m\), use

\[
\eta_{im}(a,g)
=
\alpha_m
+X_i^{\mathsf T}\beta_m
+\tau_m a
+\delta_m f_m(g)
+\kappa_m a f_m(g)
+b_{c(i),m},
\]

where:

- \(\tau_m\) controls the direct effect;
- \(\delta_m\) controls spillover for untreated rows;
- \(\kappa_m\) controls the direct--spillover interaction;
- \(b_{c,m}\) is an optional shared outcome shock independent of randomized treatment.

Useful exposure-response functions are:

\[
\begin{aligned}
f_{\text{linear}}(g)&=g,\\
f_{\text{threshold}}(g)&=\mathbf 1(g>0.5),\\
f_{\text{saturating}}(g)&=1-e^{-4g},\\
f_{\text{nonmonotone}}(g)&=(g-0.5)^2.
\end{aligned}
\]

The last function is important because its values at \(g=0.25\) and \(g=0.75\) are equal. A detector based only on that contrast should correctly report a zero ASE even though a functional dose--response test can detect interference.

Generate outcomes through an outcome-specific inverse link \(h_m^{-1}\):

\[
\mathbb E[Y_i^{(m)}(a,g)\mid X_i,b_{c(i)}]
=h_m^{-1}(\eta_{im}(a,g)).
\]

Use:

```text
continuous    identity link with multivariate Gaussian residuals
binary        logistic link and Bernoulli sampling
count         log link and Poisson or negative-binomial sampling
ordinal       cumulative-link probabilities
survival      a parametric survival model with oracle restricted mean
```

The default should use continuous outcomes with residual vector

\[
\varepsilon_i\sim\mathcal N(0,\Sigma_\varepsilon),
\]

so that the \(M\) outcomes can be correlated while rows remain conditionally independent inside the clean benchmark. Add shared cluster effects or graph-correlated residuals only in dedicated dependence scenarios.

Vary \(\delta_m\) over a grid that includes zero. Keep \(\tau_m\ne0\) in the primary no-interference null so that a method cannot pass merely by detecting any treatment effect.

### 5. Calculate oracle causal quantities

Create a large reference population or use a large Monte Carlo sample from the generator. For each supported \((a,g)\), calculate

\[
\mu_m(a,g)
=
\frac{1}{S_{\mathrm{ref}}}
\sum_{i=1}^{S_{\mathrm{ref}}}
 h_m^{-1}(\eta_{im}(a,g)).
\]

Then store

\[
\operatorname{ASE}_m(a;g_1,g_0)
=
\mu_m(a,g_1)-\mu_m(a,g_0),
\]

\[
\operatorname{ADE}_m(g)
=
\mu_m(1,g)-\mu_m(0,g),
\]

and

\[
\operatorname{INT}_m
=
\operatorname{ASE}_m(1;g_1,g_0)
-
\operatorname{ASE}_m(0;g_1,g_0).
\]

For continuous outcomes, define the true standardized spillover using the marginal outcome standard deviation under a declared reference assignment policy:

\[
\operatorname{SASE}_m
=
\frac{\operatorname{ASE}_m}{\sigma_{Y_m,\mathrm{ref}}}.
\]

For two assignment policies \(\pi_0\) and \(\pi_1\), estimate the oracle policy value by repeatedly drawing the complete treatment vector under each policy and recomputing every row's exposure:

\[
V_m(\pi)
=
\mathbb E_{A\sim\pi}
\left[
\frac1S\sum_i
\mu_{im}\left(A_i,g(A_{-i},W_i)\right)
\right],
\]

\[
\operatorname{OE}_m(\pi_1,\pi_0)=V_m(\pi_1)-V_m(\pi_0).
\]

Do not approximate policy effects by changing \(A_i\) independently while holding \(G_i\) fixed; the exposure must be recomputed from each complete allocation.

### 6. Separate calibration and evaluation simulations

Use independent seed sets for:

- choosing exposure bins, spline complexity, trimming thresholds, and decision cutoffs;
- estimating null critical values or tuning an omnibus test;
- reporting final type-I error, power, effect error, and interval coverage.

Within every evaluation replicate, use graph-aware cross-fitting for nuisance estimation. Across replicates, generate independent graphs or independent interference clusters. A single favorable network realization is not sufficient evidence of reliable detection.

## Required benchmark scenarios

| Scenario | Data-generating truth | What a good pipeline should do |
| --- | --- | --- |
| Direct-effect null | \(\tau_m\ne0\), but \(\delta_m=\kappa_m=0\) | Detect no spillover at the nominal error rate |
| Complete null | No direct or spillover effect | Calibrate the global and outcome-specific tests |
| Linear spillover | \(f(g)=g\), with positive or negative \(\delta_m\) | Recover ASE magnitude, sign, and interval coverage |
| Weak-effect grid | Increasing \(|\delta_m|\) from zero | Produce a calibrated power curve |
| Direct--spillover interaction | \(\kappa_m\ne0\) | Distinguish ASE at \(a=0\) from ASE at \(a=1\) |
| Cancellation | Spillover is positive for one own-treatment level and negative for the other | Avoid averaging the two effects into a false null |
| Threshold or saturation | Nonlinear \(f(g)\) | Recover the dose--response shape, not only one slope |
| Zero selected contrast | Nonconstant \(f(g)=(g-0.5)^2\), but ASE at 0.25 versus 0.75 is zero | Report a null target contrast while the functional test detects nonconstancy |
| Sparse multivariate effects | Only a subset of the \(M\) outcomes has spillover | Control multiplicity and recover the affected outcomes |
| Mixed outcome types | Continuous, binary, and count outcomes | Use the correct effect scale for each outcome |
| Poor exposure overlap | High degree, extreme treatment prevalence, or imbalanced saturation | Flag unsupported targets and show reduced ESS or power |
| Degree heterogeneity | Hubs, low-degree rows, and optional isolates | Avoid domination by a few highly weighted rows |
| Shared outcome shock | Related rows share \(b_c\), but treatment is randomized and spillover is zero | Avoid confusing outcome dependence with interference |
| Observed homophily | \(X\) affects \(W\), \(A\), and \(Y\), and is fully observed | Recover calibration after correct adjustment |
| Hidden confounding | A withheld cluster variable affects \(A\) and \(Y\) | Reveal bias and failed coverage as an identification violation |
| Misspecified exposure | The true effect uses weighted or distance-two exposure, but the primary fit uses direct unweighted exposure | Quantify sensitivity and attenuation without claiming the wrong estimand |
| Noisy relation matrix | Edges are deleted, added, or reweighted in \(W_{\text{observed}}\) | Show how effect estimates degrade with relation error |
| Interference outside \(W\) | Outcomes also depend on omitted relations | Fail transparently as a mapping violation |
| Policy shift | Compare low- and high-saturation allocation policies | Recover the overall effect and policy ordering |

The direct-effect null, shared-outcome-shock null, and observed-homophily null are particularly important. They distinguish causal spillover detection from ordinary association between related rows.

## Testing pipeline

### 1. Predeclare the estimand and detection rule

Before fitting, declare:

- the primary own-treatment level \(a\), or that both \(a=0\) and \(a=1\) will be tested;
- the primary exposure levels \(g_0\) and \(g_1\);
- the exposure mapping and relation matrix used in the primary analysis;
- the supported exposure grid for the dose--response curve;
- the global significance level and multiplicity correction;
- a practical effect tolerance used only to label simulation truth.

For a data-adaptive supported contrast, choose \(g_0\) and \(g_1\) from the exposure distribution without looking at \(Y\), for example the 25th and 75th percentiles within the target \(A=a\) stratum. If both own-treatment levels or an ADE are estimated, use levels inside their common support.

Use the same exposure levels for every outcome. Selecting different levels for each outcome by maximizing significance produces an invalid multivariate test.

### 2. Validate the observed inputs

Check that:

- \(W\) has the intended direction, nonnegative weights, and a zero diagonal;
- row identifiers align across \(X\), \(A\), \(Y\), and \(W\);
- all adjustment variables precede treatment;
- no outcome-derived feature is included in \(X\);
- the declared isolate rule is applied consistently;
- treatment precedes every outcome being analyzed.

Print the number of rows, clusters or connected components, degree summaries, treatment prevalence, and exposure summaries before estimating effects.

### 3. Implement a simple discrete-exposure reference estimator

For an implementation check, discretize exposure into a small prespecified set \(H_i=b(G_i)\), such as low and high exposure, and define the joint condition

\[
E_i=(A_i,H_i).
\]

Let \(Z_i\) contain pre-treatment covariates, pre-treatment network summaries, and known assignment strata. For joint condition \(e=(a,h)\), estimate

\[
\pi_i(e\mid Z_i)=\Pr(E_i=e\mid Z_i)
\]

and

\[
m_m(e,Z_i)=\mathbb E[Y_i^{(m)}\mid E_i=e,Z_i].
\]

A cross-fitted augmented inverse-probability estimator is


\[
\widehat\mu_m(e)
=
\frac1S\sum_{i=1}^S
\left[
\widehat m_m(e,Z_i)
+
\frac{\mathbf 1(E_i=e)}{\widehat\pi_i(e\mid Z_i)}
\left(
Y_i^{(m)}-\widehat m_m(e,Z_i)
\right)
\right].
\]

Then calculate the ASE by differencing \(\widehat\mu_m(a,h_1)\) and \(\widehat\mu_m(a,h_0)\).

Under a known randomized design, obtain \(\pi_i(e)\) analytically when possible or by many valid treatment reallocations from the design, recomputing \(G\) and \(H\) after every draw. Do not fit an independent propensity model that ignores the joint assignment mechanism merely because \(A_i\) is binary.

### 4. Estimate the continuous dose--response

For continuous \(G\), fit a cross-fitted model

\[
\widehat m_m(a,g,Z_i)
\approx
\mathbb E[Y_i^{(m)}\mid A_i=a,G_i=g,Z_i]
\]

using a prespecified spline, generalized additive model, or another sufficiently regularized learner. Standardize over the target population:

\[
\widehat\mu_m(a,g)
=
\frac1S\sum_i\widehat m_m(a,g,Z_i).
\]

In randomized benchmarks, compare this outcome-regression estimate with a design-based or augmented estimator. In observational benchmarks, use a generalized propensity or another valid continuous-exposure adjustment method and cross-fit all nuisance functions.

Evaluate the curve only where exposure has adequate support. Extrapolated endpoints should be omitted or marked unsupported rather than presented as precise causal estimates.

### 5. Use graph-aware cross-fitting

When the data contain independent interference clusters, assign complete clusters to folds. Never place rows from the same interference cluster in both nuisance-training and validation folds.

For a general connected graph, partition the graph into blocks and optionally remove a buffer of immediate neighbors around each validation block. Record the fraction of rows lost to buffering. Ordinary random row folds can leak information through shared treatments, exposures, covariates, and outcome shocks.

Fit preprocessing, propensity models, outcome models, spline complexity, and trimming rules using training folds only.

### 6. Estimate uncertainty with the dependence structure intact

For partial interference, resample complete interference clusters and rerun the full estimator, including:

- preprocessing;
- nuisance-model fitting;
- exposure-level selection when data-adaptive;
- trimming;
- dose--response fitting;
- all contrasts and the multivariate covariance estimate.

For a known randomized design, also implement a design-compatible randomization test. Under the complete sharp null, the whole assignment can be redrawn. Under a no-spillover null that allows direct effects, use a conditional or focal-unit randomization procedure that preserves the focal rows' own treatment; unrestricted permutation of the complete treatment vector is not an exact test of that null.

For one connected network, a row bootstrap is invalid. Use a justified network block bootstrap, network-HAC method, or repeated independent networks in the simulation benchmark.

### 7. Test the multivariate null

Collect the primary spillover contrasts in

\[
\widehat{\boldsymbol\delta}
=
(\widehat{\operatorname{ASE}}_1,\ldots,
 \widehat{\operatorname{ASE}}_M)^{\mathsf T}
\]

and estimate their covariance from the same cluster-bootstrap or randomization replicates. Test

\[
H_0:\boldsymbol\delta=0
\]

using

\[
Q=
\widehat{\boldsymbol\delta}^{\mathsf T}
\widehat\Sigma^{-1}
\widehat{\boldsymbol\delta}.
\]

When \(M\) is large relative to the number of independent clusters, use a regularized covariance or a bootstrap maximum-\(|t|\) statistic instead of an unstable matrix inverse.

After the global test, report outcome-specific intervals and false-discovery-rate-adjusted p-values. If both \(a=0\) and \(a=1\) are primary, either predeclare one global test for the stacked \(2M\)-vector or correct the two omnibus tests.

### 8. Run exposure-mapping sensitivity analyses

For every prespecified candidate mapping:

1. recompute \(G_i\) from the complete treatment vector and that mapping;
2. rerun the complete estimation and uncertainty pipeline;
3. report the effect estimate, interval, support, and ESS;
4. summarize sign and magnitude stability.

Do not treat the range over mappings as a confidence interval. The mappings generally define different estimands. In synthetic experiments, compare each estimate with the oracle effect under that same mapping and separately show performance when the fitted mapping differs from the true outcome-generating mapping.

### 9. Compare with informative baselines

Include at least:

- the estimator using the true assignment probabilities and the correctly specified outcome model;
- the proposed cross-fitted estimator;
- outcome regression without propensity adjustment;
- a naive regression of \(Y\) on \(A\) and \(G\);
- a no-interference model that excludes \(G\);
- a predictive model using \(G\) only as an ordinary feature.

The last two baselines help quantify predictive gain, but they are not causal-interference detectors.

## Detection and scoring rules

### Target-contrast detection

For simulation scoring, define the affected outcome set

\[
\mathcal M_{\mathrm{ASE}}
=
\left\{
 m:
 \left|
 \operatorname{ASE}_m(a;g_1,g_0)
 \right|>\epsilon_{\mathrm{practical}}
\right\}.
\]

Declare outcome \(m\) detected when its multiplicity-adjusted test rejects the zero-ASE null. The global detector rejects when the omnibus p-value is below the predeclared level.

Use \(\epsilon_{\mathrm{practical}}=0\) for exact analytic continuous-outcome simulations. A small positive tolerance is useful when oracle effects are computed by Monte Carlo.

### Functional interference detection

A single contrast can be zero even when the dose--response is nonconstant. Define, over the supported region \(\mathcal G_a\),

\[
D_m(a)
=
\int_{\mathcal G_a}
\left[
\mu_m(a,g)-\bar\mu_m(a)
\right]^2
w_a(g)\,dg,
\]

where \(w_a(g)\) is a fixed support weight and \(\bar\mu_m(a)\) is the corresponding weighted mean.

Score a functional test of constant \(g\mapsto\mu_m(a,g)\) against truth defined by \(D_m(a)>\epsilon_D\). Do not count failure to reject one zero contrast as a false negative for functional interference, or count a functional rejection as proof that the chosen ASE contrast is nonzero.

### Unsupported targets

If either target exposure lacks support, the pipeline should abstain from the primary contrast and print a support failure. Report:

- the fraction of simulations in which the target is supported;
- conditional performance among supported simulations;
- end-to-end power counting abstention as no detection.

This prevents a method from appearing accurate by silently dropping difficult replicates.

## Metrics to print

### Global and outcome-specific detection

Across independent simulation seeds, print:

- global null false-positive rate with a binomial confidence interval;
- global power for each spillover magnitude;
- empirical rejection rates at levels 0.01, 0.05, and 0.10;
- outcome-level precision, recall, F1, and false discovery rate;
- macro- and micro-averaged outcome F1;
- family-wise probability of at least one false outcome detection;
- exact recovery rate of the affected-outcome set;
- power separately for \(a=0\) and \(a=1\), when both are tested.

Place null calibration beside every power result. High power is not useful when the procedure also rejects most no-spillover data sets.

### ASE estimation and inference

For replicate \(r\) and outcome \(m\), let

\[
e_{rm}
=
\widehat{\operatorname{ASE}}_{rm}
-
\operatorname{ASE}_{rm}.
\]

Print:

\[
\operatorname{bias}_m=\frac1R\sum_r e_{rm},
\qquad
\operatorname{RMSE}_m=
\sqrt{\frac1R\sum_r e_{rm}^2},
\]

and also:

- mean and median absolute error;
- RMSE after dividing by the reference outcome standard deviation;
- SASE bias and RMSE for continuous outcomes;
- effect-sign accuracy when the true effect is nonzero;
- 95% interval coverage and mean interval width;
- null interval coverage of zero;
- p-value calibration under the null;
- bias and coverage by degree, cluster size, and exposure-overlap level.

For binary, count, ordinal, and survival outcomes, score the primary effect scale declared in the method rather than comparing coefficients on an arbitrary link scale.

### Dose--response quality

On a fixed supported grid \(g_1^*,\ldots,g_L^*\), print

\[
\operatorname{ISE}_m(a)
=
\frac{\sum_{\ell=1}^L
w_\ell
\left[
\widehat\mu_m(a,g_\ell^*)-
\mu_m(a,g_\ell^*)
\right]^2}
{\sum_{\ell=1}^L w_\ell}.
\]

Also report:

- integrated absolute error;
- maximum absolute curve error;
- pointwise interval coverage;
- simultaneous confidence-band coverage;
- functional-test type-I error and power;
- local-slope error where a differentiable curve is fitted;
- error separately in low-, central-, and high-support regions.

Do not average unsupported grid points into these metrics.

### Direct effects, interactions, and policy effects

Print bias, RMSE, sign accuracy, and interval coverage for:

- \(\operatorname{ADE}_m(g)\);
- \(\operatorname{INT}_m\);
- \(\operatorname{OE}_m(\pi_1,\pi_0)\).

Also print:

```text
false_interference_direct_effect_only
interaction_detection_power
policy_ordering_accuracy
policy_value_RMSE
```

If the estimated effects are used to choose between policies, report policy regret on independently simulated assignment draws.

### Overlap and weighting diagnostics

For every target condition, print:

- raw number of observed rows;
- effective sample size
  \[
  n_{\mathrm{eff}}=
  \frac{(\sum_i w_i)^2}{\sum_i w_i^2};
  \]
- minimum, median, 95th percentile, and maximum weight;
- fraction trimmed or outside common support;
- exposure quantiles separately for \(A=0\) and \(A=1\);
- standardized covariate balance before and after weighting;
- supported-target rate across simulations.

Summarize ASE error and interval coverage by ESS quintile. A diagnostic is useful only if it predicts when the effect estimate becomes unreliable.

### Relation and exposure-mapping robustness

Print:

- ASE error under the correct mapping;
- attenuation or amplification under each misspecified mapping;
- sign-stability rate across plausible mappings;
- width of the sensitivity range;
- effect error as a function of deleted, added, or reweighted edges;
- performance by direct-neighbor versus distance-two truth;
- false detection rate when \(W\) captures outcome similarity but no treatment spillover.

Keep inferential uncertainty within a mapping separate from uncertainty about which mapping is scientifically correct.

### Negative-control specificity

The following rates should be prominent:

```text
false_interference_complete_null
false_interference_direct_only
false_interference_shared_outcome_shock
false_interference_observed_homophily
false_interference_wrong_exposure_map
```

For hidden confounding and omitted-relation scenarios, report bias and failed coverage explicitly. These scenarios diagnose the cost of assumption violations; they should not be relabeled as ordinary estimator noise.

### Stability and generalization

- bootstrap standard deviation of every ASE;
- pairwise correlation of effect estimates across repeated assignments on the same \(W\) and \(X\);
- variability of selected exposure levels when they are quantile based;
- curve variability across graph-aware folds;
- sensitivity to nuisance-model class and regularization;
- runtime and number of failed or singular fits.

For policy and dose--response prediction, use an independent network or an independent treatment re-randomization with newly sampled outcomes. Recompute exposures under the new allocation.

### Supplementary predictive diagnostics

The following may be printed, but must be labeled noncausal diagnostics:

- held-out RMSE, log-loss, or deviance with and without \(G\);
- partial \(R^2\) of \(G\);
- likelihood-ratio or information-criterion improvement;
- residual network autocorrelation;
- correlation between outcomes of related rows.

Good predictive performance does not substitute for calibrated ASE estimation under a valid assignment and exposure model.

## Minimal benchmark pseudocode

```python
config = tune_on_independent_simulations(
    calibration_seeds,
    objective="global spillover power",
    constraints={
        "global_null_FPR": 0.05,
        "direct_only_false_interference": 0.05,
    },
)

records = []
for scenario in scenarios:
    for seed in evaluation_seeds:
        observed, oracle_reference, truth = generate_interference_data(
            scenario=scenario,
            seed=seed,
        )

        primary_G = compute_exposure(
            A=observed.A,
            W=observed.W_observed,
            mapping=config.primary_mapping,
        )
        targets = choose_supported_targets(
            A=observed.A,
            G=primary_G,
            rule=config.target_rule,
            use_outcomes=False,
        )

        fitted = fit_cross_fitted_interference_estimator(
            X=observed.X,
            A=observed.A,
            G=primary_G,
            Y=observed.Y,
            assignment=observed.assignment_metadata,
            fold_groups=observed.cluster_id,
            targets=targets,
            config=config,
        )

        uncertainty = dependence_aware_inference(
            observed=observed,
            fitted=fitted,
            method=(
                "conditional_randomization"
                if observed.assignment_design.is_randomized
                else "cluster_bootstrap"
            ),
            refit_complete_pipeline=True,
        )

        omnibus = joint_spillover_test(
            ase=fitted.ase,
            covariance=uncertainty.ase_covariance,
            fallback="bootstrap_max_t",
        )

        sensitivity = {}
        for mapping in config.sensitivity_mappings:
            G_alt = compute_exposure(observed.A, observed.W_observed, mapping)
            sensitivity[mapping.name] = refit_for_mapping(
                observed=observed,
                G=G_alt,
                config=config,
            )

        records.append(
            evaluate_interference_detection(
                fitted=fitted,
                uncertainty=uncertainty,
                omnibus=omnibus,
                sensitivity=sensitivity,
                oracle_reference=oracle_reference,
                truth=truth,
                count_unsupported_as_no_detection=True,
            )
        )

print_metrics_by_scenario(records)
```

The tuning objective should never maximize power alone. It should maximize power or outcome-level F1 subject to calibrated global and direct-effect-only false-positive rates.

## Suggested compact report

```text
scenario=linear_spillover_randomized_saturation  seeds=500
S=1200  clusters=60  N=12  M=4  affected_outcomes=3
target=a0:g0=0.25:g1=0.75  supported_target_rate=0.984

global_null_FPR=0.046  global_power=0.912
outcome_precision=0.938  outcome_recall=0.887  outcome_F1=0.912
outcome_FDR=0.041  affected_set_exact_recovery=0.81

ASE_bias_mean=0.008  ASE_MAE=0.052  ASE_RMSE=0.074
SASE_RMSE=0.069  ASE_sign_accuracy=0.976
ASE_CI_coverage=0.944  ASE_CI_mean_width=0.287

dose_response_ISE=0.018  simultaneous_band_coverage=0.931
ADE_RMSE=0.061  INT_RMSE=0.082  policy_OE_RMSE=0.057
policy_ordering_accuracy=0.954

ESS_g0_median=322  ESS_g1_median=301  trim_rate=0.013
mapping_sign_stability=0.89  edge_deletion_10pct_ASE_RMSE=0.096

false_interference_direct_only=0.043
false_interference_shared_outcome_shock=0.049
false_interference_observed_homophily=0.052
```

The final interpretation should describe a detected effect as causal interference only under the declared assignment, overlap, consistency, and exposure-mapping assumptions. The synthetic benchmark can show that the implementation is calibrated when those assumptions hold and how it fails when they do not; it cannot make the assumptions true for an arbitrary observed table.
