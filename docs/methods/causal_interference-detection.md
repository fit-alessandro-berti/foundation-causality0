Given a meaningful row-to-row relation (W), a treatment-like feature (A_i), and confirmation that (A) precedes the outcomes, the central quantity should be a **causal spillover effect**, not a correlation coefficient.

Define each row’s exposure to other rows as

[
G_i=g(A_{-i},W_i),
]

for example,

[
G_i=\frac{\sum_{j\neq i}W_{ij}A_j}
{\sum_{j\neq i}W_{ij}},
]

and define

[
\mu_m(a,g)
==========

\mathbb E!\left[Y_i^{(m)}(a,g)\right]
]

as the average potential value of outcome (m) when the row’s own treatment is (a) and its exposure to other rows is (g). Exposure-specific potential outcomes and contrasts between them are the standard basis for causal inference under general interference. ([PubMed Central (PMC)][1])

## 1. Primary metric: Average Spillover Effect

The most directly useful metric is

[
\boxed{
\operatorname{ASE}_m(a;g_1,g_0)
===============================

\mu_m(a,g_1)-\mu_m(a,g_0)
}
]

where (g_1) and (g_0) are two different levels of exposure to other rows.

It asks:

> Holding row (i)'s own treatment fixed, how much would outcome (m) change if the treatments of related rows changed its exposure from (g_0) to (g_1)?

This is the cleanest metric for determining whether interference exists.

For example, if (G_i) is the proportion of treated neighbors,

[
\operatorname{ASE}_m(0;0.75,0.25)
]

compares the outcome of an untreated row when 75% rather than 25% of its related rows are treated.

The null hypothesis is

[
H_{0m}:
\operatorname{ASE}_m(a;g_1,g_0)=0.
]

Use exposure levels that have good empirical support. Comparing the 25th and 75th percentiles of (G_i) is often safer than comparing (0) with (1), because the extreme exposure levels may contain very few rows.

For each outcome, report:

[
\widehat{\operatorname{ASE}}_m,
\qquad
95%,\text{CI},
\qquad
p_m.
]

The confidence interval and magnitude are more informative than the (p)-value alone.

## 2. Standardized Spillover Effect

When the (M) outcomes have different units, report

[
\boxed{
\operatorname{SASE}_m
=====================

\frac{\operatorname{ASE}_m}
{\operatorname{SD}(Y^{(m)})}
}
]

for continuous outcomes.

This makes the effects approximately comparable across outcomes. For instance,

[
\operatorname{SASE}_m=0.30
]

means that changing the exposure from (g_0) to (g_1) changes the outcome by approximately (0.30) outcome standard deviations.

The preferred scale depends on the outcome:

| Outcome type  | Primary effect scale                                   | Useful secondary scale              |
| ------------- | ------------------------------------------------------ | ----------------------------------- |
| Continuous    | Mean difference                                        | Standardized mean difference        |
| Binary        | Risk difference                                        | Risk ratio                          |
| Count         | Rate difference                                        | Rate ratio                          |
| Ordinal       | Difference in expected score or cumulative probability | Odds ratio, with caution            |
| Time-to-event | Restricted mean survival-time difference               | Hazard ratio as a secondary measure |

For binary outcomes, the risk difference is especially interpretable:

[
\operatorname{ASE}^{RD}_m
=========================

## P(Y^{(m)}=1\mid a,g_1)

P(Y^{(m)}=1\mid a,g_0).
]

## 3. Spillover dose–response curve

A single contrast may miss nonlinear interference. Therefore, a highly useful secondary metric is the entire curve

[
\boxed{
g\longmapsto\mu_m(a,g)
}
]

for (a=0) and (a=1).

This reveals whether interference is:

* approximately linear;
* present only after a threshold;
* saturating;
* positive at low exposure but negative at high exposure;
* different for treated and untreated rows.

The literature explicitly defines average dose–response functions over own treatment and neighborhood exposure, from which treatment and spillover effects can be calculated. 

For continuous (G_i), a local spillover slope can be reported:

[
\operatorname{LSE}_m(a,g)
=========================

\frac{\partial \mu_m(a,g)}{\partial g}.
]

However, the slope should not replace the dose–response plot when nonlinearity is plausible.

## 4. Direct effect conditional on interference exposure

To separate own-row effects from cross-row effects, estimate

[
\boxed{
\operatorname{ADE}_m(g)
=======================

\mu_m(1,g)-\mu_m(0,g)
}
]

This asks how changing row (i)'s own treatment affects its outcome while holding the exposure from other rows fixed.

This is important because omitting (G_i) can cause the direct effect of (A_i) to absorb part of the spillover effect.

The pair

[
\left(
\operatorname{ADE}_m(g),
\operatorname{ASE}_m(a;g_1,g_0)
\right)
]

is considerably more informative than reporting only one treatment coefficient.

## 5. Direct–spillover interaction

The effect of other rows may depend on whether the focal row itself is treated. Measure this using

[
\boxed{
\operatorname{INT}_m
====================

## \operatorname{ASE}_m(1;g_1,g_0)

\operatorname{ASE}_m(0;g_1,g_0)
}
]

or, equivalently,

[
\begin{aligned}
\operatorname{INT}_m
={}&
[\mu_m(1,g_1)-\mu_m(1,g_0)]\
&-
[\mu_m(0,g_1)-\mu_m(0,g_0)].
\end{aligned}
]

Interpretation:

* (\operatorname{INT}_m=0): spillover is approximately the same for treated and untreated rows;
* (\operatorname{INT}_m>0): own treatment amplifies the spillover;
* (\operatorname{INT}_m<0): own treatment attenuates or reverses the spillover.

This metric is important because a single average spillover effect can be near zero when positive interference in one treatment group cancels negative interference in the other.

## 6. Overall or policy effect

When the goal is to understand the effect on the complete system rather than merely detect interference, compare two treatment-allocation policies:

[
\boxed{
\operatorname{OE}_m(\pi_1,\pi_0)
================================

## \mathbb E_{\pi_1}[Y^{(m)}]

\mathbb E_{\pi_0}[Y^{(m)}]
}
]

For example:

* (\pi_0): treat approximately 20% of rows;
* (\pi_1): treat approximately 60% of rows.

The overall effect includes changes produced by both own treatment and exposure to the treatments of related rows. Direct, indirect/spillover, total, and overall effects were introduced precisely to distinguish these different causal questions under interference. ([PubMed Central (PMC)][1])

This is often the most policy-relevant metric, but it is not the best primary metric for detecting interference, because a nonzero overall effect can be generated entirely by direct effects.

## 7. Joint interference metric for (M) outcomes

Testing each of the (M) outcomes separately creates a multiple-testing problem. To answer the global question

> Is there evidence of interference in at least one outcome?

collect the estimated spillover effects into

[
\widehat{\boldsymbol\delta}
===========================

\begin{bmatrix}
\widehat{\operatorname{ASE}}_1\
\vdots\
\widehat{\operatorname{ASE}}_M
\end{bmatrix}
]

and use the omnibus statistic

[
\boxed{
Q
=

\widehat{\boldsymbol\delta}^{\mathsf T}
\widehat{\Sigma}^{-1}
\widehat{\boldsymbol\delta}
}
]

where (\widehat{\Sigma}) is the estimated covariance matrix of the (M) effects.

Test

[
H_0:
\boldsymbol\delta=\mathbf 0.
]

This is preferable to declaring interference merely because one of many uncorrected (p)-values is below (0.05).

A practical reporting strategy is:

1. use (Q) or a maximum-(t) test as the global interference test;
2. only then inspect outcome-specific effects;
3. adjust the outcome-specific (p)-values using false-discovery-rate control.

When treatment was randomized and the assignment mechanism is known, a randomization or permutation test is preferable. Exact randomization-based tests can be constructed for hypotheses such as “immediate neighbors have no effect” or “units more than two links away have no effect.” ([National Bureau of Economic Research][2])

## 8. Overlap and effective sample size

An apparently precise spillover estimate is unreliable when few rows experience comparable values of (G_i).

Report the exposure distribution separately by own-treatment status:

[
G_i\mid A_i=0,
\qquad
G_i\mid A_i=1.
]

When weighting is used, calculate

[
\boxed{
n_{\mathrm{eff}}
================

\frac{\left(\sum_i w_i\right)^2}
{\sum_i w_i^2}
}
]

where (w_i) are the causal weights.

Important diagnostics are:

* number of rows at each exposure level;
* effective sample size;
* maximum and upper-percentile weight;
* proportion of rows outside common support;
* covariate balance after weighting or matching.

A small (p)-value accompanied by poor overlap and very low effective sample size should not be considered strong evidence.

## 9. Sensitivity to the assumed interference relation

The relation matrix (W) and exposure function (g(\cdot)) are assumptions. Therefore, estimate the ASE under several prespecified plausible mappings, for example:

[
G_i^{(1)}
=========

\text{fraction of directly related rows treated},
]

[
G_i^{(2)}
=========

\text{weighted fraction using relation strength},
]

[
G_i^{(3)}
=========

\text{exposure including relations up to distance two}.
]

Report:

[
\left[
\min_r \widehat{\operatorname{ASE}}_m^{(r)},
;
\max_r \widehat{\operatorname{ASE}}_m^{(r)}
\right]
]

together with whether the sign and approximate magnitude are stable.

Exposure mapping must adequately summarize the interference mechanism; causal conclusions can be sensitive to its misspecification. 

This sensitivity analysis is particularly important for observational data because the data generally cannot uniquely determine the correct interference structure.

## 10. Metrics that are only supplementary

The following can be useful diagnostics, but should not be treated as primary evidence of causal interference:

* correlation between (G_i) and (Y_i);
* improvement in (R^2) after adding (G_i);
* likelihood-ratio or information-criterion improvement;
* residual network or cluster autocorrelation;
* correlation between outcomes of related rows.

Related rows can have correlated outcomes because of shared unobserved causes without treatments spilling over between them. Outcome dependence and treatment interference are conceptually different phenomena. 

A partial (R^2),

[
R^2_{\text{partial},G}
======================

\frac{R^2_{\text{full}}-R^2_{\text{reduced}}}
{1-R^2_{\text{reduced}}},
]

may quantify how much predictive information (G_i) adds, but it does not establish a causal effect.

## Recommended minimal set

For a simple but defensible analysis, I would use these five outputs:

[
\boxed{
\begin{array}{ll}
1.& \widehat{\operatorname{ASE}}*m(a;g*{75},g_{25})
\text{ for every outcome};[2mm]
2.& 95%\text{ confidence intervals and standardized ASEs};[2mm]
3.& \text{one joint test }H_0:
\operatorname{ASE}_1=\cdots=\operatorname{ASE}_M=0;[2mm]
4.& \text{spillover dose–response curves }\mu_m(a,g);[2mm]
5.& \text{overlap/ESS and exposure-mapping sensitivity diagnostics}.
\end{array}
}
]

Add the direct effect and direct–spillover interaction when the treatment of the focal row may modify the interference effect.

Thus, the **best single metric** is the average spillover effect. The **best inferential summary** is its estimate with a dependence-aware 95% confidence interval. For (M) outcomes, the best global decision metric is an omnibus test of the vector of spillover effects, accompanied by overlap and interference-map sensitivity analyses.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2600548/?utm_source=chatgpt.com "Toward Causal Inference With Interference - PMC - NIH"
[2]: https://www.nber.org/system/files/working_papers/w21313/w21313.pdf?utm_source=chatgpt.com "Exact P-values for Network Interference"
