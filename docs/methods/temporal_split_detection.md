## Recommended method: regression-based change-point detection

Let the table contain (T) temporally ordered rows,

[
X_t\in\mathbb{R}^N,\qquad Y_t\in\mathbb{R}^M,\qquad t=1,\ldots,T.
]

The simplest method aligned with your objective is:

> **Search for time points at which a model of the outcomes conditional on the features, (P(Y_t\mid X_t)), fits substantially better when estimated separately before and after the time point.**

This is preferable to applying ordinary change-point detection to the concatenated columns ([X,Y]), because the latter can report a split merely because the distribution of the features (P(X)) changed, even when the mechanism relating (X) to (Y) remained unchanged.

### 1. Define a regression cost for each temporal interval

For any interval of rows ([a,b]), fit a model

[
Y_t=f_{\theta_{a:b}}(X_t)+\varepsilon_t
]

and define

[
C(a,b)
======

\min_{\theta}
\sum_{t=a}^{b}
\ell!\left(Y_t,f_\theta(X_t)\right).
]

The simplest choices are:

* continuous outcomes: multivariate linear regression and squared error;
* binary outcomes: logistic regression and log-loss;
* count outcomes: Poisson regression and negative log-likelihood;
* high-dimensional (X): ridge regression rather than unregularized regression.

For continuous multivariate outcomes, standardize each outcome first and use

[
C(a,b)
======

\sum_{t=a}^{b}
\left|
Y_t-\widehat B_{a:b}^{\mathsf T}
\begin{bmatrix}
1\X_t
\end{bmatrix}
\right|_2^2.
]

Standardization prevents a high-variance outcome from dominating all the others.

## Detecting one split

For every admissible candidate split (\tau), compute

[
G(\tau)
=======

## C(1,T)

\left[
C(1,\tau)+C(\tau+1,T)
\right].
]

Then select

[
\widehat{\tau}
==============

\arg\max_\tau G(\tau).
]

A large (G(\tau)) means that two separate outcome models fit much better than one common model.

For a single continuous outcome and a linear model, this is essentially a scan over **Chow structural-break tests**, which compare whether the regression coefficients before and after a split are equal. ([JSTOR][1])

For (M=1), with (p=N+1) regression parameters, the statistic at a fixed candidate (\tau) can be written as

[
F(\tau)
=======

\frac{
\left[
RSS_0-RSS_{\mathrm{before}}(\tau)-RSS_{\mathrm{after}}(\tau)
\right]/p
}{
\left[
RSS_{\mathrm{before}}(\tau)+RSS_{\mathrm{after}}(\tau)
\right]/(T-2p)
}.
]

However, because you are **searching over many possible values of (\tau)**, you should not use the ordinary fixed-split (F)-test p-value at the selected split. Tests with an unknown change point have a different null distribution because the breakpoint exists only under the alternative hypothesis. ([JSTOR][2])

### Simplest reliable significance test

Use a bootstrap of the **whole scan**:

1. Fit one pooled model (Y=f(X)) over all rows.
2. Keep the observed (X_t) values and their order fixed.
3. Resample or simulate residuals under this pooled model.
4. Generate a bootstrap outcome table (Y_t^*).
5. Repeat the complete scan and record (\max_\tau G^*(\tau)).
6. Compare the observed (\max_\tau G(\tau)) with the bootstrap distribution.

Keeping (X) fixed is important: it allows (P(X)) to change over time under the null hypothesis while testing whether (P(Y\mid X)) changed.

For temporally dependent rows, resample residuals in blocks rather than individually. For multiple outcomes, resample the entire (M)-dimensional residual vector together so that correlations between outcomes are preserved.

## Detecting several splits

Use **PELT with a regression cost**:

[
\min_{K,\tau_1,\ldots,\tau_K}
\left{
\sum_{k=0}^{K}
C(\tau_k+1,\tau_{k+1})
+
\beta K
\right},
]

where

[
\tau_0=0,\qquad \tau_{K+1}=T,
]

and (\beta) penalizes every additional temporal segment.

PELT efficiently optimizes this penalized segmentation objective and, under its assumptions, returns the exact optimum for the chosen cost and penalty. ([arXiv][3])

A practical default is:

* segment cost: multivariate linear or ridge-regression loss;
* search method: PELT;
* penalty: BIC/MDL-type penalty;
* minimum segment length: substantially larger than the number of fitted regression parameters;
* final validation: bootstrap the complete segmentation procedure.

## What the result actually establishes

A significant split provides evidence that

[
P(Y_t\mid X_t)
]

is not temporally invariant. Depending on the loss used, this may mean that one or more of the following changed:

* regression coefficients;
* nonlinear response function;
* relevant feature set;
* interaction effects;
* residual variance;
* residual distribution.

This is appropriately called a **conditional-mechanism change** or **regression change point**.

It is not, from observational data alone, definitive proof that the full causal graph changed. A causal interpretation requires assumptions such as:

* the relevant causal parents of the outcomes are contained in (X);
* the measurement and selection processes did not change;
* important unobserved causes did not change their relationship with the measured variables;
* the model class is sufficiently appropriate;
* temporal or within-entity dependence is handled.

Modern causal change-point work similarly uses changes in observational invariance and notes that these acquire a causal meaning only under additional assumptions. ([arXiv][4])

## My concrete recommendation

For the simplest useful implementation:

1. Standardize all continuous outcomes.
2. Use multivariate linear regression—or ridge regression when (N) is not small relative to the number of rows.
3. Use the sum of squared prediction errors as the segment cost.
4. Run PELT to obtain zero, one, or several candidate splits.
5. Validate the result by bootstrapping residuals from the pooled model and rerunning PELT.
6. Report the result as a change in the **observed (X\rightarrow Y) conditional mechanism**, rather than automatically as a change in the complete causal model.

For the special case where you expect at most one split, scanning a Chow/Wald statistic and bootstrapping its maximum is even simpler than PELT.

[1]: https://www.jstor.org/stable/1910133?utm_source=chatgpt.com "Tests of Equality Between Sets of Coefficients in Two ..."
[2]: https://www.jstor.org/stable/2951764?utm_source=chatgpt.com "Tests for Parameter Instability and Structural Change With ..."
[3]: https://arxiv.org/abs/1101.1438?utm_source=chatgpt.com "Optimal detection of changepoints with a linear computational cost"
[4]: https://arxiv.org/abs/2403.12677?utm_source=chatgpt.com "Causal Change Point Detection and Localization"
