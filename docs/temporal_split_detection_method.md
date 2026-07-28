## Recommended method: lagged regression with structural-break detection

The simplest sensible approach is to model the data as a **piecewise lagged multivariate regression**, also describable as a **piecewise VARX(1)** model:

[
Y_t
===

c_k
+
A_kY_{t-1}
+
B_kX_{t-1}
+
\varepsilon_t,
\qquad
\tau_{k-1}<t\leq \tau_k .
]

Here:

* (X_t\in\mathbb{R}^N) contains the observed features in row (t);
* (Y_t\in\mathbb{R}^M) contains the observed outcomes;
* (\tau_1,\ldots,\tau_K) are unknown temporal split points;
* (A_k) describes persistence from previous outcomes;
* (B_k) describes how features in the immediately preceding row affect the current outcomes;
* the matrices (A_k) and (B_k) are constant inside a period but may change at a split.

Piecewise VAR models are specifically designed to represent time series whose lagged transition mechanism changes at unknown points. ([arXiv][1])

Including (Y_{t-1}) is important. Without it, (X_{t-1}) may appear influential merely because both (X) and (Y) are persistent over time.

## How to find the splits

For every candidate segment ([a,b]), calculate the best lagged-regression error:

[
C(a,b)
======

\min_{c,A,B}
\sum_{t=a}^{b}
\left|
Y_t-c-AY_{t-1}-BX_{t-1}
\right|_2^2.
]

Then find the partition that minimizes

[
\sum_{k=0}^{K}C(\tau_k+1,\tau_{k+1})
+
\beta K,
]

where (\beta) penalizes adding too many split points.

For a **single possible split**, the simplest implementation is merely to scan all admissible positions (s):

[
s^*
===

\arg\min_s
\left[
C(2,s)+C(s+1,T)
\right].
]

You then compare:

[
\text{one-model cost}=C(2,T)
]

against

[
\text{two-model cost}
=====================

C(2,s^*)+C(s^*+1,T)+\beta.
]

Accept the split only when the improvement is large enough to overcome the penalty.

For **multiple splits**, use either:

* **Bai–Perron multiple structural-break regression**, the classical statistical approach for testing and estimating multiple coefficient changes in regression models; or
* **PELT**, using the regression residual error above as its segment cost.

Bai–Perron directly addresses unknown multiple regression breakpoints. ([IDEAS/RePEc][2]) PELT minimizes a penalized additive segmentation objective exactly and, under its stated conditions, can have computational cost approximately linear in the number of observations. ([arXiv][3])

## How to determine whether the change is specifically “causal”

A detected split may result from several different changes:

[
c_k\neq c_{k+1},
\qquad
A_k\neq A_{k+1},
\qquad
B_k\neq B_{k+1},
\qquad\text{or}\qquad
\operatorname{Var}(\varepsilon_t)
\text{ changing}.
]

Only a change in (B_k) corresponds to a change in the lagged feature-to-outcome relationship.

Therefore, after finding the breakpoints, compare these two models inside every segment.

Restricted model:

[
Y_t=c_k+A_kY_{t-1}+\varepsilon_t.
]

Full model:

[
Y_t=c_k+A_kY_{t-1}+B_kX_{t-1}+\varepsilon_t.
]

Then check whether adding (X_{t-1}):

1. significantly reduces prediction error, using a joint Wald/F test when ordinary least squares assumptions are reasonable; or
2. reduces out-of-sample error under blocked temporal cross-validation.

Also test across consecutive segments whether

[
H_0:B_k=B_{k+1}.
]

Rejecting this hypothesis is evidence that the lagged feature-to-outcome mechanism changed at that split.

This supports **Granger-predictive causality**: past (X) helps predict current (Y) beyond what past (Y) predicts. It does not by itself establish intervention-level causal effects, because hidden common causes, trends, selection effects, or another omitted lagged variable can produce the same pattern. ([MIMUW][4])

## Minimal practical algorithm

```text
1. Sort rows by their actual timestamps.
2. Construct:
       response[t]  = Y[t]
       predictors[t] = [Y[t-1], X[t-1]]
3. Exclude the first row.
4. Define the segment cost as multivariate regression residual error.
5. Set a minimum segment length.
6. Run PELT with a BIC-type penalty.
7. Refit the model separately within every detected segment.
8. Test whether the B matrices differ between adjacent segments.
9. Compare the full and restricted models within each segment.
```

For unregularized regression, every segment must contain more observations than the number of predictors,

[
N+M+1,
]

and preferably substantially more. When (N+M) is large relative to the number of rows, use **ridge regression** for each segment and select both the ridge strength and segmentation penalty using blocked temporal validation.

## Important limitation concerning what a “row” represents

This method is appropriate only when row (t) and row (t-1) describe consecutive states of the **same evolving system**, or when there is another defensible reason why the immediately previous row can affect the current one.

If the rows are independent cases merely sorted by their occurrence time—for example, different patients, customers, process instances, or organizations—then

[
X_{t-1}\rightarrow Y_t
]

usually has no meaningful causal interpretation. In that situation, either:

* construct lags separately inside each entity or process trace; or
* aggregate observations into fixed time windows and model outcomes in window (j) from features aggregated over window (j-1).

If “preceding period” literally means the preceding **detected segment**, rather than the preceding row, an aggregation rule must first be specified, such as the mean, count, maximum, or distributional embedding of the features in that segment. Otherwise, the segmentation and the definition of the lagged predictor become circular.

## Bottom line

Use:

[
\boxed{\text{piecewise }Y_t\sim Y_{t-1}+X_{t-1}}
]

with **PELT or Bai–Perron structural-break detection**. Then retain as “causal-model splits” only those breakpoints for which the coefficient matrix on the lagged features, (B_k), changes—not merely the mean, variance, or autoregressive behavior of the outcomes.

[1]: https://arxiv.org/abs/1711.07357?utm_source=chatgpt.com "Joint Structural Break Detection and Parameter Estimation in High-Dimensional Non-Stationary VAR Models"
[2]: https://ideas.repec.org/a/ecm/emetrp/v66y1998i1p47-78.html?utm_source=chatgpt.com "Estimating and Testing Linear Models with Multiple Structura"
[3]: https://arxiv.org/abs/1101.1438?utm_source=chatgpt.com "Optimal detection of changepoints with a linear computational cost"
[4]: https://www.mimuw.edu.pl/~noble/courses/TimeSeries/RESOURCES/69EconometricaGrangerCausality.pdf?utm_source=chatgpt.com "Investigating Causal Relations by Econometric Models and ..."
