Yes, but there is an important complication specific to sparse PLS:

> **The original PLS components are often orthogonal, or nearly orthogonal, by construction.**

Therefore, computing correlations between the original PLS score vectors may produce values close to zero even when the corresponding feature groups are substantively related. This can be an artifact of sequential PLS deflation rather than evidence that the underlying constructs are independent. Depending on the sparse-PLS implementation, orthogonality may be exact or only approximate. ([ofai.at][1])

Consequently, I would use sparse PLS primarily to identify the feature groups and then estimate their relationships in a second, non-orthogonal model.

## 1. First construct non-orthogonal scores for the identified groups

Suppose sparse PLS identifies (K) groups:

[
G_1,\ldots,G_K.
]

Instead of retaining the sequentially deflated PLS components, construct one score independently for each group:

[
z_k=X_{G_k}w_k.
]

Possible choices for (w_k) include:

* the sparse-PLS weights restricted to (G_k);
* the first principal-component weights within (G_k);
* a one-factor model fitted to (X_{G_k});
* equal standardized weights when interpretability is more important than optimization.

The essential point is that the groups should be scored **without residualizing one latent variable against the previously extracted latent variables**.

You then obtain

[
Z=
\begin{bmatrix}
z_1 & \cdots & z_K
\end{bmatrix},
]

on which several relationship-analysis methods can be applied.

## Available methods

| Intended relationship             | Suitable method                                | Meaning of an edge                                                     |
| --------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| General pairwise association      | Correlation or rank correlation                | The two latent scores vary together                                    |
| Nonlinear association             | Mutual information, distance correlation, HSIC | The scores are statistically dependent                                 |
| Direct conditional association    | Partial-correlation network                    | Association remaining after controlling for the other latent variables |
| Hypothesized directional relation | Path analysis or SEM                           | Regression/path coefficient under the specified model                  |
| Exploratory directional structure | PC, GES, FCI or LiNGAM                         | Direction supported under algorithm-specific causal assumptions        |
| Temporally ordered relation       | VAR, Granger analysis or dynamic SEM           | Earlier values predict later values                                    |
| Nonlinear directional model       | Additive-noise or nonlinear SEM                | Directed nonlinear dependence under stronger assumptions               |

## 2. Pairwise correlation: the simplest option

Calculate

[
R_{k\ell}=\operatorname{Corr}(z_k,z_\ell).
]

Use:

* Pearson correlation for approximately linear relationships;
* Spearman correlation for monotonic relationships;
* a nonlinear dependence measure when non-monotonic relationships are plausible.

This produces an undirected network:

[
z_k-z_\ell
]

whenever the corresponding association is sufficiently strong.

However, pairwise correlation can produce misleading edges. For example,

[
z_1\longleftrightarrow z_2\longleftrightarrow z_3
]

can generate a correlation between (z_1) and (z_3), even when their relationship is entirely explained by (z_2).

Therefore, pairwise correlation is useful mainly for an initial exploratory visualization.

## 3. Partial-correlation network: the best simple default

A more informative method is to estimate the covariance matrix of (Z),

[
\Sigma_Z=\operatorname{Cov}(Z),
]

and its inverse,

[
\Theta=\Sigma_Z^{-1}.
]

The partial correlation between latent variables (z_k) and (z_\ell), conditional on all the other latent variables, is

[
\rho_{k\ell\mid-{k,\ell}}
=========================

-\frac{\Theta_{k\ell}}
{\sqrt{\Theta_{kk}\Theta_{\ell\ell}}}.
]

A nonzero value means that (z_k) and (z_\ell) remain associated after controlling for the remaining latent variables.

This gives an undirected conditional-dependence graph:

[
z_k-z_\ell.
]

When (K) is small relative to the number of rows, the ordinary inverse covariance matrix may be sufficient. When (K) is large, or the covariance matrix is unstable, use the **graphical lasso**:

[
\widehat{\Theta}
================

\arg\max_{\Theta\succ0}
\left[
\log\det\Theta
--------------

## \operatorname{tr}(S\Theta)

\lambda\lVert\Theta\rVert_1
\right].
]

The penalty (\lambda) removes weak edges and produces a sparse network. Graphical lasso was developed specifically for sparse inverse-covariance estimation and undirected Gaussian graphical models. ([Hastie][2])

### Interpretation

An edge means:

> “These two latent scores are conditionally associated, given the other latent scores.”

It does **not** determine whether

[
z_k\rightarrow z_\ell,
\qquad
z_\ell\rightarrow z_k,
]

or whether both are affected by another variable.

For the setting you described, this is probably the **simplest defensible method** for discovering relationships among the latent groups.

## 4. Structural equation modeling: best for hypothesized directions

When you have substantive reasons to hypothesize directions, use a structural equation model:

[
Z=BZ+\zeta,
]

where (B_{k\ell}) represents a directed path

[
z_\ell\rightarrow z_k.
]

The complete model can jointly represent the feature groups, relationships among latent variables, and outcomes:

[
X=\Lambda Z+\epsilon,
]

[
Z=BZ+\zeta,
]

[
Y=\Gamma Z+\eta.
]

Here:

* (\Lambda) connects observed features to latent variables;
* (B) represents relationships among latent variables;
* (\Gamma) connects the latent variables to the outcomes;
* (\epsilon,\zeta,\eta) are residual terms.

For example:

[
z_1\rightarrow z_2\rightarrow z_3\rightarrow Y
]

allows estimation of:

* the direct effect of (z_1) on (z_2);
* the direct effect of (z_2) on (z_3);
* the indirect relationship between (z_1) and (z_3);
* the mediated relationship between (z_1) and (Y).

SEM is preferable when the variables are regarded as genuine latent constructs because the measurement and structural relationships can be estimated jointly, rather than treating estimated scores as perfectly observed quantities. ([PubMed Central (PMC)][3])

### Important distinction: factors versus composites

Sparse PLS normally produces **composites**:

[
z_k=X_{G_k}w_k.
]

These are deterministic combinations of observed features.

A common-factor model instead assumes:

[
X_{G_k}=\lambda_k z_k+\epsilon_k,
]

meaning that the latent variable is an underlying common source of the observed variables.

Therefore:

* use **path analysis on composite scores** when the latent variables are operational summaries of their feature groups;
* use **factor-based SEM** when each latent variable is interpreted as an underlying construct generating its observed indicators.

## 5. Factor-score path analysis: a two-stage alternative

A computationally simpler alternative to full SEM is:

1. estimate each latent variable or factor;
2. compute factor scores;
3. estimate a path model between those scores.

However, naively treating estimated factor scores as error-free observed variables can bias path coefficients. Bias-corrected factor-score path analysis, such as Croon-corrected factor-score path analysis, adjusts the score covariance matrix before estimating the structural paths. ([Hogrefe Econtent][4])

This approach is useful when:

* there are many indicators;
* fitting one large SEM is difficult;
* the measurement groups have already been established;
* a relatively simple structural model is desired.

For PLS composites rather than common factors, bootstrap-based uncertainty propagation is more natural than ordinary factor-score correction.

## 6. Exploratory causal discovery

When the directions between the latent variables are not known in advance, causal-discovery methods can be applied to the non-orthogonal (Z) matrix.

### PC algorithm

The PC algorithm uses conditional-independence tests to infer a graph. It generally returns a partially directed equivalence-class graph rather than one uniquely directed DAG. Its interpretation depends on assumptions such as acyclicity, the causal Markov condition, faithfulness and, in its standard form, absence of unmeasured common causes. ([Journal of Machine Learning Research][5])

Possible output:

[
z_1\rightarrow z_2-z_3.
]

This means that some directions are identified while others remain ambiguous.

### FCI or RFCI

When additional unobserved common causes may remain, FCI or RFCI is more appropriate. These methods can represent ambiguity and possible hidden confounding using a partial ancestral graph. RFCI was designed as a faster alternative to FCI while retaining asymptotically valid causal information under its assumptions. ([arXiv][6])

Possible output:

[
z_1\leftrightarrow z_2,
]

which may indicate that an unobserved common cause cannot be excluded.

### LiNGAM

For approximately linear, acyclic relationships with non-Gaussian and mutually independent disturbances, DirectLiNGAM can estimate both a causal ordering and connection strengths:

[
Z=BZ+\epsilon.
]

Unlike covariance-only methods, non-Gaussianity can make the full ordering identifiable under LiNGAM’s assumptions. The ordinary DirectLiNGAM model assumes no latent confounders among the analyzed variables. ([Journal of Machine Learning Research][7])

LiNGAM is attractive when its assumptions are plausible, but estimated PLS scores can complicate the independent-error assumption because different scores are generated from the same original feature table.

## 7. Dedicated latent-variable causal discovery

Instead of first estimating scores and then applying a conventional causal-discovery algorithm, one can use a method that reasons directly about the indicators and latent variables.

For example, latent-structure discovery methods can:

1. identify groups of observed variables associated with common latent causes;
2. derive conditional-independence information among the resulting latent variables;
3. return possible structural relationships between those latent variables.

Silva and colleagues developed such procedures for linear latent-variable models, including the discovery of measurement structures and information about relations among the identified latent factors. These approaches require substantially stronger measurement-model assumptions than sparse PLS, but they are more faithful to the claim that the variables are genuinely latent. ([UCL Homepages][8])

## 8. Temporal relationships

When each row belongs to a meaningful time point and the latent scores can be calculated repeatedly over time, use lagged models such as

[
Z_t=A_1Z_{t-1}+A_2Z_{t-2}+\epsilon_t.
]

An entry

[
(A_1)_{k\ell}\neq0
]

means that the previous value of (z_\ell) helps predict the current value of (z_k), conditional on the other lagged latent variables.

Appropriate models include:

* vector autoregression;
* Granger-predictive networks;
* dynamic structural equation models;
* structural VAR;
* cross-lagged models for repeated subjects.

These provide stronger directional information than a cross-sectional table because temporal precedence is available, although temporal prediction alone still does not guarantee causal influence.

## Recommended procedure for your setting

Given only the table of observed features and outcomes, I would use the following pipeline.

### Step 1: use sparse PLS only for grouping

Obtain stable groups

[
G_1,\ldots,G_K
]

using sparse PLS and bootstrap selection frequencies.

### Step 2: reconstruct the group variables without PLS deflation

For every group, estimate

[
z_k=X_{G_k}\widetilde{w}_k
]

independently, or estimate all (K) constructs using a joint factor model that permits correlated latent variables.

### Step 3: estimate an undirected relationship network

Use partial correlations:

[
\rho_{k\ell\mid-{k,\ell}}.
]

Use ordinary inverse covariance when (K) is small and graphical lasso when regularization is necessary.

### Step 4: assess stability

Repeat the **entire procedure**, including sparse-PLS grouping and score construction, in bootstrap samples. Report:

* edge-selection frequency;
* median partial correlation;
* bootstrap interval;
* stability of the feature membership of both endpoint groups.

This is important because uncertainty in the original latent-variable identification propagates into uncertainty in the relationships between them.

### Step 5: add directions only when justified

Use:

* SEM when directions come from theory or domain knowledge;
* temporal models when lagged observations exist;
* PC or FCI for assumption-dependent exploratory orientation;
* LiNGAM when linearity, acyclicity, non-Gaussianity and independent disturbances are plausible.

## Bottom line

For a **simple and effective exploratory analysis**, the best choice is:

[
\boxed{
\text{sparse PLS grouping}
\rightarrow
\text{non-orthogonal group scores}
\rightarrow
\text{partial-correlation network}
}
]

For a model containing **directed relationships and mediation**, use:

[
\boxed{
X=\Lambda Z+\epsilon,\qquad
Z=BZ+\zeta,\qquad
Y=\Gamma Z+\eta
}
]

estimated as a structural equation model.

The outcomes (Y) can help estimate which latent variables are outcome-relevant, but they do not by themselves identify the causal direction between (z_k) and (z_\ell). Without temporal ordering, interventions, domain restrictions or additional distributional assumptions, relationships between the latent variables should normally be reported as **conditional associations**, not causal influences.

[1]: https://www.ofai.at/~roman.rosipal/Papers/pls_book06.pdf?utm_source=chatgpt.com "Overview and Recent Advances in Partial Least Squares"
[2]: https://hastie.su.domains/Papers/graph.pdf?utm_source=chatgpt.com "Sparse inverse covariance estimation with the lasso"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5794813/?utm_source=chatgpt.com "An overview of structural equation modeling: its beginnings ..."
[4]: https://econtent.hogrefe.com/doi/10.1027/1614-2241/a000130?utm_source=chatgpt.com "Factor Score Path Analysis: An Alternative for SEM?"
[5]: https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf?utm_source=chatgpt.com "Estimating High-Dimensional Directed Acyclic Graphs with ..."
[6]: https://arxiv.org/abs/1104.5617?utm_source=chatgpt.com "Learning high-dimensional directed acyclic graphs with latent and selection variables"
[7]: https://www.jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf "shimizu11a.dvi"
[8]: https://www.homepages.ucl.ac.uk/~ucgtrbd/papers/jmlr-final.pdf?utm_source=chatgpt.com "Learning the Structure of Linear Latent Variable Models"
