## Recommended method: sparse multi-response PLS

Assuming each row is an observation, let

[
X\in\mathbb{R}^{S\times N}
]

contain the (N) observed features and

[
Y\in\mathbb{R}^{S\times M}
]

contain the (M) outcomes.

A simple and effective starting point is **sparse Partial Least Squares with multiple responses**, often called **sparse PLS2**.

PLS constructs latent components from (X) while explicitly using their relationship with (Y). This distinguishes it from PCA or ordinary factor analysis, which discover structure in (X) without considering whether that structure is relevant to the outcomes. Sparse PLS additionally forces each component to use only a limited subset of features, simultaneously performing dimension reduction and feature selection. 

### Model

For latent component (k), sparse PLS approximately solves

[
\max_{w_k,c_k}
\operatorname{Cov}\left(Xw_k,Yc_k\right)^2
]

subject to normalization constraints and a sparsity constraint on (w_k), such as

[
\lVert w_k\rVert_1\leq s.
]

The resulting latent variables are

[
z_k=Xw_k,
]

and the outcomes are modeled as

[
Y\approx ZB,
\qquad
Z=[z_1,\ldots,z_K].
]

Here:

* (W=[w_1,\ldots,w_K]) describes which observed features define each latent component.
* (B) describes how strongly each latent component is associated with each outcome.
* (K) is the number of latent components.

Sparse PLS was specifically proposed to produce sparse combinations of predictors while retaining the predictive dimension-reduction properties of PLS, including for multivariate outcomes. ([Wiley Online Library][1])

## Practical procedure

### 1. Standardize both blocks

Center and scale every feature and every outcome:

[
x_{ij}^{*}=\frac{x_{ij}-\bar{x}_j}{s_j}.
]

Scaling (Y) is particularly important. Otherwise, an outcome with a numerically larger variance may dominate the latent components. Outcomes can instead be deliberately weighted when some are scientifically more important.

Missing-value imputation and scaling should be learned inside each training fold, not before cross-validation.

### 2. Select the number of components and sparsity

Fit candidate models over:

[
K=1,2,\ldots,K_{\max}
]

and several sparsity levels, expressed either as:

* the number of features retained per component, or
* an (L_1)-penalty parameter.

Choose them using repeated nested cross-validation and a multivariate validation loss such as

[
\mathrm{CV\ loss}
=================

\sum_{m=1}^{M}\alpha_m
\operatorname{NRMSE}(Y_m,\widehat{Y}_m),
]

where (\alpha_m) controls the importance of outcome (m).

The inner cross-validation chooses (K) and sparsity. The outer cross-validation estimates predictive performance without optimistic bias.

### 3. Construct feature groups

Sparse PLS gives sparse loadings, but it does **not necessarily produce a strict partition**: a feature may load on more than one component.

For an interpretable division, assign feature (j) to the component on which it has the largest absolute loading:

[
g(j)=
\arg\max_k |w_{jk}|.
]

A prudent assignment rule is

[
G_k=
\left{
j:
k=\arg\max_{\ell}|w_{j\ell}|,
\quad
|w_{jk}|>\tau
\right}.
]

Features with weak loadings should remain **unassigned**, rather than being forced into a latent group.

For prediction, retain the original soft loadings. Use the hard assignment only for naming and presenting the latent groups.

### 4. Check stability

Repeat the complete fitting procedure on bootstrap samples. Because latent components can exchange order or reverse sign, first align the bootstrapped components with the original components.

For every feature, report:

* selection frequency,
* most frequent component assignment,
* loading magnitude and sign,
* assignment consistency across resamples.

A latent group should not be interpreted strongly when its membership changes substantially across bootstrap samples. With highly correlated features, individual features may exchange positions even when the broader feature group is stable.

### 5. Relate the latent components to outcomes

After obtaining (Z), inspect the estimated matrix (B):

[
B=
\begin{bmatrix}
b_{11} & \cdots & b_{1M}\
\vdots & \ddots & \vdots\
b_{K1} & \cdots & b_{KM}
\end{bmatrix}.
]

The coefficient (b_{km}) indicates the conditional predictive association between latent component (k) and outcome (m), given the other retained components.

A useful final presentation is a three-layer graph:

[
\text{observed features}
\longrightarrow
\text{latent components}
\longrightarrow
\text{outcomes},
]

where feature-to-component edges are based on (W), and component-to-outcome edges are based on (B).

## Important interpretation limitation

Sparse PLS identifies **outcome-relevant predictive components**. It does not establish that the latent components causally influence the outcomes.

From an ordinary observational table, the defensible wording is:

> “Latent components associated with, or predictive of, the observed outcomes.”

To say “influencing” causally, additional assumptions or evidence are needed, such as temporal ordering, measured confounders, interventions, instrumental variables, or an identifiable causal structural model.

## When factor analysis would be preferable

When the latent variables are intended to represent genuine underlying constructs that generate the observed features, use a sparse factor model or exploratory factor analysis followed by a structural model:

[
X=Z\Lambda^\top+E_X,
]

[
Y=Z\Gamma^\top+E_Y.
]

Here, (\Lambda) is the feature-to-latent loading matrix and (\Gamma) is the latent-to-outcome coefficient matrix. Sparse identifiable latent-variable models can be used to encourage a small, interpretable set of loadings. ([Journal of Machine Learning Research][2])

The distinction is:

| Primary objective                                          | Appropriate method                                                   |
| ---------------------------------------------------------- | -------------------------------------------------------------------- |
| Find feature combinations most predictive of the outcomes  | **Sparse PLS2**                                                      |
| Discover latent constructs underlying feature correlations | **Exploratory factor analysis**                                      |
| Model latent constructs and outcomes jointly               | **Sparse factor regression or SEM**                                  |
| Require each feature to belong to exactly one group        | Constrained sparse factor model, or hard assignment after sparse PLS |

For the formulation in the question, **sparse PLS2 with cross-validated component selection and bootstrap-stable feature assignment is the simplest practical choice**. If “latent” is intended in the stronger measurement-theoretic sense, a sparse factor model with (X=Z\Lambda^\top) and (Y=Z\Gamma^\top) is more faithful, but less straightforward to estimate and validate.

[1]: https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2009.00723.x?utm_source=chatgpt.com "Sparse partial least squares regression for simultaneous ..."
[2]: https://jmlr.org/papers/volume12/henao11a/henao11a.pdf?utm_source=chatgpt.com "Sparse Linear Identifiable Multivariate Modeling"
