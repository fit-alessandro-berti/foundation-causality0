## Recommended method: model-based recursive partitioning

The simplest defensible method is **model-based recursive partitioning**, often abbreviated **MOB**.

It is essentially a decision tree, but instead of splitting the data to improve ordinary outcome prediction, it splits whenever the **parameters of the fitted causal or structural model change**. The selected splitting features are candidates for the observed variables that determine the causal regime. MOB was designed precisely around this sequence: fit a model, test its parameters for instability over candidate variables, split on the strongest instability, and repeat recursively. ([Tandfonline][1])

Let:

[
X=(X_1,\ldots ,X_N)
]

be the observed candidate features and

[
Y=(Y_1,\ldots ,Y_M)
]

the observed outcomes. Your assumption is approximately:

[
Y \sim \mathcal M_{\theta(X_S)}
]

where (S\subseteq{1,\ldots ,N}) is an unknown subset of features and (\theta) represents the parameters—or potentially the structure—of the causal model.

### Procedure

First, fit one common model to all observations:

[
\mathcal M_{\hat\theta_0}.
]

This could be a multivariate regression, structural equation model, causal effect model, Bayesian network, or another model appropriate to the outcomes.

Then, for every candidate feature (X_j), test:

[
H_{0j}: \theta \text{ is constant over the values of }X_j
]

against:

[
H_{1j}: \theta \text{ changes with }X_j.
]

For a continuous feature, examine admissible thresholds (c):

[
X_j\leq c
\qquad\text{versus}\qquad
X_j>c.
]

For each candidate split, fit the model separately on the two sides and calculate something such as:

[
G(j,c)=
\ell!\left(\hat\theta_{j,c}^{L}\right)
+
\ell!\left(\hat\theta_{j,c}^{R}\right)
--------------------------------------

\ell!\left(\hat\theta_0\right),
]

where (\ell) is the log-likelihood. Equivalently, you can measure the reduction in held-out prediction loss or structural-equation residual loss.

Choose the feature and threshold with the strongest statistically significant improvement:

[
(j^*,c^*)=\arg\max_{j,c}G(j,c).
]

Repeat the procedure within the resulting subsets. The output might look like:

```text
X7 ≤ 2.4
├── X12 = A       → causal model 1
├── X12 = B       → causal model 2
└── otherwise     → causal model 3

X7 > 2.4          → causal model 4
```

Here, (X_7) and (X_{12}) are the candidate **context variables**, **regime variables**, or **causal moderators**.

## The simplest initial implementation

Start with a **one-level MOB tree**, also describable as a structural-change decision stump:

1. Test all (N) features.
2. Find the best threshold or categorical partition for each feature.
3. Compare the global model against the two-regime model.
4. Use a permutation test or held-out data to determine whether the improvement is real.
5. Rank the features by their instability statistic and bootstrap selection frequency.

Only build a deeper tree when the first split is convincingly supported. This makes the result easier to interpret and reduces false regime discovery.

For (M) outcomes, either fit a single multivariate model or combine the outcome-specific likelihoods:

[
G_{\text{joint}}(j,c)
=====================

\sum_{m=1}^{M} w_m G_m(j,c),
]

with standardized weights (w_m). A multivariate likelihood is preferable when the outcomes are strongly correlated.

## Why an ordinary decision tree or random forest is insufficient

A standard tree answers:

> Which features help predict the outcomes?

Your question is different:

> Which features make the relationship among variables change?

For example, suppose:

[
Y=3X_1+2X_2+\epsilon
]

holds everywhere. (X_1) may be the strongest predictor, but it does not select between causal models because there is only one model.

Now suppose:

[
Y=
\begin{cases}
3X_1+2X_2+\epsilon, & X_7\leq 2.4,\
-1X_1+5X_2+\epsilon, & X_7>2.4.
\end{cases}
]

Then (X_7) is the model-determining feature, although it might not itself be a strong direct predictor of (Y). A standard feature-importance method can easily miss this distinction.

## When to use a causal tree instead

When you have a clearly defined treatment or intervention (T) and want to determine which features modify its effect on the outcomes, use a **causal tree**:

[
\tau(x)
=======

## E[Y\mid do(T=1),X=x]

E[Y\mid do(T=0),X=x].
]

The tree searches for features over which (\tau(x)) changes, rather than features over which the raw outcome prediction changes. Causal trees were developed to partition observations into subpopulations with heterogeneous causal effects. ([PNAS][2])

Thus:

| Meaning of “causal model changes”            | Simplest suitable method                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------ |
| Regression or structural coefficients change | Model-based recursive partitioning                                       |
| Effect of a known treatment changes          | Causal tree                                                              |
| Entire causal graph or edge set changes      | Context-specific causal discovery, possibly after recursive partitioning |

## Important limitations

### 1. Model misspecification can create false regimes

Suppose the true mechanism is globally:

[
Y=X_1^2+\epsilon.
]

If you fit only a linear model (Y=\beta X_1+\epsilon), the method may split on (X_1) and conclude that the coefficients change across regimes. In reality, there is one nonlinear mechanism.

Therefore, the within-regime model should already account for plausible:

* nonlinear direct effects;
* ordinary interactions;
* time trends;
* heteroskedasticity;
* known confounders.

A split should represent instability remaining **after** those effects are modeled.

### 2. Use pre-outcome candidate features

For a causal interpretation, the candidate regime features should normally be measured before the outcomes and should not be descendants of the outcomes. Splitting on a downstream consequence can create apparently different models merely through conditioning or selection.

### 3. Observational data may not identify the complete causal graph

If “underlying causal model” means a fully directed causal graph, purely observational data generally identify only an equivalence class without stronger assumptions or interventions. Interventional data can reduce this ambiguity. ([OUP Academic][3])

Consequently, MOB can robustly identify **where the fitted mechanism changes**, but calling those differences changes in the true causal graph additionally requires assumptions such as:

* a known causal ordering;
* no relevant unobserved confounding;
* appropriate functional-form or noise assumptions;
* temporal information;
* interventions or multiple environments.

## Practical recommendation

Use:

[
\boxed{\text{Model-based recursive partitioning with parameter-instability tests}}
]

and begin with a single split. Validate each selected feature with:

* out-of-sample improvement;
* a permutation test that includes the threshold search;
* bootstrap selection frequency;
* minimum subgroup sizes;
* re-estimation of the local models on data not used to choose the split.

The final evidence for a model-determining feature should therefore be not merely that it appears once in the tree, but that it repeatedly produces a substantial and reproducible change in the relevant causal-model parameters.

[1]: https://www.tandfonline.com/doi/abs/10.1198/106186008X319331?utm_source=chatgpt.com "Model-Based Recursive Partitioning"
[2]: https://www.pnas.org/doi/10.1073/pnas.1510489113?utm_source=chatgpt.com "Recursive partitioning for heterogeneous causal effects"
[3]: https://academic.oup.com/jrsssb/article-abstract/77/1/291/7041960?utm_source=chatgpt.com "Jointly Interventional and Observational Data: Estimation of ..."
