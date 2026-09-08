# CWFM reference checkpoint bundle

This bundle contains the released CWFM reference model for static tabular
effects, observed-regime determination, and network interference. Applications
verify `model_state.pt` against the SHA-256 recorded in `manifest.json` before
loading it.

Identification is a formal gate derived from declared assumptions. It is not a
learned probability inferred from observational data. Calibration is bound to
the selected checkpoint and must not be recomputed during application startup.

The reported experiment found a nearly closed residual gate and nearly uniform
world particles. These values are available as research diagnostics but should
not be interpreted as demonstrated gains over the compiled expert estimate.

An answered status establishes compliance with the implemented rules, not the
truth or completeness of user metadata or correctness of a causal estimate.
Incorrect randomization, confounding, or exposure declarations can pass the gate.
The empirical ridge overlap screen is a heuristic: the revision audit finds
false refusals at small sample sizes and systematic false acceptance for a
quadratic treatment-assignment mechanism. It does not certify population positivity.

The encoder has no separate column-position embedding, but the query includes
normalized treatment, outcome, and exposure indices. Presentation randomization
and the confirmed covariate-permutation repair do not establish invariance to
arbitrary relocation of those columns or to changes in the selected covariates.

In the existing 1,800-episode confirmatory comparison, the shallow random-forest
risk selector and convex stack have MAEs 0.141 and 0.147, versus 0.161 for this
frozen router. The release remains the neural baseline; this revision does not
replace its weights, inference policy, or calibration files. See
`artifacts/cwfm/revision/` and `artifacts/cwfm/support_revision/` for the evidence.
