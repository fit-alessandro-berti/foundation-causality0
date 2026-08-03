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
