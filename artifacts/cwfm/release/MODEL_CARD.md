# CWFM reference checkpoint bundle

This bundle describes the released CWFM reference model for static tabular
effects, observed-regime determination, and network interference. The archive
does not currently include `model_state.pt`; applications must report the model
as unavailable until weights matching the SHA-256 in `manifest.json` are placed
here.

Identification is a formal gate derived from declared assumptions. It is not a
learned probability inferred from observational data. Calibration is bound to
the selected checkpoint and must not be recomputed during application startup.

The reported experiment found a nearly closed residual gate and nearly uniform
world particles. These values are available as research diagnostics but should
not be interpreted as demonstrated gains over the compiled expert estimate.
