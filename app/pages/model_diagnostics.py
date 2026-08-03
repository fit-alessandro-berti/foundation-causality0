"""Model card, bundle integrity, calibration, and known limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import model_status
from app.runtime import PROJECT_ROOT, runner


def render() -> None:
    st.title("Model diagnostics and model card")
    bundle = runner().bundle
    model_status.render(bundle)
    st.subheader("Architecture configuration")
    st.json(bundle.config.to_dict())
    st.subheader("Bundle integrity")
    st.json(bundle.describe())
    if bundle.calibration:
        st.subheader("Checkpoint-bound calibration")
        calibration = bundle.calibration
        st.dataframe(
            pd.DataFrame(
                [{"group": key, "radius": value, "count": calibration.counts.get(key)} for key, value in calibration.normalized_radius.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.json(calibration.regime)
    st.subheader("Known limitations")
    st.warning("The selected checkpoint's neural residual gate is nearly closed; final estimates are effectively compiled expert estimates.")
    st.warning("World particles remain almost uniform and are not demonstrated as separated causal worlds.")
    st.warning("Prior-mismatch and graph-logit diagnostics are research outputs, not validated decision rules.")
    st.markdown(
        "\n".join(
            [
                f"- [Checkpoint manifest]({(PROJECT_ROOT / 'artifacts/cwfm/checkpoint_manifest.json').as_posix()})",
                f"- [Results manifest]({(PROJECT_ROOT / 'artifacts/cwfm/results/manifest.json').as_posix()})",
                f"- [Experimental report]({(PROJECT_ROOT / 'docs/foundation_model/experimental_results.md').as_posix()})",
            ]
        )
    )

