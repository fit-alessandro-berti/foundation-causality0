"""Model availability status card."""

from __future__ import annotations

import streamlit as st

from cwfm.application import ModelBundleStatus


def render(bundle, compact: bool = False) -> None:
    available = bundle.status == ModelBundleStatus.AVAILABLE
    if available:
        st.success("Model available · checkpoint and calibration verified")
    elif bundle.status == ModelBundleStatus.INVALID:
        st.error(f"Model bundle invalid · {bundle.reason}")
    else:
        st.warning(f"Model unavailable · {bundle.reason}")
    if not compact:
        first, second, third = st.columns(3)
        first.metric("Maximum variables", bundle.config.max_variables)
        second.metric("Calibration", "Loaded" if bundle.calibration else "Missing")
        third.metric("Inference", "Available" if available else "Unavailable")

