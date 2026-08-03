"""Home and explicit scope/status disclosure."""

from __future__ import annotations

import streamlit as st

from app.components import model_status
from app.runtime import catalog, runner


def render() -> None:
    st.title("CWFM Application")
    st.write(
        "Build once, expose twice: the scripts and this interface use the same truth-safe application runner."
    )
    model_status.render(runner().bundle)
    supported, reference, unavailable = st.columns(3)
    with supported:
        st.subheader("Supported by current checkpoint")
        st.write("✓ Static treatment effects")
        st.write("✓ Observed-regime determination")
        st.write("✓ Network interference")
    with reference:
        st.subheader("Reference techniques")
        st.write("• Latent grouping")
        st.write("• Latent graph reconstruction")
        st.write("• Temporal split detection")
        st.write("• Temporal VARX attribution")
    with unavailable:
        st.subheader("Not implemented in this checkpoint")
        st.write("• General temporal CWFM")
        st.write("• Latent-variable CWFM")
        st.write("• Partial-identification bounds")
    st.divider()
    st.subheader("Repository inputs")
    st.metric("Controlled catalog cases", len(catalog()))
    st.info(
        "Normal analysis reads discovery data and allowlisted descriptive metadata only. Simulated truth is isolated from fitting."
    )

