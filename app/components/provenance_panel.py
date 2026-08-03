"""Assumption and reproducibility disclosure."""

from __future__ import annotations

import streamlit as st


def render(result) -> None:
    with st.expander("Assumptions and provenance"):
        st.write("Assumption ledger")
        st.json(result.provenance.assumptions)
        st.write("Provenance")
        st.json(result.provenance.__dict__)

