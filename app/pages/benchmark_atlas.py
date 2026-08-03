"""Truth-safe catalog of CWFM and classical reference benchmark families."""

from __future__ import annotations

import streamlit as st

from app.runtime import catalog
from cwfm.application import load_reference_result


REFERENCE_LABELS = {
    "latent_variable_determination": "Latent grouping",
    "latent_variable_determination2": "Latent graph reconstruction",
    "temporal_split_detection": "Temporal coefficient break detection",
    "temporal_split_detection_method": "Temporal VARX break attribution",
}


def render() -> None:
    st.title("Benchmark Atlas")
    st.info(
        "These families use classical reference evaluators. The current CWFM checkpoint is not used and no CWFM prediction is claimed."
    )
    method = st.selectbox(
        "Technique family", list(REFERENCE_LABELS), format_func=REFERENCE_LABELS.get
    )
    source = catalog()
    scenario = st.selectbox("Scenario", source.scenarios(method))
    seed = st.selectbox("Seed", source.seeds(method, scenario), format_func=lambda value: f"{value:04d}")
    case = source.resolve(method, scenario, seed)
    st.success("Backend: classical reference evaluator · CWFM checkpoint used: no")
    st.subheader("Safe discovery-data preview")
    st.json(case.preview())
    st.subheader("Stored classical evaluator result")
    reference = load_reference_result(case.project_root, method, scenario, seed)
    st.json(reference.metrics)
    st.caption(
        "Ground truth is not loaded in this view. A future explicit audit control may add post-hoc truth comparisons without changing a fitted result."
    )
