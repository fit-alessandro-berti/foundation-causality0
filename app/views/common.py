"""Common result composition."""

from __future__ import annotations

import streamlit as st

from app.components import expert_panel, export_panel, provenance_panel, result_status


def start(result) -> None:
    result_status.render(result)
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)


def finish(result, command: str, research_mode: bool) -> None:
    expert_panel.render(result)
    if research_mode and result.expert_results:
        st.subheader("Research diagnostics")
        first, second, third = st.columns(3)
        first.metric("Residual gate", f"{result.diagnostics.residual_gate:.4f}" if result.diagnostics.residual_gate is not None else "—")
        second.metric("World effective count", f"{result.diagnostics.world_effective_count:.3f}" if result.diagnostics.world_effective_count is not None else "—")
        third.metric("Prior mismatch", f"{result.diagnostics.prior_mismatch_probability:.3f}" if result.diagnostics.prior_mismatch_probability is not None else "—")
        with st.expander("Full model diagnostics"):
            st.json(result.diagnostics.__dict__)
        st.caption("Graph logits are exploratory and are not a validated causal-graph estimator for this checkpoint.")
    provenance_panel.render(result)
    export_panel.render(result, command)

