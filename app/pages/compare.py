"""Compare completed analyses without refitting them."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    st.title("Compare and sensitivity")
    results = st.session_state.get("comparison_results", [])
    if not results:
        st.info("Run analyses and add them to the comparison set from the Analyze page.")
        return
    rows = [
        {
            "analysis_id": result.analysis_id,
            "task": result.task.value,
            "status": result.status.value,
            "estimate": result.estimate,
            "lower": result.interval[0] if result.interval else None,
            "upper": result.interval[1] if result.interval else None,
            "selected_expert": result.selected_expert,
            "residual_gate": result.diagnostics.residual_gate,
        }
        for result in results
    ]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    effect_frame = frame.dropna(subset=["estimate"])
    if not effect_frame.empty:
        st.subheader("Effect estimates")
        st.bar_chart(effect_frame.set_index("analysis_id")["estimate"])
    if st.button("Clear comparison set"):
        st.session_state["comparison_results"] = []
        st.rerun()

