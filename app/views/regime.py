"""Observed-regime result view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cwfm.plotting import regime_evidence_frame

from . import common


def render(result, command: str, research_mode: bool) -> None:
    common.start(result)
    regime = result.regime_result
    if regime:
        first, second, third = st.columns(3)
        first.metric("Calibrated decision", "Split" if regime.split_detected else "No split")
        second.metric("Evidence score", f"{regime.evidence_score:.4g}")
        third.metric("Calibration threshold", f"{regime.calibrated_threshold:.4g}")
        if regime.split_detected:
            st.write(
                f"Selected variable **{regime.selected_variable}** at threshold "
                f"**{regime.selected_threshold:.4g}** ({regime.left_count} left / {regime.right_count} right)."
            )
        if regime.candidate_screening:
            st.subheader("Discovery-only candidate screening")
            st.caption(
                f"{len(regime.candidate_screening)} candidates were screened; the selected subset was passed to CWFM."
            )
            st.dataframe(pd.DataFrame(regime.candidate_screening), use_container_width=True, hide_index=True)
        if regime.evidence_table:
            frame = regime_evidence_frame(result)
            st.subheader("Feature-by-threshold evidence")
            heatmap = frame.pivot(index="variable", columns="quantile", values="evidence_score")
            st.dataframe(heatmap.style.background_gradient(cmap="Blues"), use_container_width=True)
        st.caption("The raw neural split probability is a research diagnostic; the principal decision uses the held-out calibrated evidence threshold.")
    common.finish(result, command, research_mode)
