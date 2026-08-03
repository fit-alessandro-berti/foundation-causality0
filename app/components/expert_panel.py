"""Shared compiled-estimator routing panel."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cwfm.plotting import expert_frame


EXPERT_HELP = {
    "linear": "Low-variance linear response model",
    "orthogonal": "Cross-fitted nuisance-adjusted estimate",
    "spline": "Smooth nonlinear response",
    "interaction": "Treatment or exposure heterogeneity",
    "piecewise": "Hinge or threshold-like behavior",
    "design_or_null": "Randomized estimate or null shrinkage, depending on task",
}


def render(result) -> None:
    if not result.expert_results:
        return
    st.subheader("Expert comparison and routing")
    frame = expert_frame(result)
    frame = frame[frame["expert"].isin(EXPERT_HELP)].copy()
    frame["interpretation"] = frame["expert"].map(EXPERT_HELP)
    left, right = st.columns([3, 2])
    left.dataframe(frame, use_container_width=True, hide_index=True)
    right.bar_chart(frame.set_index("expert")["weight"], horizontal=True)
    st.caption(f"Selected expert: {result.selected_expert or 'not available'}")
