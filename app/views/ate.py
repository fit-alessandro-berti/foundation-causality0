"""Static ATE result view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import common


def render(result, command: str, research_mode: bool) -> None:
    common.start(result)
    if result.estimate is not None:
        estimate, lower, upper = result.estimate, result.interval[0], result.interval[1]
        first, second = st.columns(2)
        first.metric("Average effect", f"{estimate:.4g}")
        second.metric("90% calibrated interval", f"[{lower:.4g}, {upper:.4g}]")
    overlap = result.diagnostics.preprocessing.get("treatment_overlap")
    counts = result.diagnostics.preprocessing.get("treatment_counts")
    if counts or overlap:
        st.subheader("Treatment support")
        if counts:
            st.dataframe(pd.DataFrame([counts], index=["count"]), use_container_width=True)
        if overlap:
            st.json(overlap)
    common.finish(result, command, research_mode)

