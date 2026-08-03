"""Network-interference result view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cwfm.plotting import exposure_support_frame

from . import common


def render(result, command: str, research_mode: bool) -> None:
    common.start(result)
    network = result.interference_result
    if network:
        first, second = st.columns(2)
        first.metric("Mapped exposure contrast", f"{network.exposure_low:g} → {network.exposure_high:g}")
        second.metric("Mapping", network.exposure_mapping)
        if result.estimate is not None:
            third, fourth = st.columns(2)
            third.metric("Average spillover contrast", f"{result.estimate:.4g}")
            fourth.metric("90% calibrated interval", f"[{result.interval[0]:.4g}, {result.interval[1]:.4g}]")
        if network.support:
            st.subheader("Exposure support")
            st.json(network.support)
            groups = exposure_support_frame(result)
            if not groups.empty:
                st.dataframe(groups, use_container_width=True, hide_index=True)
    graph = result.diagnostics.preprocessing
    if graph:
        st.subheader("Observed network summary")
        keys = [key for key in graph if key.startswith("graph_") or key.startswith("degree_") or key == "isolated_units"]
        st.json({key: graph[key] for key in keys})
    common.finish(result, command, research_mode)
