"""CWFM Streamlit entry point using explicit navigation."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pages import analyze, benchmark_atlas, compare, home, model_diagnostics  # noqa: E402


st.set_page_config(
    page_title="CWFM Application",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    {
        "CWFM": [
            st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
            st.Page(analyze.render, title="Analyze", icon=":material/query_stats:", url_path="analyze"),
            st.Page(compare.render, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
        ],
        "Research": [
            st.Page(
                benchmark_atlas.render,
                title="Benchmark Atlas",
                icon=":material/experiment:",
                url_path="benchmark-atlas",
            ),
            st.Page(
                model_diagnostics.render,
                title="Model diagnostics",
                icon=":material/monitoring:",
                url_path="model-diagnostics",
            ),
        ],
    }
)
navigation.run()
