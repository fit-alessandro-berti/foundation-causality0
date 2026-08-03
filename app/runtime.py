"""Cached repository and model resources shared across Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cwfm.application import CWFMRunner, RepositoryCatalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False)
def catalog() -> RepositoryCatalog:
    return RepositoryCatalog.from_project_root(PROJECT_ROOT)


@st.cache_resource(show_spinner="Loading CWFM model bundle…")
def runner(device: str = "auto") -> CWFMRunner:
    return CWFMRunner.from_project_root(PROJECT_ROOT, device=device)

