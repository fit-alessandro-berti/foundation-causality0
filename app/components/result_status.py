"""Common status-first result header."""

from __future__ import annotations

import streamlit as st

from cwfm.application import AnalysisStatus, Task


def render(result) -> None:
    label = result.status.value.replace("_", " ")
    if result.status.answered:
        st.success(label)
        if result.task != Task.OBSERVED_REGIME:
            st.caption(
                "This answer is conditional on your declared assumptions. Passing the checks "
                "does not verify those assumptions or establish population overlap."
            )
    elif result.status in {
        AnalysisStatus.MODEL_UNAVAILABLE,
        AnalysisStatus.ABSTAINED_IDENTIFICATION_NOT_ESTABLISHED,
        AnalysisStatus.ABSTAINED_INADEQUATE_SUPPORT,
    }:
        st.warning(label)
    else:
        st.error(label)
    st.subheader(result.question)
    st.caption(result.estimand)
    for reason in result.decision_reasons:
        st.write(f"• {reason}")
