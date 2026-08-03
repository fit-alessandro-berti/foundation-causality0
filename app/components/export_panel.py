"""Downloadable analysis bundle and reproduction snippets."""

from __future__ import annotations

import streamlit as st

from cwfm.application import analysis_zip


def render(result, command: str = "") -> None:
    st.subheader("Reproduce and export")
    st.write("Equivalent command")
    st.code(command or "# Configure an analysis above to generate a command", language="bash")
    query = result.provenance.query
    st.write("Minimal Python equivalent")
    st.code(
        "\n".join(
            [
                "from cwfm.application import CWFMRunner, RepositoryCatalog",
                "catalog = RepositoryCatalog.from_project_root('.')",
                "runner = CWFMRunner.from_project_root('.')",
                f"# Query configuration: {query!r}",
                "# result = runner.analyze(case, query, assumptions)",
            ]
        ),
        language="python",
    )
    st.download_button(
        "Download analysis bundle",
        data=analysis_zip(result, command),
        file_name=f"analysis_{result.analysis_id}.zip",
        mime="application/zip",
    )

