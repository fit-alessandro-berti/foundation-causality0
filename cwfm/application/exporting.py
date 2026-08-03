"""Shared result exports for scripts and the graphical interface."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .results import AnalysisResult, _jsonable


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def analysis_files(
    result: AnalysisResult,
    reproduction_command: str = "",
) -> dict[str, str]:
    summary = [
        {
            "analysis_id": result.analysis_id,
            "task": result.task.value,
            "status": result.status.value,
            "estimate": result.estimate,
            "lower": result.interval[0] if result.interval else None,
            "upper": result.interval[1] if result.interval else None,
            "selected_expert": result.selected_expert,
        }
    ]
    experts = [_jsonable(asdict(item)) for item in result.expert_results]
    files = {
        "result.json": result.to_json() + "\n",
        "summary.csv": _csv_text(summary),
        "expert_results.csv": _csv_text(experts),
        "assumptions.json": json.dumps(
            _jsonable(asdict(result.assumptions)), indent=2, sort_keys=True
        )
        + "\n",
        "diagnostics.json": json.dumps(
            _jsonable(asdict(result.diagnostics)), indent=2, sort_keys=True
        )
        + "\n",
        "provenance.json": json.dumps(
            _jsonable(asdict(result.provenance)), indent=2, sort_keys=True
        )
        + "\n",
        "input_schema.json": json.dumps(
            {
                "selected_variables": result.provenance.selected_variables,
                "task": result.task.value,
                "query": _jsonable(result.provenance.query),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "reproduction_command.txt": reproduction_command.rstrip() + "\n",
    }
    if result.regime_result:
        files["candidate_screening.csv"] = _csv_text(
            result.regime_result.candidate_screening
        )
        files["threshold_evidence.csv"] = _csv_text(
            result.regime_result.evidence_table
        )
    if result.interference_result:
        files["network_summary.json"] = json.dumps(
            _jsonable(asdict(result.interference_result)), indent=2, sort_keys=True
        ) + "\n"
    return files


def export_analysis(
    result: AnalysisResult,
    output_dir: str | Path,
    reproduction_command: str = "",
) -> Path:
    target = Path(output_dir) / result.analysis_id
    target.mkdir(parents=True, exist_ok=True)
    for name, content in analysis_files(result, reproduction_command).items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return target


def analysis_zip(
    result: AnalysisResult,
    reproduction_command: str = "",
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in analysis_files(result, reproduction_command).items():
            archive.writestr(name, content)
    return buffer.getvalue()
