"""Shared presentation utilities for the thin CWFM example clients."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cwfm.application import AnalysisResult, AnalysisStatus, export_analysis  # noqa: E402


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--quiet", action="store_true")


def print_result(result: AnalysisResult, as_json: bool = False, quiet: bool = False) -> None:
    if quiet:
        return
    if as_json:
        print(result.to_json())
        return
    print("QUESTION")
    print(result.question)
    print("\nASSUMPTIONS")
    for key, value in result.provenance.assumptions.items():
        print(f"{key.replace('_', ' ').title()}: {getattr(value, 'value', value)}")
    print("\nDECISION")
    print(result.status.value.replace("_", " ").title())
    for reason in result.decision_reasons:
        print(f"- {reason}")
    if result.estimate is not None:
        print("\nRESULT")
        print(f"Estimate: {result.estimate:.6g}")
        if result.interval:
            level = 100 * (result.interval_level or 0)
            print(f"{level:.0f}% calibrated interval: [{result.interval[0]:.6g}, {result.interval[1]:.6g}]")
    if result.regime_result:
        regime = result.regime_result
        print("\nREGIME DECISION")
        print(f"Evidence score: {regime.evidence_score:.6g}")
        print(f"Calibrated threshold: {regime.calibrated_threshold:.6g}")
        print(f"Split detected: {'yes' if regime.split_detected else 'no'}")
        if regime.split_detected:
            print(f"Variable: {regime.selected_variable}")
            print(f"Threshold: {regime.selected_threshold:.6g}")
    if result.selected_expert:
        print("\nROUTING")
        print(f"Selected expert: {result.selected_expert}")
    if result.warnings:
        print("\nWARNINGS")
        for warning in result.warnings:
            print(f"- {warning}")


def finish(
    result: AnalysisResult,
    args: argparse.Namespace,
    default_output: Path,
    command: str,
) -> int:
    print_result(result, args.as_json, args.quiet)
    output = args.output_dir or default_output
    target = export_analysis(result, output, command)
    if not args.quiet and not args.as_json:
        print(f"\nSaved analysis: {target}")
    return 2 if result.status in {
        AnalysisStatus.MODEL_UNAVAILABLE,
        AnalysisStatus.INVALID_INPUT,
        AnalysisStatus.UNSUPPORTED_QUERY,
        AnalysisStatus.UNSUPPORTED_TASK,
    } else 0


def print_dry_run(payload: dict, as_json: bool, quiet: bool) -> None:
    if quiet:
        return
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("DRY RUN — no model inference performed")
        print(json.dumps(payload, indent=2, sort_keys=True))


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
