"""Convenience entry point: python qualitative_evaluation/run_all.py ..."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualitative_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
