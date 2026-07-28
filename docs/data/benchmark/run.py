from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .common import process_memory_mb, write_json


METHODS = {
    "causal_model_determination": "causal_model",
    "latent_variable_determination": "latent_variable",
    "latent_variable_determination2": "latent_graph",
    "temporal_split_detection": "temporal_split",
    "temporal_split_detection_method": "temporal_varx",
}


def _run_internal(args: argparse.Namespace) -> int:
    data_root = args.output_root / "generated"
    metrics_root = args.output_root / "metrics"
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    started = time.time()
    status_path = metrics_root / args.method / "run_status.json"
    write_json(
        status_path,
        {
            "method": args.method,
            "status": "running",
            "started_unix": started,
            "seeds": seeds,
            "calibration_reps": args.calibration_reps,
            "stability_reps": args.stability_reps,
            "benchmark_version": __version__,
        },
    )
    try:
        if args.method == "causal_model_determination":
            from .causal_model import run

            records = run(
                data_root,
                metrics_root,
                seeds,
                args.calibration_reps,
                args.stability_reps,
            )
        elif args.method == "latent_variable_determination":
            from .latent_variable import run

            records = run(data_root, metrics_root, seeds, args.stability_reps)
        elif args.method == "latent_variable_determination2":
            from .latent_graph import run

            records = run(data_root, metrics_root, seeds, args.stability_reps)
        elif args.method == "temporal_split_detection":
            from .temporal_split import run

            records = run(data_root, metrics_root, seeds, args.calibration_reps)
        elif args.method == "temporal_split_detection_method":
            from .temporal_varx import run

            records = run(data_root, metrics_root, seeds, args.calibration_reps)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        failed = sum(record.get("status") != "ok" for record in records)
        final_status = "completed" if failed == 0 else "completed_with_failures"
        exit_code = 0 if failed == 0 else 2
        write_json(
            status_path,
            {
                "method": args.method,
                "status": final_status,
                "started_unix": started,
                "finished_unix": time.time(),
                "runtime_seconds": time.time() - started,
                "n_records": len(records),
                "n_success": len(records) - failed,
                "n_failed": failed,
                "seeds": seeds,
                "calibration_reps": args.calibration_reps,
                "stability_reps": args.stability_reps,
                "peak_memory_mb": process_memory_mb(),
                "benchmark_version": __version__,
            },
        )
        return exit_code
    except Exception as exc:
        write_json(
            status_path,
            {
                "method": args.method,
                "status": "failed",
                "started_unix": started,
                "finished_unix": time.time(),
                "runtime_seconds": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "benchmark_version": __version__,
            },
        )
        raise


def _run_all(args: argparse.Namespace) -> int:
    metrics_root = args.output_root / "metrics"
    logs_root = metrics_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    overall_started = time.time()
    for method in METHODS:
        command = [
            sys.executable,
            "-m",
            "docs.data.benchmark.run",
            "--method",
            method,
            "--output-root",
            str(args.output_root),
            "--seeds",
            str(args.seeds),
            "--seed-start",
            str(args.seed_start),
            "--calibration-reps",
            str(args.calibration_reps),
            "--stability-reps",
            str(args.stability_reps),
            "--internal",
        ]
        started = time.time()
        status_path = metrics_root / method / "run_status.json"
        try:
            completed = subprocess.run(
                command,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLCONFIGDIR": str(args.output_root / ".matplotlib"),
                },
                check=False,
            )
            (logs_root / f"{method}.stdout.log").write_text(completed.stdout)
            (logs_root / f"{method}.stderr.log").write_text(completed.stderr)
            status = {
                "method": method,
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "runtime_seconds": time.time() - started,
            }
            if completed.returncode != 0 and not status_path.exists():
                write_json(status_path, status)
        except subprocess.TimeoutExpired as exc:
            (logs_root / f"{method}.stdout.log").write_text(exc.stdout or "")
            (logs_root / f"{method}.stderr.log").write_text(exc.stderr or "")
            status = {
                "method": method,
                "status": "timed_out",
                "timeout_seconds": args.timeout_seconds,
                "runtime_seconds": time.time() - started,
                "note": "The component process was killed at the requested hard limit; partial per-seed artifacts remain.",
            }
            write_json(status_path, status)
        statuses.append(status)
        write_json(
            metrics_root / "run_manifest.json",
            {
                "status": "running",
                "started_unix": overall_started,
                "timeout_seconds_per_component": args.timeout_seconds,
                "components": statuses,
                "benchmark_version": __version__,
            },
        )
    failed = [status for status in statuses if status["status"] != "completed"]
    write_json(
        metrics_root / "run_manifest.json",
        {
            "status": "completed" if not failed else "completed_with_failures",
            "started_unix": overall_started,
            "finished_unix": time.time(),
            "runtime_seconds": time.time() - overall_started,
            "timeout_seconds_per_component": args.timeout_seconds,
            "components": statuses,
            "benchmark_version": __version__,
        },
    )
    return 0 if not failed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["all", *METHODS],
        default="all",
        help="Benchmark component to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data"),
        help="Parent for generated/ and metrics/.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--calibration-reps",
        type=int,
        default=19,
        help="Full-search permutation/bootstrap replicates; 19 permits a 0.05 Monte Carlo p-value.",
    )
    parser.add_argument(
        "--stability-reps",
        type=int,
        default=5,
        help="Conditional bootstrap replicates for stability summaries.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1200,
        help="Hard subprocess limit per component when --method all.",
    )
    parser.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.calibration_reps < 1:
        parser.error("--calibration-reps must be positive")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if args.method == "all" and not args.internal:
        return _run_all(args)
    if args.method == "all":
        raise ValueError("--internal cannot be combined with --method all")
    return _run_internal(args)


if __name__ == "__main__":
    raise SystemExit(main())
