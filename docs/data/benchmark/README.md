# Synthetic data benchmarks

This package implements the generators and evaluations specified by the six
proposal documents in `docs/data`.

Run the recorded evaluation:

```bash
python -m docs.data.benchmark.run \
  --method all \
  --seeds 5 \
  --calibration-reps 19 \
  --stability-reps 5 \
  --inference-reps 29 \
  --timeout-seconds 1200
```

Each of the six components runs in a separate subprocess. The parent kills a
component after 1,200 seconds (20 minutes), records `timed_out` in its
`run_status.json`, and retains any per-seed data and metrics written before the
timeout.

Artifacts use this layout:

```text
docs/data/
├── generated/<method>/<scenario>/seed_NNNN/
│   ├── discovery.npz       observed discovery/training arrays
│   ├── evaluation.npz      independent observed evaluation arrays
│   ├── truth.npz           evaluator-only generator truth
│   └── metadata.json       roles, names, scenario, and seed
└── metrics/<method>/
    ├── per_seed.csv
    ├── per_seed.jsonl
    ├── summary.csv
    ├── summary.json
    ├── manifest.json
    └── run_status.json
```

The default five-seed run is an executable engineering evaluation, not a
publication-sized Monte Carlo study. Increase `--seeds` to 200 or more for
stable rate estimates. The calibration replicate count should be at least 19;
using fewer replicates cannot produce a Monte Carlo p-value at or below 0.05.

All preprocessing is fitted on discovery data and applied unchanged to the
evaluation data. Truth arrays are never supplied to fitters; they are loaded
only by evaluator functions.
