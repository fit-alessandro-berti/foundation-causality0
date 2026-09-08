# Foundation Causality

Tools and experiments for causal structure discovery and a reference **Causal World Foundation Model (CWFM)**.

## What's in this repo

| Path | Description |
|------|-------------|
| `cwfm/` | PyTorch CWFM model and public application API |
| `app/` | Streamlit UI for analysis and diagnostics |
| `examples/` | Runnable scripts for common workflows |
| `artifacts/cwfm/` | Trained checkpoint, release bundle, and experiment results |
| `docs/data/benchmark/` | Six classical synthetic causal-structure benchmarks |
| `docs/foundation_model/` | Design notes and full experiment report |
| `qualitative_evaluation/` | Small worked examples from benchmark artifacts |

## Quick start

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/00_check_installation.py
streamlit run app/streamlit_app.py
```

Try the examples (add `--dry-run` to validate without full inference):

```bash
python examples/01_static_ate.py
python examples/02_observed_regimes.py
python examples/03_network_interference.py
python examples/04_safety_and_abstention.py
```

See `examples/README.md` for the full list and options.

## What the model covers

The reference CWFM implements three tasks:

- **Static ATE**: average treatment effects on tabular data
- **Observed regimes**: detecting shifts in causal mechanisms
- **Network interference**: effects with spillovers on graphs

It is **not** the full proposed system (high-dimensional latents, full temporal modeling, partial identification). That distinction is documented in `docs/foundation_model/`.

Inference uses the release bundle at `artifacts/cwfm/release/`. If weights are missing or fail integrity checks, tools still browse cases and return a typed `MODEL_UNAVAILABLE` status. Ordinary analysis never reads simulator truth (`truth.npz`).

## Reproduce training and evaluation

```bash
python -m cwfm.train \
  --steps 1600 \
  --batch-size 24 \
  --validation-interval 100 \
  --output artifacts/cwfm/checkpoint.pt

python -m cwfm.experiment \
  --checkpoint artifacts/cwfm/checkpoint.pt \
  --output-dir artifacts/cwfm/results \
  --seeds 30 \
  --calibration-episodes 360
```

Full methodology, seeds, and results: [`docs/foundation_model/experimental_results.md`](docs/foundation_model/experimental_results.md).

The confirmatory comparison favors the shallow risk selector and convex stack
over the frozen neural router. Gate compliance is conditional on the declared
assumptions; it does not verify their truth. The empirical support screen can
both reject adequate samples and accept poor overlap under nonlinear assignment.
The release checkpoint and calibration remain unchanged.

The reviewer revision adds a small, CPU-only audit of the actual application
support functions, with no model training or threshold tuning:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m cwfm.support_experiment
```

Its fixed protocol is `experiments/support_protocol.json`; complete rates,
pointwise Wilson intervals, per-seed outcomes, and provenance are in
`artifacts/cwfm/support_revision/`. The full 40,000-episode run took about ten
seconds in the revision environment. See [`RESPONSE.txt`](RESPONSE.txt) for the
point-by-point reviewer response and [`paper/README.md`](paper/README.md) for
manuscript builds and figure regeneration.

## Qualitative benchmark examples

```bash
python -m qualitative_evaluation \
  --project-root . \
  --output-dir qualitative_evaluation_outputs
```
