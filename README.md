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

- **Static ATE** — average treatment effects on tabular data
- **Observed regimes** — detecting shifts in causal mechanisms
- **Network interference** — effects with spillovers on graphs

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

## Qualitative benchmark examples

```bash
python -m qualitative_evaluation \
  --project-root . \
  --output-dir qualitative_evaluation_outputs
```
