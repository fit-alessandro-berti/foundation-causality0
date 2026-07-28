# Foundation Causality

This repository contains:

- six classical synthetic causal-structure benchmarks under `docs/data/benchmark`;
- the design of the Causal World Foundation Model (CWFM) under
  `docs/foundation_model`;
- a runnable PyTorch CWFM reference implementation under `cwfm`;
- a frozen checkpoint and reproducible held-out comparison under
  `artifacts/cwfm`;
- the complete CWFM experiment report in
  `docs/foundation_model/experimental_results.md`.

## Reproduce the CWFM experiment

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m cwfm.train \
  --steps 1600 \
  --batch-size 24 \
  --output artifacts/cwfm/checkpoint.pt
python -m cwfm.experiment \
  --checkpoint artifacts/cwfm/checkpoint.pt \
  --output-dir artifacts/cwfm/results \
  --seeds 30 \
  --calibration-episodes 360
```

The committed final run uses pretraining seed `1729`, calibration seeds
beginning at `19,000,000`, and final evaluation seeds beginning at
`20,000,000`. The result manifest records package versions, runtime,
configuration, calibration radii, and the checkpoint SHA-256.

## Scope

The executable reference model implements the first scope recommended by the
architecture document: static tabular effects, observed-regime determination,
and network interference. It is not presented as the full proposed
384-dimensional, temporal, latent-variable, partial-identification system.
The distinction between implemented features and future architecture is
explicit in the architecture and experimental report.
