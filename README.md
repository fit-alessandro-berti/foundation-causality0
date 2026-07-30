# Foundation Causality

This repository contains:

- six classical synthetic causal-structure benchmarks under `docs/data/benchmark`;
- a qualitative evaluation package under `qualitative_evaluation` that produces
  small worked examples from the persisted benchmark artifacts (run with
  `python -m qualitative_evaluation --project-root . --output-dir qualitative_evaluation_outputs`);
- the design of the Causal World Foundation Model (CWFM) under
  `docs/foundation_model`;
- a runnable PyTorch CWFM reference implementation under `cwfm`;
- a revised, constraint-selected checkpoint and reproducible held-out comparison under
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
  --validation-interval 100 \
  --output artifacts/cwfm/checkpoint.pt
python -m cwfm.experiment \
  --checkpoint artifacts/cwfm/checkpoint.pt \
  --output-dir artifacts/cwfm/results \
  --seeds 30 \
  --calibration-episodes 360
```

The committed revised run uses pretraining seed `1729`, separate ID and OOD
development banks, calibration seeds
beginning at `19,000,000`, and final evaluation seeds beginning at
`20,000,000`. The result manifest records package versions, runtime,
configuration, finite-sample conformal radii, estimator routing, paired
comparisons, and the checkpoint SHA-256. Append-only training and grouped
validation logs are stored beside the checkpoint.

## Scope

The executable reference model implements the first scope recommended by the
architecture document: static tabular effects, observed-regime determination,
and network interference. It is not presented as the full proposed
384-dimensional, temporal, latent-variable, partial-identification system.
The distinction between implemented features and future architecture is
explicit in the architecture and experimental report.
