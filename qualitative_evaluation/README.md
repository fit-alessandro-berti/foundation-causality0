# Qualitative Evaluation Package

This folder is designed to be copied directly into the root of the
`foundation-causality-main` project.

It generates small, paper-ready examples that compare model outputs with
evaluator-only ground truth:

- observed-regime tree, boundary rows, matched probes, and coefficient changes;
- correlated-proxy visualization;
- correct versus misspecified nonlinear local models;
- outcome-relevant latent-group assignments;
- Level A/B/C latent-graph reconstruction;
- temporal coefficient break versus intercept-only control;
- VARX B-change, A-only control, and an honest false-B failure case.

## Run

From the project root:

```bash
python -m qualitative_evaluation \
  --project-root . \
  --output-dir qualitative_evaluation_outputs
```

The package reads the persisted artifacts under:

```text
docs/data/generated/<method>/<scenario>/seed_NNNN/
```

This ensures the qualitative examples correspond to the same data as the
reported benchmark metrics. Fitting uses only `discovery.npz`; `truth.npz` is
used only after fitting for comparison and annotation.

## Seed selection

Success examples use the seed whose designated primary metric is closest to the
five-run median. Ties are resolved deterministically. The VARX failure example
uses the smallest seed with `false_B_attribution == 1`. All selected seeds are
written to `seed_manifest.json`.

## Outputs

Each case directory contains:

- PNG figures;
- small examples in CSV, Markdown, and LaTeX;
- JSON summaries of learned and true structures;
- parameter-comparison tables.

The root output directory also contains:

- `QUALITATIVE_EVALUATION.md`;
- `PAPER_SECTION_DRAFT.md`;
- `FIGURE_CAPTIONS.md`;
- `manifest.json`.

## Dependencies

The package uses the same scientific Python stack as the benchmark:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
networkx
tabulate
```

Several benchmark helpers currently have private names beginning with `_`.
This package imports those helpers intentionally so that the qualitative
computation exactly mirrors the existing evaluators. A future cleanup can
expose public fitting functions without changing these analyses.
