# CWFM application examples

The numbered scripts are thin clients of `cwfm.application`. They never import
private experiment helpers and never read `truth.npz` during ordinary analysis.

```bash
python examples/00_check_installation.py
python examples/01_static_ate.py
python examples/02_observed_regimes.py
python examples/03_network_interference.py
python examples/04_safety_and_abstention.py
python examples/05_compare_repository_cases.py
python examples/06_model_diagnostics.py
python examples/07_reference_benchmark_atlas.py
```

All analysis scripts support `--project-root`, `--device`, `--output-dir`,
`--seed`, `--dry-run`, `--json`, and `--quiet`. The committed archive does not
contain `artifacts/cwfm/release/model_state.pt`, so answerable analyses report a
typed `MODEL_UNAVAILABLE` status until matching weights are installed. Catalog
preview and dry-run validation remain available.

Static ATE examples are deterministic model-native fixture specifications. The
catalog materializes observed `X`, `A`, and `Y` arrays while discarding simulator
truth. Regime and interference examples read repository `discovery.npz` files
through method-specific metadata allowlists.
