# Pretraining ablation and router diagnostics

This post-review analysis addresses the corrected Reviewer 1 comments. It reuses
the existing confirmatory effect bank explicitly, with 300 episodes in each of
six scenarios. It does not constitute a newly independent confirmatory study.

Run from the repository root:

```bash
python -m cwfm.pretraining_experiment
```

The protocol is `experiments/pretraining_protocol.json`. It was specified before
the comparison, following a timing-only pilot on a disjoint 32-episode stream.
The recorded run took 126.7 seconds on an NVIDIA L40S with batches of 32 and four
CPU threads. `--device cpu` is also supported, with environment-dependent runtime
and small numerical differences. No neural training is performed.

The pretrained checkpoint is compared with five untrained initializations of the
same architecture, uniform compatible weights, and pretrained weights averaged
on the separate 1,200-episode tuning stream. The fixed averages match task and
compatibility mask. Candidate fits, observed inputs, masks, and declared gate are
shared. Targets are excluded from forward inputs. Residual effects are below
2.1e-7, and neural states are verified unchanged. All 1,800 test episodes are
answered by every control.

Pretraining reduces pooled mean absolute error from 0.1673, averaged across the
untrained initializations, to 0.1612. The paired reduction is 0.0061, with a 95%
interval of [0.0047, 0.0076]. This averages **errors**, not predictions from an
ensemble of untrained models. Uniform aggregation gives 0.1683; fixed mean
pretrained weights give 0.1621. The learned weights express broad candidate
preferences, with a smaller benefit from adapting them to individual datasets.
Threshold interference favors the untrained and uniform controls. The stronger
shallow-policy findings from the main comparison remain unchanged.

Bootstrap intervals resample paired episodes within scenarios and condition on
the single trained checkpoint and five fixed initializations. The random controls
are not multiple training runs. This analysis evaluates point estimates and does
not estimate new interval coverage or establish training stability.

| File | Contents |
|---|---|
| `per_seed.csv` | Predictions, truth, error, and acceptance for eight methods on all 1,800 episodes |
| `paired_errors.csv` | Matched errors and the mean error across untrained initializations |
| `summary.csv` | Every method and initialization, by scenario and pooled |
| `paired_comparisons.json` | Stratified paired bootstrap differences and intervals |
| `router_weights.csv` | All 10,800 neural predictions' weights, candidates, masks, normalized entropy, and oracle top-choice indicator |
| `router_summary.csv` | Scenario and initialization averages of the diagnostics |
| `router_weight_variation.csv` | Trained weight standard deviations within task/mask groups |
| `acceptance.csv` | Confirmatory acceptance counts, with all 300 episodes accepted per scenario |
| `development_episodes.csv` | All 36 saved step-100 development records, including 24 effect and 12 regime episodes |
| `development_effect_errors.csv` | The four individual effect errors for each of the six RQ4 families |
| `manifest.json` | Protocol, source and CSV hashes, environment, model size, training costs, phase timings, and checks |
| `verification.json` | Additional artifact, document-build, response-synchronization, and preservation checks |

The development estimates are exported from the original saved diagnostics,
without recomputing estimates after the estimator-layer repair. Seed columns are
reconstructed from the saved training seed and `validate()` generation order;
all 24 effect targets were checked against those seeds. These observations were
used in checkpoint selection and are not an independent generalization test.

The original checkpoint, training logs, confirmatory results, application logic,
support thresholds, and released calibration are preserved. The rerun reproduces
the original pretrained estimates within 4.4e-7 and uniform estimates within
4.5e-16. The manifest reports 917,040 full-model parameters, including exploratory
heads, and distinguishes the original 33.3-minute training run from its selected
step-100 snapshot.
