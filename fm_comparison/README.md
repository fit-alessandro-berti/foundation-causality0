# Comparison of the uploaded CWFM implementation with causality foundation models

## Bottom line

The current open-source CWFM result is **ATE MAE = 0.166** on its own evaluation setting. This is in the same broad `10^-1` error regime as published causal-effect foundation models, but the public papers do not provide a legitimate single leaderboard on the same test distribution, outcome scaling, estimand, and metric.

The defensible paper claim is therefore:

> On its native evaluation, the open-source CWFM implementation obtains an ATE MAE of 0.166. Cross-paper comparison places this result in the same broad error regime as published foundation models for treatment-effect and interventional-query estimation. However, because CausalPFN and Do-PFN use different benchmark distributions and CausalFM primarily reports PEHE for heterogeneous effects, these comparisons indicate competitiveness rather than a head-to-head state-of-the-art result.

## Recommended main table

| Model | Causal target | Native result used | Harmonized view | Evidence status | Valid conclusion |
|---|---|---:|---:|---|---|
| **CWFM (ours, current OSS)** | Population ATE | **MAE 0.166** | **ATE-MAE 0.166** | Exact local result | Reference |
| CausalPFN | ATE/CATE, depending experiment | Benchmark-specific effect-error values | **~0.08–0.22 ATE-MAE** | Cross-benchmark interpolation | Overlaps CWFM; no ranking |
| Do-PFN | Interventional mean / ATE | Benchmark-specific RMSE or related effect error | **~0.10–0.25 ATE-MAE** | Cross-benchmark interpolation | Overlaps CWFM; no ranking |
| CausalFM | CATE/ITE | **PEHE 0.515** | ITE-MAE proxy **0.361–0.464**; absolute ATE-error bound **0–0.515** | Formula-derived, not a rerun | Harder target; no ranking |
| ARROW / TabCausal / CDFM | Graph recovery / edge orientation | SHD, F1, AUROC | No valid conversion | Capability-only | Do not mix with effect error |

### How to label the rows

- **Exact local result:** directly found in the uploaded repository/manuscript artifacts.
- **Published native result:** copied from the comparator paper without changing metric or benchmark.
- **Derived:** deterministic formula applied to a published score.
- **Interpolated:** an intentionally broad envelope used only to discuss order of magnitude across different benchmarks.

## Why `0.166 < 0.515` is not a valid superiority claim

CWFM's 0.166 is a population-level **ATE MAE**. CausalFM's 0.515 is **PEHE**, the root mean squared error over individual or conditional treatment effects. PEHE penalizes heterogeneity errors that can cancel when effects are averaged into an ATE. Moreover, the outcome scales and test distributions differ. A direct numerical inequality would therefore reward the easier estimand and potentially a different normalization.

For errors `e_i = tau_hat_i - tau_i`,

`|mean(e_i)| <= sqrt(mean(e_i^2)) = PEHE`.

Thus 0.515 only yields the loose same-sample bound `ATE absolute error <= 0.515`; it does not predict CausalFM's ATE MAE on CWFM's benchmark. Under a symmetric Gaussian error approximation, PEHE 0.515 corresponds to an individual-effect MAE of `sqrt(2/pi) * 0.515 = 0.411`. A sensitivity interval using ratios 0.70–0.90 is 0.361–0.464. This remains an ITE/CATE quantity, not ATE error.

## Suggested manuscript wording

### Results paragraph

> The current open-source implementation achieves an ATE MAE of 0.166 on the locked evaluation set. We contextualize this value against foundation models for causal effect estimation using their published results, without rerunning their implementations. The closest systems—CausalPFN and Do-PFN—report benchmark-dependent errors for treatment effects or interventional queries. After retaining only comparable effect-error experiments and applying a transparent RMSE-to-MAE sensitivity conversion where required, their published results occupy an indicative low-to-mid `10^-1` range that overlaps our result. CausalFM reports a PEHE of 0.515 for heterogeneous effects; because PEHE is an individual-effect RMSE rather than an ATE MAE, it is included only as a related-task reference. These results support the claim that CWFM is competitive in order of magnitude, but they do not constitute a common-benchmark leaderboard.

### Threats-to-validity paragraph

> Cross-paper comparisons are affected by differences in the causal estimand, data-generating process, sample size, confounding strength, overlap, outcome normalization, and aggregation protocol. Derived values are used only for scale contextualization: for an RMSE `r`, we report a Gaussian MAE point approximation `sqrt(2/pi)r` and a sensitivity band `[0.70r, 0.90r]`. We do not convert causal-discovery metrics such as SHD or edge F1 into treatment-effect error. All inferred values are visually distinguished from exact published or repository-derived numbers.

## Strong and weak claims

**Supported:**

- CWFM's native ATE error is in the same broad order of magnitude as published causal-effect foundation models.
- CWFM provides a competitive open-source result under its own evaluation protocol.
- The comparison covers both effect-estimation FMs and, separately, causal-discovery FMs in a capability matrix.

**Not supported without rerunning comparators:**

- “CWFM outperforms CausalFM because 0.166 is lower than 0.515.”
- “CWFM is state of the art across causal foundation models.”
- Any ranking that mixes ATE MAE, PEHE, SHD, F1, and AUROC.

## Reproducibility files

- `comparison.csv`: editable comparison table with status labels.
- `scripts/harmonize_metrics.py`: conversions and sensitivity analysis.
- `scripts/audit_local_results.py`: searches result artifacts and recomputes MAE/RMSE/normalized errors when true/predicted columns are available.
- `evidence/local_audit.*`: automatically detected result artifacts from the uploaded repository.
- `evidence/published_self_rows.*`: mechanically extracted candidate result rows from primary-paper PDFs; check table headers before copying any number into the manuscript.

## Recommended next paper revision

Use two tables rather than one:

1. **Effect-estimation comparison**, containing only ATE/CATE/interventional-query models and explicit metric/benchmark columns.
2. **Causality-FM capability matrix**, containing graph-discovery models such as ARROW, TabCausal, and CDFM, with no attempt to merge SHD/F1 with effect errors.
