# Empirical support audit

This revision evaluates the actual public support functions in
`cwfm/application/validation.py`, together with the existing minimum of ten
observations in each treatment group. It does not evaluate causal effect error.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m cwfm.support_experiment
```

The fixed protocol is `experiments/support_protocol.json`. Its 80 cells use
500 independent dataset replicates each: 30,000 static datasets and 10,000
network datasets. Sample sizes are 32, 64, 128, 256, and 512. The recorded run
took approximately ten seconds on CPU; no neural training, neural inference,
effect fitting, or threshold selection is involved.

Static cases use five independent normal covariates. Treatment probabilities
are logistic in either `s * X1` or `s * (X1**2 - 1)`, for slopes 0, 0.5, 1, 2, 4,
and 8. Population adequacy applies the same probability-tail policy to the
known true propensity distribution, calculated analytically. Slopes at most
one pass; slopes at least two fail. All finite logistic slopes retain strict
positivity. This reference concerns practical overlap under a fixed policy.

Network cases use a degree-four ring and independent treatment probabilities
0.5, 0.3, 0.1, and 0.02. Neighbor exposure has marginal distribution
`Binomial(4, p) / 4`; it is independent of own treatment, but neighboring
exposures remain dependent. The targets are 0.25 and 0.75. The population
quantile reference passes at probabilities 0.5 and 0.3. Nearby counts and
treatment-group sizes are finite-sample restrictions.

`per_seed.csv` records every decision, including separate screening and
treatment-count failures. `summary.csv` contains all 80 cells, with refusal
rates and pointwise 95% Wilson intervals over independent dataset replicates.
False refusal is rejection in a population-adequate cell. False acceptance is
acceptance in a population-inadequate cell. The inapplicable error direction
is missing, rather than zero. If "positive" denotes detection of inadequate
support, the false-positive rate is the false-refusal rate. A complementary
false-acceptance interval is `[1 - refusal_ci_high, 1 - refusal_ci_low]`.

Key findings include decreasing false refusals with sample size in several
linear cases and persistent misspecification error in quadratic cases. At
slope two and sample sizes 128 and 512, all quadratic cases pass the empirical
screen despite failing the population reference. Under network probability
0.3, the false-refusal rate remains 0.246 at size 512. These findings do not
validate arbitrary assignment models, covariate dimensions, or networks.

`manifest.json` records the protocol, code and artifact hashes, environment,
and elapsed time. The experiment evaluates the corrected support composition
that preserves an earlier treatment-group failure. Numerical screening
thresholds and the released neural checkpoint and calibration are unchanged.
