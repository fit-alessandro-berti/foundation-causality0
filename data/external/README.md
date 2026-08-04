# External energy datasets

## Primary semi-synthetic benchmark

`appliances-energy.zip` is the UCI Appliances Energy Prediction dataset by
Luis Candanedo (2017), downloaded without modification from
<https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip>.

- UCI record: <https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction>
- DOI: <https://doi.org/10.24432/C5VC8G>
- License: CC BY 4.0
- Downloaded: 2026-08-04
- ZIP SHA-256:
  `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a`

The source contains 19,735 ten-minute observations from a low-energy house.
Temperature and humidity were measured by a ZigBee sensor network, appliance
energy by m-bus meters, and weather at a nearby station. The revision protocol
selects eight sensor/weather covariates without using the outcome, uses logged
appliance energy as a prognostic baseline, and adds a frozen reproducible
treatment-assignment and heterogeneous-effect mechanism. This makes the
causal target semi-synthetic; it is not attributed to UCI.
