# H3ac: independent finite-source-bank replication

Validated tracked exports for [the full report](../../../../research/20261114/loki/h3ac/REPORT.md). Four 128/176 source-bank draws, four BP AdamW fits, two frozen search kernels, one common fresh target cohort.

- `summary.json`: all decisions, source/target timing, capacities, audits and hashes.
- `per_seed.csv`: every scored search.
- `calibration-selections.json`: all timing trials and fixed selections.
- `source-banks.json`: sampled indices and overlap counts.
- `b0`–`b3`: source design, fitted parameters, transferred model and loss validation.
- `per-input-capacity.npz`: exact zero-F capacity for every fresh input/model.
- Figure files and `plot-provenance.json`: deterministic rendering from tracked results.

The confidence endpoints are Bonferroni-adjusted one-sided bootstrap bounds across four banks, separately for the lower and upper families; they are not a simultaneous 95% two-sided interval. Four banks from one finite rule family do not establish broad rule-family generalization.
