# H1ah: confidence weights for capped-return accuracy

The frozen conjunction is **supported_conjunction**. Endpoints are reported separately:

- Evidence minus uniform, required gain 10 pp: **supported_10pp_gain**. Mean 56.6963 pp; marginal paired two-way 95% interval [51.4962, 61.6688] pp.
- Evidence minus shuffled, required gain 10 pp: **supported_10pp_gain**. Mean 61.2453 pp; marginal paired two-way 95% interval [56.0305, 66.1054] pp.
- Evidence accuracy, required floor 70%: **supported_70percent_floor**. Mean 76.5691%; marginal two-way 95% interval [74.7087, 78.1591]%.
- Zero invalid certified stops across all three arms: **passed**, with 0 invalid certificates.

Primary accuracy is exact match to the canonical solution, including capped returns. All three intervals have marginal nominal 95% coverage; no simultaneous-coverage claim is made.

| Weights | Canonical accuracy (%) | Correct certified (%) | Correct capped (%) | Canonical compatibility (%) | Given–given infeasibility (%) | Invalid certificates | Noncanonical valid returns |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — |
| uniform | 19.8728 | 11.3462 | 8.5266 | 12.3800 | 25.7609 | 0 | 0 |
| evidence | 76.5691 | 10.5922 | 65.9769 | 12.3800 | 25.7609 | 0 | 0 |
| shuffled | 15.3238 | 10.6100 | 4.7138 | 12.3800 | 25.7609 | 0 | 0 |

Descriptive weight placement for evidence: 0 weak true-edge–bank pairs and 860 weak false-edge–bank pairs. Geometry enters only this postfit audit, never weight assignment.

Descriptive weight placement for shuffled: 822 weak true-edge–bank pairs and 38 weak false-edge–bank pairs. Geometry enters only this postfit audit, never weight assignment.

All arms share the same 32 learned supports: 0 missing true edges and 873 extra edges in total. Each fit costs 57,500 queries (2,000 calibration plus 55,500 training); inference weightings do not consume additional labels or create independent training fits.

Uniform weights are 1. Evidence weights are 1 for exact rational LR≤1 and 1/16 otherwise. Shuffled weights have the same multiset, placed by a frozen full-pair permutation generated from an independent stream before calibration/training observations. No geometry or held-out label enters assignment.

Strong penalty unit 1, temperatures .4→.31, theta .45, tau0/ebar0 and tolerance .001 remain fixed. There is no per-model or mean-weight normalization. Positive weights preserve the zero set. Compatibility is for the stored canonical solution; uniqueness is not independently established. Given–given conflicts preclude zero-energy completion, but neither measure bounds positive-energy capped-return accuracy.

The 32 shared calibration bounds cover actual 5% noise in 32/32 banks, with 0 no-pruning fallbacks. All outcomes are retained. Source audits verified 100,410 distinct boards (31,533 calibration and 68,877 training), with globally disjoint source roles and all 1,628,683 prior Ripper rows excluded.

Before inference, independent replay checked 71,040 masks, 230,169,600 log-LR values, 103,680 first-crossing paths and 96 exact weight/permutation assignments. Each fresh search passed 150 general/sparse and 132 hidden-target state comparisons plus six exact integer field/energy/focus checks, including 127 incremental moves without refresh. Integer energies independently verified every returned certificate.

All bank/search seeds share 10,000 inputs [380000,390000). Complete weight/settings/seed/input equality reduces 96 requested cells to 96 searches; all 960,000 returned boards were audited. Total GPU allocation including standalone preflights is 13.3201 GPU-min. Earlier studies and the six implementation pilots are separate and not pooled.

[Protocol](protocol-confidence-weight.json) · [Design and proof](CONFIDENCE_WEIGHTING.md) · [Weight rule](confidence_weight_policy.py) · [Acquisition audit](audit_confidence_weight_acquisition.py) · [Results](../../../results/research/ripper/confidence_weight.json)
