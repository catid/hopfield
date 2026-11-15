# H1aa: repeated-query noise calibration

The frozen 5 pp noninferiority plus zero primary-arm wrong-stop target is **supported_5pp_noninferiority**. Learned minus exact-known-noise accuracy is -1.9284 pp, with paired two-way 95% bootstrap interval [-4.4191, 0.6149] pp. Support requires a lower bound of at least -5 pp and zero wrong certified stops in both primary arms.

| Method | Mean solved (%) | Total queries per bank | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing true edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — |
| Exact known 1% noise | 76.1344 | 37,500 | 0 | 76370 | 14 | 0 | 23 |
| Repeated-query learned bound | 74.2059 | 57,500 | 0 | 82541 | 11 | 0 | 35 |
| Fixed 2% upper bound | 54.0003 | 37,500 | 0 | 147199 | 1 | 0 | 179 |

The 32 frozen calibration outcomes give bounds from 1.0481% to 1.3678%. Post-fit coverage of the actual 1% rate is 32/32; no-pruning fallback count is 0. These diagnostics never select, truncate or change a model. Every bank uses 10,000 repeated-query pairs and 1,500 fresh training boards, with 25 noisy queries per training board. The learned arm pays 20,000 calibration queries plus 37,500 training queries; the known-parameter controls pay 37,500.

The estimator receives only 10,000 disagreement indicators. A one-sided 99.5% Clopper–Pearson disagreement bound is inverted on the monotone independent-replacement-noise branch and rounded outward on a 1e-6 rational grid. Exact integer binomial tails certify the endpoint; an independent 80-digit backwards sum verifies the endpoint and rejects the previous grid point. If the bound reaches 8/9, every pair is retained.

Calibration error budget .005 plus conditional edge-deletion budget .005 gives at most .01 true-edge deletion risk per learned-bound model under the declared independent noise family, fixed across calibration and training. Both known-parameter controls use the same .005 edge budget and LR boundary 648000. This protects true edges; it does not guarantee few extra edges, exact graph recovery, solve accuracy, or arbitrary capped returns. It is not a single global 1% event across banks or arms.

The source split and every stream were frozen before the implementation-only pilot. Fresh calibration and training pools are globally disjoint within this study. The 320,000 calibration instances contain 293,179 distinct boards; the 48,000 training instances contain 47,358 distinct boards. All 356,066 previous Ripper boards are excluded. Cross-bank reuse is reported, not treated as new distinct inputs. Source reconstruction checked all 340,537 distinct boards against the CSV with an independent parser.

All 144,000 adaptive masks, 466,560,000 log-LR values, training and calibration noise draws, 96 supports, and 311,040 first-crossing paths passed independent replay before inference. All three masks are finalized before any current training label is revealed. A separate six-bank pilot passed the same audits and is not pooled.

Inference uses the same 10,000 full-all inputs [250000,260000) across 32 paired bank/search seeds. Exact model/settings/seed/input deduplication reduces 96 requested cells to 90 searches; all 900,000 returned boards were independently audited. The four-GPU queue envelope is 7.64 GPU-minutes. Four chains, 262,144 moves per chain, unit1, B0, tau0, ebar0 and search temperatures remain fixed. Wrong certified stops and invalid capped returns are distinct outcomes.

The fixed-2% comparison is descriptive. Earlier H1z remains a separate falsified study. No pilot or previous result is pooled, and the 5 pp margin, M10000, N1500 and bank count were not changed after outcomes.

[Protocol](protocol-calibrated.json) · [Proof](REPEATED_QUERY_CALIBRATION.md) · [Estimator](calibration_estimator.py) · [Acquisition](calibrated_acquire.py) · [Replay](audit_calibrated_acquisition.py) · [Search](calibrated_search.py) · [Public results](../../../results/research/ripper/calibrated.json)
