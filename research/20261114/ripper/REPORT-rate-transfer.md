# H1ac: calibration transfer from 1% to 5% noise

The frozen conjunction is **falsified_conjunction**. The three endpoints at 5% actual noise are reported separately:

- 5 pp noninferiority: **falsified_5pp_noninferiority**. Learned minus exact accuracy is -28.1453 pp; marginal paired two-way 95% interval [-33.9996, -22.2559] pp.
- 70% usefulness floor: **falsified_70percent_floor**. Learned accuracy is 24.1741%; marginal two-way 95% interval [19.5090, 29.1770]%.
- Zero wrong certified stops in both 5% arms: **passed**, with 0 wrong stops.

Both intervals have marginal nominal 95% coverage. No simultaneous-coverage claim is made. The conjunction requires both interval endpoints to pass and zero primary-arm wrong stops.

| Actual noise / method | Mean solved (%) | Calibration / training queries | Total queries | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |
| 1% exact | 80.9112 | 0 / 55,500 | 55,500 | 0 | 61084 | 32 | 0 | 0 |
| 1% learned | 80.9112 | 2,000 / 55,500 | 57,500 | 0 | 61084 | 32 | 0 | 0 |
| 5% exact | 52.3194 | 0 / 55,500 | 55,500 | 0 | 152578 | 0 | 0 | 197 |
| 5% learned | 24.1741 | 2,000 / 55,500 | 57,500 | 0 | 242643 | 0 | 0 | 673 |

All arms train on 2,220 boards with 25 noisy queries per board. Each learned arm additionally spends 2,000 queries on 1,000 repeated calibration pairs. Matched N isolates the calibration penalty; learned 57,500 versus exact 55,500 queries is not an equal-total-cost comparison. The 1% arms are descriptive references.

1% learned bounds: 1.2082%–2.0575%; coverage 32/32; no-pruning fallback 0. Diagnostics never select or alter a fit, including any undercoverage outcome.
5% learned bounds: 4.9194%–7.3629%; coverage 31/32; no-pruning fallback 0. Diagnostics never select or alter a fit, including any undercoverage outcome.

Each estimator receives only its own rate’s 1,000 disagreement indicators. The one-sided 99.5% Clopper–Pearson bound is inverted and rounded outward on a 1e-6 rational grid, with exact integer tails and independent 80-digit endpoint/predecessor checks. Calibration failure at most .005 plus conditional edge deletion at most .005 bounds true-edge deletion by .01 per learned model under the fixed independent replacement-noise family. There is no joint 1% claim across arms, rates or banks.

Both rates share source boards, calibration cells, uniform noise potentials and replacement offsets. Low-rate flips are a subset of high-rate flips; adaptive masks can differ and are not assumed nested. All four current masks finalize before any current training labels. Calibration, training noise, source, tie and cell streams are separate. Calibration and training source pools are globally disjoint within this study.

All settings, the 70% floor, 5 pp margin and stream rules froze before the implementation pilot and fresh outcomes. The 32 fresh banks use 31,650 distinct calibration boards among 32,000 instances and 69,304 distinct training boards among 71,040 instances, excluding all 1,059,144 previous Ripper boards. All 100,954 distinct source boards were independently matched against the official CSV.

Independent replay before inference checked 284,160 masks, 920,678,400 log-LR values, both-rate calibration/training noise, 128 supports and 414,720 first-crossing paths. The six-bank old-board pilot passed separately and is never pooled. Edge retention does not imply the 70% usefulness endpoint, noninferiority, sparse recovery or validity of arbitrary capped returns.

All 32 bank/search seeds share 10,000 full-all inputs [280000,290000). Exact support/settings/seed/input deduplication reduces 128 requested cells to 96 searches; all 960,000 returned boards passed independent audit. The four-GPU queue envelope is 9.88 GPU-min. Four chains, 262,144 moves per chain, unit1/B0/tau0/ebar0 and temperatures stay fixed. Wrong certified stops and invalid capped returns remain separate.

H1aa and H1ab are unchanged and not pooled. Other accuracy contrasts are descriptive and unadjusted. No endpoint or sample count was changed after observing outcomes.

[Protocol](protocol-rate-transfer.json) · [Design and proof](RATE_TRANSFER.md) · [Acquisition](rate_transfer_acquire.py) · [Replay](audit_rate_transfer_acquisition.py) · [Results](../../../results/research/ripper/rate_transfer.json)
