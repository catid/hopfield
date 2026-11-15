# H1ae: LR evidence for acquisition at fixed cost

The frozen conjunction is **falsified_conjunction**. Its endpoints are reported separately:

- Gain of at least 10 pp: **falsified_10pp_gain**. LR/learned minus exposure/learned accuracy is -21.5431 pp; marginal paired two-way 95% interval [-26.6386, -16.8924] pp.
- 70% usefulness floor: **falsified_70percent_floor**. LR/learned accuracy is 0.0009%; marginal two-way 95% interval [0.0000, 0.0037]%.
- Zero wrong certified stops in both learned arms: **passed**, with 0 wrong stops.

Both intervals have marginal nominal 95% coverage, without a simultaneous-coverage claim. Exact-bound controls and other contrasts are descriptive.

| Selector / bound | Mean solved (%) | Calibration / training queries | Total queries | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |
| Exposure / exact | 50.0881 | 0 / 55,500 | 55,500 | 0 | 159718 | 0 | 0 | 217 |
| LR / exact | 0.0694 | 0 / 55,500 | 55,500 | 0 | 319778 | 0 | 0 | 4230 |
| Exposure / learned | 21.5441 | 2,000 / 55,500 | 57,500 | 0 | 251059 | 0 | 0 | 750 |
| LR / learned | 0.0009 | 2,000 / 55,500 | 57,500 | 0 | 319997 | 0 | 0 | 10128 |

The 32 unique learned bounds span 5.2179%–7.6015%, covering the actual 5% rate in 32/32 banks, with 0 no-pruning fallbacks. Each bank’s one calibration result is reused across selectors. Every outcome is retained; repeated reporting does not create 64 independent estimates.

Every arm uses actual independent replacement noise 5%, N2220 and 25 training queries per board. Learned arms pay 2,000 additional M1000 calibration queries, for 57,500 total; exact controls cost 55,500. The selector comparison within each calibration type has equal query cost. No training or calibration budget, q1, crossing law, temperature or search cap was tuned.

For unexcluded pairs, the candidate weights are exp(min(logLR, log648000)/2), with zero diagonal and excluded weights. Its greedy 25-cell choice and tie permutation are unchanged. The policy sees only past queried joint/equality counts, exclusions and frozen calibration multipliers; read-only state hashes and event order verify all current masks finalize before current noise is generated or revealed. Capping before exponentiation prevents overflow; underflow to zero is harmless.

The predictable-mask supermartingale proof applies despite using past LR evidence for selection. On valid calibration, edge-deletion risk is at most .005; calibration failure at most .005 gives at most .01 per learned model. Shared calibration does not create a joint 1% guarantee across selectors or banks. The proof requires the declared independent replacement-noise family and does not guarantee gain, usefulness, graph recovery or valid capped returns.

All settings, endpoints and streams froze before the old-board pilot and fresh outcomes. The 32 fresh banks contain 31,624 distinct calibration boards among 32,000 instances and 69,097 distinct training boards among 71,040 instances, excluding all 1,326,762 prior Ripper boards. Source roles are globally disjoint within this study. All 100,721 distinct source boards were independently matched against the CSV.

Before inference, independent cached read-only replay verified 284,160 masks and ties, 142,080 LR weight matrices, 920,678,400 log-LR values, all noise, 128 supports and 414,720 first-crossing paths. Exact rational crossings remain separate from floating selector weights. The six-bank pilot is separate and not pooled.

All 32 bank/search seeds share 10,000 full-all inputs [300000,310000). Exact support/settings/seed/input deduplication reduces 128 requested cells to 128 searches; all 1,280,000 returned boards passed independent audit. The four-GPU queue envelope is 19.37 GPU-min. Four chains, 262,144 moves per chain and unit1/B0/tau0/ebar0 remain fixed. Wrong certified stops and invalid capped returns are separate.

Earlier H1ac and H1ad results are unchanged and not pooled. No weight, gain threshold, usefulness floor or sample count was changed in response to outcomes.

[Protocol](protocol-lr-selector.json) · [Design and proof](LR_EVIDENCE_SELECTOR.md) · [Policy](lr_selector_policy.py) · [Acquisition](lr_selector_acquire.py) · [Replay](audit_lr_selector_acquisition.py) · [Results](../../../results/research/ripper/lr_selector.json)
