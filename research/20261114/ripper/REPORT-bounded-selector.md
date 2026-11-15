# H1af: Bounded LR evidence at fixed query cost

The frozen conjunction is **falsified_conjunction**. Its endpoints are reported separately:

- Gain of at least 10 pp: **falsified_10pp_gain**. bounded/learned minus exposure/learned accuracy is 3.0369 pp; marginal paired two-way 95% interval [-0.0696, 6.1585] pp.
- 70% usefulness floor: **falsified_70percent_floor**. bounded/learned accuracy is 24.6603%; marginal two-way 95% interval [19.1524, 30.7087]%.
- Zero wrong certified stops in both learned arms: **passed**, with 0 wrong stops.

Both intervals have marginal nominal 95% coverage, without a simultaneous-coverage claim. Exact-bound controls and other contrasts are descriptive.

| Selector / bound | Mean solved (%) | Calibration / training queries | Total queries | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |
| Exposure / exact | 52.4072 | 0 / 55,500 | 55,500 | 0 | 152297 | 0 | 0 | 191 |
| Bounded / exact | 53.0278 | 0 / 55,500 | 55,500 | 0 | 150311 | 0 | 0 | 189 |
| Exposure / learned | 21.6234 | 2,000 / 55,500 | 57,500 | 0 | 250805 | 0 | 0 | 744 |
| Bounded / learned | 24.6603 | 2,000 / 55,500 | 57,500 | 0 | 241087 | 0 | 0 | 688 |

The 32 unique learned bounds span 4.8596%–7.5419%, covering the actual 5% rate in 30/32 banks, with 0 no-pruning fallbacks. Each bank’s one calibration result is reused across selectors. Every outcome is retained; repeated reporting does not create 64 independent estimates.

Every arm uses actual independent replacement noise 5%, N2220 and 25 training queries per board. Learned arms pay 2,000 additional M1000 calibration queries, for 57,500 total; exact controls cost 55,500. The selector comparison within each calibration type has equal query cost. No training or calibration budget, q1, crossing law, temperature or search cap was tuned.

For unexcluded pairs, candidate weights are (8/9)^m times exp(clip(logLR/2,0,log16)), with zero diagonal and excluded weights. The fixed factor is bounded in [1,16], equivalent to at most 23.54 joint exposures of priority, without a coverage guarantee. The same greedy 25-cell choice and tie permutation apply. The policy sees only past queried joint/equality counts, exclusions and frozen calibration multipliers; read-only state hashes and event order verify all current masks finalize before current noise is generated or revealed. FP64 order is logLR expression, divide2, clip, exp, multiply exposure base.

The predictable-mask supermartingale proof applies despite using past LR evidence for selection. On valid calibration, edge-deletion risk is at most .005; calibration failure at most .005 gives at most .01 per learned model. Shared calibration does not create a joint 1% guarantee across selectors or banks. The proof requires the declared independent replacement-noise family and does not guarantee gain, usefulness, graph recovery or valid capped returns.

All settings, endpoints and streams froze before the old-board pilot and fresh outcomes. The 32 fresh banks contain 31,555 distinct calibration boards among 32,000 instances and 69,091 distinct training boards among 71,040 instances, excluding all 1,427,483 prior Ripper boards. Source roles are globally disjoint within this study. All 100,646 distinct source boards were independently matched against the CSV.

Before inference, independent cached read-only replay verified 284,160 masks and ties, 142,080 LR weight matrices, 920,678,400 log-LR values, all noise, 128 supports and 414,720 first-crossing paths. Exact rational crossings remain separate from floating selector weights. The six-bank pilot is separate and not pooled.

All 32 bank/search seeds share 10,000 full-all inputs [310000,320000). Exact support/settings/seed/input deduplication reduces 128 requested cells to 128 searches; all 1,280,000 returned boards passed independent audit. The four-GPU queue envelope is 15.39 GPU-min. Four chains, 262,144 moves per chain and unit1/B0/tau0/ebar0 remain fixed. Wrong certified stops and invalid capped returns are separate.

H1ae and all earlier studies remain unchanged and are not pooled. H1ae’s posthoc starvation diagnostic motivated this newly frozen bounded factor. No weight, gain threshold, usefulness floor or sample count was changed in response to H1af outcomes.

[Protocol](protocol-bounded-selector.json) · [Design and proof](BOUNDED_EVIDENCE_SELECTOR.md) · [Policy](bounded_selector_policy.py) · [Acquisition](bounded_selector_acquire.py) · [Replay](audit_bounded_selector_acquisition.py) · [Results](../../../results/research/ripper/bounded_selector.json)
