# H1ag: Fixed mixture evidence at matched query cost

The frozen conjunction is **falsified_conjunction**. Its endpoints are reported separately:

- Gain of at least 10 pp: **falsified_10pp_gain**. mixture/learned minus fixed/learned accuracy is -4.0703 pp; marginal paired two-way 95% interval [-6.8674, -1.1778] pp.
- 70% usefulness floor: **falsified_70percent_floor**. mixture/learned accuracy is 15.6691%; marginal two-way 95% interval [12.4175, 19.4258]%.
- Zero wrong certified stops in both learned arms: **passed**, with 0 wrong stops.

Both intervals have marginal nominal 95% coverage, without a simultaneous-coverage claim. Exact-bound controls and other contrasts are descriptive.

| Evidence law / bound | Mean solved (%) | Calibration / training queries | Total queries | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |
| Fixed / exact | 49.8428 | 0 / 55,500 | 55,500 | 0 | 160503 | 0 | 0 | 222 |
| Mixture / exact | 45.9944 | 0 / 55,500 | 55,500 | 0 | 172818 | 0 | 0 | 259 |
| Fixed / learned | 19.7394 | 2,000 / 55,500 | 57,500 | 0 | 256834 | 0 | 0 | 784 |
| Mixture / learned | 15.6691 | 2,000 / 55,500 | 57,500 | 0 | 269859 | 0 | 0 | 937 |

The 32 unique learned bounds span 5.5756%–7.3629%, covering the actual 5% rate in 32/32 banks, with 0 no-pruning fallbacks. Each bank’s one calibration result is reused across evidence laws. Every outcome is retained; repeated reporting does not create 64 independent estimates.

Every arm uses actual independent replacement noise of 5%, N2220 and 25 training queries per board. Learned arms pay 2,000 additional M1000 calibration queries, for 57,500 total; exact controls pay 55,500. The evidence-law comparison within each calibration type has equal query cost. All arms use the unchanged exposure selector.

The fixed control uses qA=1/9. The candidate uses E=(LqA+LqB)/2, with qB=1/6 and fixed equal weights. This is a mixture of component products. Exact integer cross-products certify the minimal crossing count and reject its predecessor; logaddexp values are diagnostic only. The nonfallback branch asserts that q0 is below both alternatives. The safe eta=8/9 fallback sets E=1 and retains all pairs.

Each component product is a nonnegative supermartingale under the true-edge null with predictable masks. A fixed equal mixture preserves that property. Ville and union bounds at 648000 give at most .005 edge-deletion risk; adding .005 calibration risk gives at most .01 per learned model. Shared calibration does not produce a joint 1% guarantee across arms or banks. Actual nonedge equality probabilities need not equal the alternatives. The proof does not guarantee gain, usefulness, graph recovery or valid capped returns.

All settings, endpoints and streams froze before the old-board pilot and fresh outcomes. The 32 fresh banks contain 31,591 distinct calibration boards among 32,000 instances and 68,963 distinct training boards among 71,040 instances, excluding all 1,528,129 prior Ripper boards. Source roles are globally disjoint within this study. All 100,554 distinct source boards were independently matched against the CSV.

Before inference, independent cached read-only replay verified 284,160 masks and ties, 142,080 mixture evidence rows, 920,678,400 logE values, all noise, 128 supports and 414,720 first-crossing paths. Exact rational crossings remain separate from floating diagnostic logE values. The six-bank pilot is separate and not pooled.

All 32 bank/search seeds share 10,000 full-all inputs [330000,340000). Exact support/settings/seed/input deduplication reduces 128 requested cells to 128 searches; all 1,280,000 returned boards passed independent audit. The four-GPU queue envelope is 16.09 GPU-min. Four chains, 262,144 moves per chain and unit1/B0/tau0/ebar0 remain fixed. Wrong certified stops and invalid capped returns are separate.

H1ae, H1af and all earlier studies remain unchanged and are not pooled. No mixture weight, alternative probability, gain threshold, usefulness floor or sample count was changed in response to H1ag outcomes.

[Protocol](protocol-mixture-law.json) · [Design and proof](MIXTURE_EVIDENCE_LAW.md) · [Policy](mixture_law_policy.py) · [Acquisition](mixture_law_acquire.py) · [Replay](audit_mixture_law_acquisition.py) · [Results](../../../results/research/ripper/mixture_law.json)
