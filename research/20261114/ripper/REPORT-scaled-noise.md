# H1ad: doubled training at fixed 5% noise

The frozen N4440 conjunction is **supported_conjunction**. The primary endpoints are reported separately:

- 5 pp noninferiority: **supported_5pp_noninferiority**. Learned minus exact accuracy is 0.0000 pp; marginal paired two-way 95% interval [0.0000, 0.0000] pp.
- 70% usefulness floor: **supported_70percent_floor**. Learned accuracy is 82.3419%; marginal two-way 95% interval [81.9209, 82.7497]%.
- Zero wrong certified stops in both N4440 arms: **passed**, with 0 wrong stops.

Both intervals have marginal nominal 95% coverage, with no simultaneous-coverage claim. N2220 prefixes are descriptive. The earlier H1ac result remains falsified, unchanged and separate.

| Method / training N | Mean solved (%) | Calibration / training queries | Total queries | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |
| Exact, N2220 | 50.0081 | 0 / 55,500 | 55,500 | 0 | 159974 | 0 | 0 | 217 |
| Learned, N2220 | 17.7856 | 2,000 / 55,500 | 57,500 | 0 | 263086 | 0 | 0 | 806 |
| Exact, N4440 | 82.3419 | 0 / 111,000 | 111,000 | 0 | 56506 | 32 | 0 | 0 |
| Learned, N4440 | 82.3419 | 2,000 / 111,000 | 113,000 | 0 | 56506 | 32 | 0 | 0 |

The 32 unique learned bounds span 5.4564%–7.3629%, covering the actual 5% rate in 32/32 banks, with 0 no-pruning fallbacks. Each bank’s one M1000 calibration result is reused at both N values; duplicate reporting across prefixes does not create 64 independent estimates. All outcomes are retained without filtering.

N4440 costs 111,000 training queries. Each learned model also pays 2,000 calibration queries, for 113,000 total. Its N2220 prefix costs 57,500; the exact short control costs 55,500. Within each method, shorter masks, observations and log-LR paths are literal prefixes of the long trajectory. Short arms receive no labels after N2220. All active current masks finalize before any current training labels.

The noise family, M1000 calibration, q1=1/9, selector, .005 calibration budget and .005 edge budget remain fixed. The existing outward Clopper–Pearson endpoint and predictable-mask proof give at most .01 true-edge deletion probability per learned model. Nested prefixes preserve each marginal statement without creating a joint 1% event across N, arms or banks. This bound does not guarantee usefulness, noninferiority, graph recovery or valid capped returns.

All settings, the 70% floor, 5 pp margin and stream rules froze before the old-board implementation pilot and fresh outcomes. The 32 fresh banks contain 31,626 distinct calibration boards among 32,000 instances and 135,038 distinct training boards among 142,080 instances. Role pools are globally disjoint within this study; all 1,160,098 prior Ripper boards are excluded. All 166,664 distinct source boards were independently matched against the official CSV.

Independent replay before inference checked 426,240 masks, 1,381,017,600 log-LR values, calibration/training noise, 128 supports and 414,720 first-crossing paths. Additional prefix-identity checks covered 142,080 masks and 460,339,200 log-LR entries already included in those paths. The six-bank implementation pilot is separate and not pooled.

All 32 bank/search seeds share 10,000 full-all inputs [290000,300000). Exact support/settings/seed/input deduplication reduces 128 requested cells to 96 searches; all 960,000 returned boards passed independent audit. The four-GPU queue envelope is 10.02 GPU-min. Four chains, 262,144 moves per chain, unit1/B0/tau0/ebar0 and temperatures stay fixed. Wrong certified stops and invalid capped returns remain separate.

Other accuracy contrasts are descriptive and unadjusted. No endpoint, query allocation, selector or q1 was changed in response to any pilot or fresh outcome.

[Protocol](protocol-scaled-noise.json) · [Design and proof](SCALED_NOISE.md) · [Acquisition](scaled_noise_acquire.py) · [Replay](audit_scaled_noise_acquisition.py) · [Results](../../../results/research/ripper/scaled_noise.json)
