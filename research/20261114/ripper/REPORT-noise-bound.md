# H1z: conservative noise calibration

The frozen joint target of 10 pp noninferiority and zero protected wrong stops is **falsified_10pp_noninferiority**. Accuracy with an assumed 2% noise rate minus accuracy with the exact 1% rate is -21.8850 pp, with a paired two-way 95% bootstrap interval [-25.4467, -18.0880] pp. Noninferiority requires the lower bound to be at least -10 pp. The two protected arms produced 0 wrong certified stops.

| Assumed noise rate | Mean solved (%) | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing true edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — |
| 1% exact | 78.9841 | 0 | 67251 | 20 | 0 | 17 |
| 2% conservative upper bound | 57.0991 | 0 | 137283 | 0 | 0 | 160 |
| 0.5% unprotected underbound | 81.9725 | 0 | 57688 | 31 | 1 | 0 |

All 32 independent banks use 1,500 boards, an actual noise rate of 1%, 25 queries per board, fixed q1=1/9, and the same exponential mask rule. N1500 and the 10 pp margin were fixed before the implementation pilot. The new banks contain 47,651 distinct source boards among 48,000 row instances and exclude 308,415 previously used boards.

The stored, fully labeled source boards simulate the label oracle. Each policy receives only 37,500 noisy queried digits per bank and finalizes each mask before seeing any current noisy values. Source, noise, and tie streams are independent across banks. Common potential noisy values pair the three calibrations within each bank.

When 0 <= eta_true <= eta_bound < 8/9, the never-equal noisy-equality probability increases with eta. With q0_bound < q1, the conditional expected likelihood-ratio multiplier is at most one. The predictable-mask supermartingale and Ville/union proof therefore retain the 1% per-model protection against deleting true edges. This does not guarantee few extra edges, solve accuracy, or the separate zero-wrong-stop condition. At eta_bound=8/9, the likelihood ratio is identically one and cannot exclude pairs.

The 0.5% underbound has a conditional expected null multiplier above one and lacks this probability guarantee. Its measured success or failure is not assumed; observing zero errors does not establish protection. All 144,000 masks, 466,560,000 log-LR values, noise draws, 96 supports, and 311,040 first-crossing paths passed independent audit. The exact control reproduced the original old-bank masks and paths during implementation validation.

Inference uses the same 10,000 new full-all inputs [230000,240000) across 32 paired bank/search seeds. Exact support/settings/seed/input deduplication reduces 96 requested cells to 77 searches; all 770,000 returned boards were independently checked. Reusing an identical model does not create independent input evidence. Four chains, 262,144 moves per chain, unit 1, B=0, tau=0, and ebar=0 remain fixed. Wrong certified stops and invalid capped returns are counted separately.

The earlier exact-calibration H1y result remains unchanged. The implementation pilot and earlier studies are not pooled. Other accuracy contrasts and structural comparisons are descriptive and unadjusted.

[Protocol](protocol-noise-bound.json) · [Upper-bound proof](NOISE_UPPER_BOUND_EPROCESS.md) · [Policy](noise_bound_policy.py) · [Acquisition](noise_bound_acquire.py) · [Replay](audit_noise_bound_acquisition.py) · [Search](noise_bound_search.py) · [Audit](audit_noise_bound.py) · [Public results](../../../results/research/ripper/noise_bound.json)
