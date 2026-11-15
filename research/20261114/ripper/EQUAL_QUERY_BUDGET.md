# Equal total noisy-query budget: H1ab

The previous H1aa experiment established a 5 pp noninferiority result at different total query costs. This prospective follow-up compares ways of spending exactly 57,500 noisy oracle queries per bank. All parameters, margins, stream definitions, and sample counts freeze before its old-board implementation pilot or fresh fits.

| Arm | Calibration pairs | Calibration queries | Training boards | Training queries | Total queries |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this mechanism study; not run | — | — | — | — | — |
| Exact known 1% | 0 | 0 | 2,300 | 57,500 | 57,500 |
| Fixed 2% upper bound | 0 | 0 | 2,300 | 57,500 | 57,500 |
| Learned small | 1,000 | 2,000 | 2,220 | 55,500 | 57,500 |
| Learned large | 10,000 | 20,000 | 1,500 | 37,500 | 57,500 |

The primary hypothesis is that learned-small minus learned-large accuracy is noninferior within 5 percentage points and both learned arms produce zero wrong certified stops. Over 32 fresh banks and one shared reserved 10,000-puzzle inference cohort, a paired two-way bootstrap with 5,000 draws estimates the mean accuracy difference. Support requires the lower 95% bound >= -5 pp and zero primary-arm wrong certified stops. An upper bound < -5 pp or any primary-arm wrong certified stop falsifies the joint hypothesis. Otherwise it is inconclusive. Exact and fixed-upper controls, structural errors, and other accuracy contrasts are descriptive.

The replacement-noise rate is independently .01 for each query. For calibration, each bank has 10,000 distinct boards, each queried twice at one randomly chosen cell. The first 1,000 pairs form the small calibration; the large calibration contains all 10,000. The wrapper generates one (10000,2) Bernoulli array then one same-shape int8 offset array. Each estimator receives only its own disagreement-indicator prefix, no clean labels, later indicators, cell locations, or board identities. Shared calibration prefixes pair the arms without claiming the two estimates are independent.

Training uses the first N boards of one 2,300-board ordered bank. Every still-active arm finalizes its current-row mask from past queried data before any current noisy label is revealed. The arms share one tie permutation and one 81-cell potential-noise row. An arm stops querying at its frozen N and does not receive later labels. Calibration and training source pools are globally disjoint within this study. Source, cell, calibration-noise, training-noise and tie streams are separate. Calibration outcomes, undercoverage and fallback are retained without filtering or refitting.

For each fixed calibration size M, the disagreement count is Binomial(M,d), where d=2 eta - 9 eta²/8. The unchanged estimator computes a one-sided 99.5% Clopper–Pearson upper endpoint, inverts the monotone branch, and rounds outward on a 1e-6 grid. It certifies the chosen grid endpoint with exact integer binomial tails and rejects the previous endpoint. Each model uses its own bound, with no-pruning fallback at 8/9. The equal-budget interface changes only the declared number of binary indicators supplied to this estimator.

Each learned model separately has calibration failure probability <= .005. Conditional on a valid upper bound, the unchanged predictable-mask e-process with q1=1/9 and first LR crossing648000 has probability <= .005 of deleting any true edge. Their union is <= .01 per learned model. Nesting the two calibrations does not invalidate either marginal bound, but no joint .01 guarantee across the two learned models, all banks, or all controls is claimed. Both known-parameter controls use the same .005 edge budget. This is restricted to the fixed independent replacement-noise family shared across calibration and training; it does not cover correlated or drifting noise. Edge retention does not guarantee sparse recovery, inference accuracy, or validity of arbitrary capped outputs.

All source exclusions include both the calibration and training boards of H1aa. Every source board is reconstructed against the official CSV before inference. The six-bank implementation pilot uses previously seen boards and is never pooled. Independent endpoint, threshold, mask/noise, count, support, and first-crossing audits gate fresh solver inference. Search settings and 30 GPU-minute budget remain unchanged. Exact support/settings/seed/input equality alone permits reuse, even where acquisition arms have different N; all searched energies are unit1/B0/tau0/ebar0, so N does not affect the search once support is fixed.
