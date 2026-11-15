# H1ad: doubling training at fixed 5% noise

H1ac remains a falsified rate-transfer study: at 5% noise, economical M1000/N2220 calibration failed its 5 pp noninferiority and 70% usefulness targets. This separate prospective study doubles training to N4440 while keeping M1000, independent replacement noise eta=.05, fixed q1=1/9, the selector, and both .005 error budgets unchanged.

| Method | Calibration queries | Training queries | Total queries | Role |
|---|---:|---:|---:|---|
| BP AdamW: outside this study; not run | — | — | — | — |
| Exact, N2220 | 0 | 55,500 | 55,500 | Descriptive prefix |
| Learned, M1000/N2220 | 2,000 | 55,500 | 57,500 | Descriptive prefix |
| Exact, N4440 | 0 | 111,000 | 111,000 | Primary control |
| Learned, M1000/N4440 | 2,000 | 111,000 | 113,000 | Primary candidate |

Exactly 32 fresh banks and one reserved 10,000-puzzle cohort are used. Within a bank, both learned arms use the same fixed calibration result; the shorter training trajectories are literal prefixes of the longer trajectories for the same method. Source boards, tie permutations and potential training noise are shared. The shorter arms stop querying at N2220. Independent replay must verify prefix identity, every current mask before labels, and all per-model query costs.

At N4440 the three endpoints are: learned minus exact accuracy within 5 percentage points, learned absolute accuracy at least 70%, and zero wrong certified stops in both primary arms. Each interval uses 5,000 two-way bootstrap draws over banks and the shared test-puzzle indices. NI is supported if its marginal 95% lower bound is at least -5 pp, falsified if its upper bound is below -5 pp, and otherwise inconclusive. Usefulness is supported if its marginal 95% lower bound is at least 70%, falsified if its upper bound is below 70%, and otherwise inconclusive. Any wrong certified stop falsifies the empirical stop gate. The conjunction is supported only when both interval endpoints are supported and the stop gate passes; it is falsified if either interval endpoint is falsified or the stop gate fails, and otherwise is inconclusive. No simultaneous-coverage claim is made. N2220 prefix results and all other contrasts are descriptive.

One random cell on each of 1,000 disjoint calibration boards is queried twice. The estimator receives only the 1,000 disagreement indicators. It uses the unchanged one-sided 99.5% Clopper–Pearson disagreement upper, monotone inverse and outward 1e-6 rational grid. Exact integer tails certify the endpoint and reject its predecessor. Reaching eta=8/9 triggers no pruning. All outcomes, including undercoverage, are retained and never recalibrated using known true noise.

The earlier proof applies unchanged: calibration failure at most .005 plus conditional edge-deletion failure at most .005 gives at most .01 true-edge deletion risk per learned model. The exact controls use the same .005 edge budget, LR boundary648000 and q1=1/9. Nested training and reused calibration preserve the marginal bound at either fixed N without creating a joint1% event across models or banks. Edge retention does not guarantee the usefulness floor, noninferiority, graph recovery or valid capped returns. The guarantee requires the declared fixed independent replacement-noise family, with the same rate during calibration and training.

Source roles are globally disjoint within this study; every previous Ripper source row is excluded from fresh inputs. The six-bank implementation pilot uses only old boards: each training sequence concatenates the matching H1ab2300 and H1ac2220 old training banks, taking the first4440. Its calibration sequence is the first1000 old H1aa calibration boards. H1ab/H1ac excluded H1aa sources, ensuring calibration/training disjointness. The pilot is not pooled with fresh evidence. Source, endpoint, threshold, noise, mask, count, support, prefix and first-crossing audits gate fresh inference.

All settings and streams freeze before pilot or fresh outcomes. The same 10,000 fresh inference inputs repeat over 32 paired bank/search seeds, using fixed unit1/B0/tau0/ebar0, four chains and262144moves. The budget is30GPU-min including preflight. Only exact support/settings/seed/input equality permits reuse; no filler or duplicate searches are run. No selector or q1 tuning occurs in this scaling study.
