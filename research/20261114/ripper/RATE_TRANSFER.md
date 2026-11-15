# H1ac: economical calibration at fivefold greater replacement noise

The prior equal-budget study found that M1000/N2220 recovered all true graphs in 32/32 banks at independent replacement noise eta=.01. This prospective study keeps M1000/N2220 and tests transfer to eta=.05, with eta=.01 as a descriptive reference. It contains 32 fresh source banks and one new reserved 10,000-puzzle inference cohort. Pilot and earlier results are never pooled.

| Method | Actual noise | Calibration queries | Training queries | Total queries |
|---|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — |
| Exact known 1% | 1% | 0 | 55,500 | 55,500 |
| Learned 1% | 1% | 2,000 | 55,500 | 57,500 |
| Exact known 5% | 5% | 0 | 55,500 | 55,500 |
| Learned 5% | 5% | 2,000 | 55,500 | 57,500 |

Matched training N2220 isolates the effect of calibration error. It does not establish equal-total-cost superiority over known-noise controls. Every training board supplies 25 queried noisy labels; the learned arms also query one random cell twice on each of 1,000 calibration boards. Calibration and training source pools are globally disjoint within this study. Each estimator receives exactly its own 1,000 binary disagreement indicators, without clean labels or the other noise rate's indicators.

Three prespecified endpoints apply at 5% actual noise:

1. Noninferiority: learned minus exact mean accuracy must be within 5 percentage points. Its marginal paired two-way 95% bootstrap interval supports this endpoint if the lower bound is >= -5 pp, falsifies it if the upper bound is < -5 pp, and is otherwise inconclusive.
2. Usefulness: learned mean accuracy must be at least 70%. Its marginal two-way 95% bootstrap interval supports this endpoint if the lower bound is >=70%, falsifies it if the upper bound is <70%, and is otherwise inconclusive.
3. Zero wrong certified stops in both 5% primary arms. Any wrong certified stop falsifies this empirical gate.

The conjunction is supported only if both interval endpoints are supported and the stop gate passes. It is falsified if either interval endpoint is falsified or the stop gate fails; otherwise it is inconclusive. There are 5,000 bootstrap draws over 32 banks and the same 10,000 puzzle indices. The NI and absolute intervals each have marginal 95% nominal coverage; no simultaneous-coverage claim is made. The one-percent arms, additional contrasts, exact graph fractions, extra edges, and invalid capped returns are descriptive. No sample count, margin, floor, noise rate or inference budget changes after outcomes.

Noise is independent replacement by one of the other eight symbols. Within a bank, both rates use the same source boards, calibration cells, uniform random potentials and replacement offsets; a potential flips at .01 or .05 according to its fixed threshold. Thus lower-rate flips are a subset of higher-rate flips. Source and potential noise are paired; adaptive query masks can differ and are not assumed to be nested. Every active mask is finalized before any arm receives the current training noisy values. Calibration, training noise, source draws, tie permutations and cell choices use separate streams.

The unchanged fixed-M estimator computes a one-sided 99.5% Clopper–Pearson upper bound on disagreement d=2 eta-9 eta²/8, inverts its monotone branch, and rounds outward on a 1e-6 rational grid. Exact integer tails certify the endpoint and reject its predecessor. Reaching 8/9 triggers no pruning, and every outcome, including undercoverage, is retained.

For each learned model, calibration failure probability <=.005 and conditional true-edge deletion probability <=.005 imply total deletion probability <=.01. The exact controls use the same edge budget .005. For a valid upper eta_bar below8/9, q0=eta_bar/4-9eta_bar²/64 is conservative for a never-equal pair under the declared family. With fixed q1=1/9, predictable masks and first likelihood-ratio crossing648000, the earlier supermartingale/Ville/union proof applies. Pairing rates does not change either marginal bound, but no joint1% event across rates, arms or banks is claimed. The proof does not imply the usefulness floor, NI, exact graph recovery or validity of arbitrary capped returns. It does not cover correlated noise, noise drift, or noise families different from the declared independent replacement mechanism.

Independent source, calibration, threshold, mask/noise/count/path and support audits gate inference. All settings, streams, and endpoints freeze before an old-board implementation pilot or fresh fits. Solver settings stay unit1/B0/tau0/ebar0, four chains,262144moves, fixed temperatures/focus. The study uses at most30GPU-min including preflight. Exact support/settings/seed/input equality alone permits search reuse; no duplicates are run for utilization.
