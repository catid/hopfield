# H1ab: equal total noisy-query budget

The frozen 5 pp noninferiority plus zero learned-arm wrong-stop target is **supported_5pp_noninferiority**. Learned M1000/N2220 minus learned M10000/N1500 accuracy is 6.8600 pp, with paired two-way 95% bootstrap interval [4.7597, 9.1448] pp. The lower bound must be at least -5 pp and both learned arms must have zero wrong certified stops.

| Method | Calibration / training queries | Total | Mean solved (%) | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |
| Exact 1%, N2300 | 0 / 57,500 | 57,500 | 83.1678 | 0 | 53863 | 32 | 0 | 0 |
| Fixed 2%, N2300 | 0 / 57,500 | 57,500 | 82.9766 | 0 | 54475 | 31 | 0 | 1 |
| M1000, N2220 | 2,000 / 55,500 | 57,500 | 83.1678 | 0 | 53863 | 32 | 0 | 0 |
| M10000, N1500 | 20,000 / 37,500 | 57,500 | 76.3078 | 0 | 75815 | 11 | 0 | 37 |

Every arm has exactly 57,500 noisy queries per bank. The small calibration saves 18,000 queries and spends them on 720 additional training boards. The known-parameter controls use all 57,500 queries on 2,300 boards. Those controls are descriptive; they assume exact noise knowledge or a valid fixed upper bound.

M1000, N2220: learned bounds 1.2082%–2.6204%; post-fit coverage 32/32; no-pruning fallbacks 0. These diagnostics never select or modify models.
M10000, N1500: learned bounds 1.0204%–1.2964%; post-fit coverage 32/32; no-pruning fallbacks 0. These diagnostics never select or modify models.

Calibration labels are nested: the small estimator receives only the first 1,000 of 10,000 disagreement indicators. It never receives clean labels, later indicators, cell choices, or board identities. Each training arm uses its frozen prefix of 2,300 boards and stops querying at N. On every row, all still-active masks finalize before any current training noisy label is revealed. Source, calibration-cell, calibration-noise, training-noise, and tie streams are separate.

For each fixed M, the unchanged estimator uses a one-sided 99.5% Clopper–Pearson disagreement bound, monotone inversion, and outward rounding on a 1e-6 grid. Exact integer binomial tails certify the endpoint; independent 80-digit backwards sums verify both endpoint and rejected predecessor. A bound reaching 8/9 triggers no pruning. All outcomes, including undercoverage or fallback, remain in the study.

Each learned model separately has calibration failure probability <= .005 and conditional true-edge deletion probability <= .005, totaling <= .01 under the fixed independent replacement-noise family. Nested calibration preserves these marginal statements, with no joint 1% guarantee across arms or banks. Known-parameter controls use the same .005 edge budget. The proof does not ensure few extra edges, graph recovery, solve accuracy, or valid capped returns.

All settings and stream rules froze before the six-bank implementation pilot and fresh fits. The 32 fresh banks contain 290,605 distinct calibration boards among 320,000 instances and 71,936 distinct training boards among 73,600 instances. The source roles are globally disjoint within this study, and all 696,603 prior Ripper boards are excluded. All 362,541 source boards were independently matched to the official CSV.

Before inference, independent replay checked all 266,240 adaptive masks, 862,617,600 log-LR values, calibration and training noise, 128 supports, and 414,720 first-crossing paths. It also checked each arm stops querying at its declared N and every total query cost equals 57,500. The implementation pilot is reported separately and never pooled.

Inference uses the same 10,000 full-all inputs [270000,280000) across 32 paired bank/search seeds. Exact support/settings/seed/input deduplication reduces 128 requested cells to 54 searches; all 540,000 returned boards were independently audited. The four-GPU queue envelope is 3.90 GPU-minutes. Four chains, 262,144 moves per chain, unit1/B0/tau0/ebar0 and temperatures remain fixed. Wrong certified stops and invalid capped returns are separate.

The previous unequal-cost H1aa result remains unchanged. Other comparisons are descriptive and unadjusted. Sample counts, query allocations, error budgets, margin and noise family were not changed in response to any pilot or fresh outcome.

[Protocol](protocol-equal-budget.json) · [Design and proof](EQUAL_QUERY_BUDGET.md) · [Estimator](equal_budget_estimator.py) · [Acquisition](equal_budget_acquire.py) · [Replay](audit_equal_budget_acquisition.py) · [Search](equal_budget_search.py) · [Results](../../../results/research/ripper/equal_budget.json)
