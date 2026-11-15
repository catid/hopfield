# H1ai: half-budget confidence weighting, source-only stage

The frozen finite-bank structural conjunction is **falsified_finite_bank_conjunction**.

This stage uses no held-out puzzles, GPU search or accuracy outcomes. It tests the unchanged H1ah rule at actual 5% noise, M1000/N1110: 29,750 queries per shared fit.

| Method | Missing true edges | Weak true edges | Retained extra edges | Weak extra edges | Strong extra edges |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — |
| uniform | 0 | 0 | 19948 | 0 | 19948 |
| evidence | 0 | 82 | 19948 | 19445 | 503 |
| shuffled | 0 | 10962 | 19948 | 8565 | 11383 |

Counts are pooled pair-bank observations across 32 fresh banks, not independent edge samples. Sudoku geometry enters only this postfit descriptive audit and never acquisition or weight placement.

Evidence weakens 97.4784% of retained extra edges; the frozen target is at least 95%. The other frozen targets require zero missing true edges and zero weak true edges. No universal or simultaneous statistical guarantee is claimed.

Calibration coverage is 32/32, with 0 safe fallbacks. All banks remain included irrespective of calibration coverage or structure.

Source audit verified 66,498 distinct complete boards, with calibration and training roles globally disjoint and all 1,729,093 previous source rows excluded. Independent replay checked 35,520 masks, 115,084,800 likelihood-ratio values, 103,680 first-crossing paths and 96 exact weight assignments.

Strong weight 1, weak weight 1/16, exact rational final LR threshold 1, no mean normalization and the independent shuffled multiset control remain unchanged. No thresholds, source banks or future search settings are selected from these outcomes.

A future held-out stage requires its own frozen cohort and execution registration. Its prospective targets remain at least 70% evidence canonical accuracy, at least 10 percentage-point gains against both uniform and shuffled weights, and zero invalid certified stops. Correct capped returns, certificates, canonical compatibility and given–given infeasibility must be reported separately. Source structure alone does not establish any of these accuracy targets.

[Protocol](protocol-confidence-budget-source.json) · [Source results](../../../results/research/ripper/confidence_budget_source.json)
