# H1an: an observed-label balance filter under shared collapse

The preregistered target of at least 29 exact filtered graphs is **falsified_finite_bank_target**: 9/32 fresh banks recovered the graph exactly.

| Arm | Exact graphs | Missing true edges | Retained extras | Rejected training boards | Accepted corrupt boards | Rejected clean boards |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — | — |
| unfiltered | 0/32 | 25181 | 0 | 0 | 1961 | 0 |
| filtered | 9/32 | 0 | 54 | 1961 | 0 | 0 |

The filter uses an explicitly supplied global Sudoku prior: each digit occurs at most nine times in a completed board. After all 25 labels are observed, the complete row is rejected if any observed digit count exceeds nine. The spent queries are charged; rejected rows update no pair evidence. No peer graph, unqueried value, or clean label enters filtering or mask selection.

Shared collapse draws one Bernoulli(9/160) event and a uniform output digit per board; all queries and calibration repeats share that draw. Fixed pre-query cells therefore have marginal error (9/160)(8/9) = 5%. Clean 25-cell subsets obey the balance prior; collapsed subsets contain 25 identical labels. Filter purity is a mathematical consequence for this law, separate from the empirical exact-recovery target.

The independent replay checked every mask, observed label, likelihood ratio and first crossing in 64 acquisitions. All 35,520 filtered rows matched a separate clean accepted-subsequence state, including tie-RNG advancement on rejected rows and original row-indexed crossings. Filtered accepted data contain zero corrupt rows and zero label errors; no clean row was rejected and no true edge was pruned. The number of retained extras remains an empirical outcome.

Both arms spend 27,750 training queries per bank with M1000/N1110 and 25 labels per training board. They share one 2,000-query calibration fit, so there are 32 calibrations and 57,500 physical queries per paired bank, not 64 independent calibrations. Both arms have identical clean sources, potential collapse draws and tie streams; adaptive masks may diverge through history.

All 66,407 new distinct source boards were independently checked against the source CSV after excluding 1,862,070 prior boards. Source roles are globally disjoint. All 32 paired banks and every outcome are retained. This source-only study uses no solver cohort or GPU.

Calibration repeats still agree under collapse. The nominal scalar noise estimate remains below 5% in every bank; the filter repairs observed pair evidence for this specific corruption law, not calibration coverage. Its supplied balance prior need not hold in other tasks, and other correlated corruptions may evade the filter.

[Protocol](protocol-balanced-filter.json) · [Filter](balanced_filter_policy.py) · [Independent replay](audit_balanced_filter_acquisition.py) · [Results](../../../results/research/ripper/balanced_filter.json)

Execution used 1.5224 aggregate CPU-min and 20.2746 seconds wall time, within the frozen15 CPU-min/180-second ceilings. Up to eight CPU workers ran; no GPUs were allocated.
