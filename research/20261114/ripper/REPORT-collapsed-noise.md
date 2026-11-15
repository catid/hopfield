# H1am: destructive shared corruption defeats the repeated-query calibration assumption

The frozen finite-bank target is **supported_at_least_90percent_banks_lose_true_edges**: 32/32 banks lose at least one true edge; at least 29/32 was required.

| Noise model | Banks losing true edges | Missing true edges | Retained extras | Calibration coverage | Realized calibration error (%) | Realized adaptive training error (%) | Observed true-pair equality (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — | — | — |
| iid | 0/32 | 0 | 19614 | 32/32 | 5.0516 | 4.9686 | 1.2000 |
| collapse | 32/32 | 25302 | 0 | 0/32 | 5.0156 | 5.0787 | 5.7171 |

The collapse model draws one Bernoulli(9/160) event and one uniform output digit per board, after the training mask is finalized. On a bad board every query returns that digit. Both calibration repeats share the same event/output. For any fixed pre-query cell, marginal error is exactly (9/160)(8/9)=1/20. Realized errors on the adaptive query sequence are measured separately above; no draw is filtered or rescaled.

Conditional on past history, a current clean board and its pre-noise mask, an unequal clean pair has equality probability 9/160 under collapse; an equal clean pair stays equal with probability 1. These exact law statements were independently checked for all 81 clean digit pairs. Actual co-query exposures and equal counts were independently reconstructed for every row and compared with the saved counts.

Repeated calibration queries always agree, so the iid-based estimator produces eta_upper=0.002647 and misses the nominal 5% marginal error in all 32 banks. The resulting learned true-pair null is 0.00066076, below the actual conditional 9/160. Those calibration identities are separate from the empirical pruning outcome.

The collapse arm loses between 602 and 810 true edges per bank. This removes a premise of zero-energy Sudoku certificate soundness. No solver was run, so this study does not claim observed invalid certificates or solver accuracy.

All 32 H1al source banks are reused without selection, with their archived iid controls. Both acquisition methods use M1000/N1110, 25 training queries per board and 29,750 queries per fit, matched source/tie/cell streams and independent current-noise draws. No new source cohort, solver cohort or GPU allocation is used. New acquisition replay checks 35,520 masks, 115,084,800 LR values and 103,680 first-crossing paths.

H1al shared derangements preserved equality despite calibration failure; this explicitly nonbijective collapse law destroys inequalities. These are two specified correlation models, not a universal ordering of correlated noise difficulty. The frozen target is a finite-bank structural failure criterion.

[Protocol](protocol-collapsed-noise.json) · [Corruption law](collapsed_noise_policy.py) · [Independent audit](audit_collapsed_noise_acquisition.py) · [Results](../../../results/research/ripper/collapsed_noise.json)

Execution used 0.7565 aggregate CPU-min and 5.9235 seconds wall time, within the frozen 15 CPU-min/180-second ceilings. Up to eight CPU workers ran; no GPUs were allocated.
