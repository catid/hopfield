# Exact mixture threshold recurrence

H1ag preparation spent much of its CPU time repeatedly raising large rational numerators and denominators to powers. A separate helper updates the same unreduced integer products from the previous joint count. Increasing the equality count replaces one non-equality factor by an equality factor using exact division. Integer cross-products certify the threshold and reject its predecessor.

The helper is **not used by H1ag**. Its frozen policy, acquisition runners, thresholds and scientific outcomes remain unchanged. This is a validated implementation candidate for a future protocol.

All 64 archived H1ag mixture tables matched exactly: 142,144 thresholds across 23 distinct supplied noise bounds. The safe no-pruning fallback also matched. Three fresh full-table comparisons against the original implementation gave:

| Supplied noise bound | Original table (s) | Recurrence table (s) | Observed speedup |
|---|---:|---:|---:|
| 5% | 3.004 | 1.432 | 2.10× |
| 6.4095% | 15.582 | 7.557 | 2.06× |
| 7.5419% | 18.874 | 9.149 | 2.06× |

These are single sequential CPU measurements in one process while GPU inference ran on the host. The original table cache was cleared before timing. They are not randomized equal-load throughput measurements.

[Helper](fast_mixture_thresholds.py) · [Validation](validate_fast_mixture_thresholds.py) · [Evidence](../../../results/research/ripper/fast_mixture_threshold_validation.json)
