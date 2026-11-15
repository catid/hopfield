# H1al: shared corruption can break calibration while preserving constraints

The frozen finite-bank structural conjunction is **supported_finite_bank_conjunction**: shared retained extras must be at most 10% of iid extras, and both arms must retain every true edge.

| Corruption model | Retained extras | Missing true edges | Calibration coverage | Calibration disagreements | Realized calibration error (%) | Realized training error (%) |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — | — |
| iid | 19614 | 0 | 32/32 | 3140 | 5.0516 | 4.9686 |
| shared | 32 | 0 | 0/32 | 0 | 4.8812 | 4.7973 |

The observed pooled extra-edge reduction is 99.8369%. Counts are dependent pair-bank totals, not independent edge trials. This empirical target is separate from the mathematical invariance checks.

Both models have exactly 5% marginal digit error and a uniform wrong digit among eight alternatives. The shared model draws one Bernoulli(1/20) corruption event and one uniform derangement of nine digits per board. Both repeats on a calibration board share that draw; all training queries on a board share its draw. Rejection sampling of uniform permutations produces uniform derangements. Exhaustive enumeration checked all 133,496 derangements and 16,687 mappings for every ordered pair of distinct digits. A merely nonidentity permutation would not preserve the requested 5% marginal error.

Every shared bijection preserves pair equality. Shared calibration repeats therefore disagree zero times; the iid-based estimator returns eta_upper=0.002647, below the true marginal 0.05, in every bank. This coverage failure was predicted mathematically. It does not imply unsafe edge pruning here: true-edge equality stays exactly zero under the shared corruption.

The adaptive acquisition audit independently replayed each history-only mask before reconstructing current noise. For all 35,520 shared training rows, the complete clean and corrupted query trajectories, joint/equal counts, exclusions and first crossings agreed exactly at the same fitted threshold. Unqueried/future values never enter the mask selector. Clean labels enter only this postfit reference audit.

All 32 paired banks use identical clean calibration/training sources across arms, M1000/N1110, 25 queries per training board and 29,750 queries per arm. Source roles are globally disjoint; all 1,795,591 prior source rows were excluded. 66,479 distinct new boards were independently verified. Every draw is retained. No solver input, trajectory or GPU allocation was used.

This is a constraint-preserving noise family. It demonstrates that failed scalar noise calibration and successful relational learning can coexist; it does not show that correlation is generally easier or that the original iid calibration theorem applies.

[Protocol](protocol-correlated-noise.json) · [Corruption law](correlated_noise_policy.py) · [Whole-trajectory audit](audit_correlated_noise_acquisition.py) · [Results](../../../results/research/ripper/correlated_noise.json)

Execution used 1.8066 aggregate CPU-min and 22.8957 seconds wall time, within the frozen 15 CPU-min/180-second ceilings. Up to eight CPU workers ran; no GPUs were allocated.
