# H1r: compressing a fully labeled training pool

**Selection reads all 1,000 complete labels per bank. This is training-set compression, not learning from only the retained labels.**

The frozen hypothesis that the fixed greedy algorithm selects at most 16 boards in every pool is **falsified**. This rule retains 26–27 boards in the six pools. Every retained subset produces the same count energy as its full pool. The bounded MILP timeouts do not falsify the existence of smaller covers.

| Seed | Labels read | Greedy cover | MILP incumbent | Exact lower bound | Solver status | Gap |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 1,000 | 26 | 26 | 10 | Time limit | 61.54% |
| 1 | 1,000 | 27 | 27 | 10 | Time limit | 62.96% |
| 2 | 1,000 | 26 | 26 | 10 | Time limit | 61.54% |
| 3 | 1,000 | 27 | 27 | 10 | Time limit | 62.96% |
| 4 | 1,000 | 27 | 27 | 10 | Time limit | 62.96% |
| 5 | 1,000 | 27 | 27 | 10 | Time limit | 62.96% |

Each MILP received its greedy cover as a feasible warm start. All six runs hit the 55-second wall cap, used less than the 60-CPU-second budget and retained their initial incumbents. HiGHS reports a lower bound of ten for every pool; the remaining 61.5–63.0% gaps prevent an optimality claim.

The selector sees only equal-digit pair coverage and source-row identifiers. It greedily maximizes newly covered pairs, breaking ties by the original source row. A reversed pool order and independent per-board digit relabelings preserve the selected coverage. No peer geometry is used for construction.

An independent post-selection geometry certificate establishes the ten-board lower bound exactly, separately from the HiGHS floating-point dual estimate of ten. The elementary counting bound is ceil(2430/324)=8. Every board covers 162 of the 1,458 different-band-and-stack pairs, giving the exact orbit bound nine. Nine boards could cover that orbit only without overlap, but every pair of candidate boards in each pool overlaps on at least one such pair, giving the exact finite-pool bound ten. All 2,997,000 candidate-pair overlaps were checked with integer arithmetic. This does not determine the minimum over boards outside the fixed pools.

Greedy and MILP subsets were independently verified against the full equality union. Their unit1 equal-digit energies, zero biases and zero stopping/focus calibration are identical to the existing full-count control. No duplicate GPU inference was run. The mapped six searches on [100000,110000) remain the earlier 80.9517% short-budget result, not new evidence.

BP AdamW is outside this count-energy compression study and was not run.

[Protocol](protocol-compression.json) · [Selector and solver](compress_pool.py) · [Independent audit](audit_compression.py) · [Exact lower-bound proof](COMPRESSION_LOWER_BOUND.md) · [Public results](../../../results/research/ripper/compression.json)
