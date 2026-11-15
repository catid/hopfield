# H1v: exponential versus mixture acquisition ablation

The frozen 10% relative-extra-edge noninferiority target is **supported_10percent_noninferiority**. Exponential scoring changes total N300 false extra edges relative to the unchanged spike-mixture policy by -8.249%, with paired-bank 95% bootstrap interval [-17.471%, 1.578%]. Noninferiority requires the upper bound to be at most 10%. This uncertainty uses 32 fresh disjoint banks and one fixed tie stream per bank.

| Policy | Total N300 extra edges (32 banks) | Mean test solved (%) | Wrong stops |
|---|---:|---:|---:|
| BP AdamW: outside this acquisition study; not run | — | — | — |
| candidate | 1714 | 3.3897 | 0 |
| spike | 885 | 13.6003 | 0 |
| exponential | 812 | 15.8919 | 0 |

Every policy queries exactly 25 cells per incoming board: 7,500 revealed digits at N300 in each bank. The stored complete boards form a simulated label oracle. The policy receives only past joint/equal observation counts and a fixed tie order; it cannot access the current board, source identity, geometry or oracle handle.

The candidate and mixture controls dispatch to their unchanged frozen implementations. With m prior unequal co-queries and no observed equality, candidate weight is 1/(1+m), mixture weight is 2u/(1+u), and exponential weight is u, where u=(8/9)^m. All three weights become zero after any witnessed equality. The fixed 25-cell greedy selection and tie rule are shared. The exponential arm ablates the mixture denominator while retaining the same fixed rate; no rate or margin was tuned. These are acquisition heuristics, not calibrated Sudoku posterior probabilities. Zero weight after a witnessed equality means no value in reobserving an already excluded pair.

All 28,800 mask decisions were independently reconstructed from past queried archives. Masks were fixed before current labels were revealed, premature reveals were rejected, and hidden current-row permutations left those masks unchanged. All 96 prefix supports and 192 hidden-value substitution recounts passed. Geometry appears only in the final audit.

Fresh evaluation uses full-all rows [190000,200000), 32 paired bank/search seeds, four chains and 262,144 moves per chain. The same 10,000 test inputs are repeated for each bank/search seed. All 960,000 returned boards were independently checked; there were no wrong stops. The count energy uses literal unit 1, zero biases, tau 0 and ebar 0 without hidden-label calibration.

Secondary spike minus candidate accuracy: 10.2106 pp, descriptive 95% interval [8.759046875000001, 11.801601562499998].
Secondary exponential minus candidate accuracy: 12.5022 pp, descriptive 95% interval [10.5243671875, 14.739203124999998].
Secondary exponential minus spike accuracy: 2.2916 pp, descriptive 95% interval [-0.18571875, 4.849148437499999].

All earlier positive and inconclusive acquisition studies remain separate and are not pooled. Secondary accuracy intervals are descriptive and unadjusted. Adaptive acquisition does not satisfy the previous independent-uniform-mask theorem. This experiment does not establish universal optimality or equivalence.

[Protocol](protocol-exponential.json) · [Restricted policy](exponential_policy.py) · [Acquisition harness](exponential_acquire.py) · [Independent replay](audit_exponential_acquisition.py) · [Search](exponential_search.py) · [Final audit](audit_exponential.py) · [Public results](../../../results/research/ripper/exponential.json)
