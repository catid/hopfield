# H1u: fixed spike-mixture query scoring

The frozen 10% improvement-over-candidate target is **supported_at_least_10percent_reduction**. Spike scoring reduces total N300 false extra edges relative to the unchanged candidate policy by 50.604%, with a paired-bank 95% bootstrap interval [46.851%, 54.160%]. This uncertainty uses only 32 fresh disjoint banks and one frozen tie stream per bank.

| Policy | Total N300 extra edges (32 banks) | Mean test solved (%) | Wrong stops |
|---|---|---:|---:|
| BP AdamW: outside this acquisition study; not run | — | — | — |
| uniform | 2423 | 0.9678 | 0 |
| candidate | 1739 | 3.3044 | 0 |
| spike | 859 | 14.4550 | 0 |

Every policy queries exactly 25 cells per incoming board: 7,500 revealed digits at N300 in each bank. The stored complete boards form a simulated label oracle. The policy receives only past joint/equal observation counts and a fixed tie order; it cannot access the current board, source identity, geometry or oracle handle.

Uniform and inverse-exposure candidate selection use the unchanged original policy. The new heuristic assigns prior probability one half to an impossible pair, otherwise a fixed equality probability one ninth. After m observed unequal co-queries with no equality, its predictive score is (1/9)*(8/9)^m/(1+(8/9)^m); an observed equality makes the score zero. Greedy selection uses 18 times this score, preserving real-arithmetic rankings while making initial weights exactly one for correct FP64 tie handling. Zero after an observed equality means there is no value in reobserving a pair already excluded from the count graph; it is not a prediction that future equality has probability zero. The prior and rate were never tuned. This heuristic is not a calibrated posterior guarantee for Sudoku.

All 28,800 mask decisions were independently reconstructed from past queried archives. Masks were fixed before current labels were revealed, premature reveals were rejected, and hidden current-row permutations left those masks unchanged. All 96 prefix supports and 192 hidden-value substitution recounts passed. Geometry appears only in the final audit.

Fresh evaluation uses full-all rows [180000,190000), 32 paired seeds, four chains and 262,144 moves per chain. All 960,000 returned boards were independently checked; there were no wrong stops. The count energy uses literal unit1, zero biases, tau0 and ebar0 without hidden-label calibration.

Secondary candidate minus uniform accuracy: 2.3366 pp, descriptive 95% interval [1.9399609375, 2.7932656249999983].
Secondary spike minus uniform accuracy: 13.4872 pp, descriptive 95% interval [12.0521875, 14.952906249999998].
Secondary spike minus candidate accuracy: 11.1506 pp, descriptive 95% interval [9.663140624999999, 12.677203125].

The earlier positive candidate-policy confirmation remains separate and is not pooled into this improvement test. Secondary accuracy comparisons are descriptive. The acquisition policies adapt to prior observations, so the independent-uniform-mask theorem does not apply directly. This experiment does not establish a universally optimal acquisition policy.

[Protocol](protocol-spike.json) · [Restricted policy](spike_policy.py) · [Acquisition harness](spike_acquire.py) · [Independent replay](audit_spike_acquisition.py) · [Search](spike_search.py) · [Final audit](audit_spike.py) · [Public results](../../../results/research/ripper/spike.json)
