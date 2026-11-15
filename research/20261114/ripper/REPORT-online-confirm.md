# H1t: prospective confirmation of unchanged online acquisition

The frozen 25% coverage-improvement target is **supported_at_least_25percent_reduction**. Candidate-focused selection reduces total N300 false extra edges by 31.130%, with a paired-bank 95% bootstrap interval [26.928%, 35.315%]. This uncertainty uses only 32 fresh disjoint banks and one frozen tie stream per bank.

| Policy | Total N300 extra edges (32 banks) | Mean test solved (%) | Wrong stops |
|---|---|---:|---:|
| BP AdamW: outside this acquisition study; not run | — | — | — |
| uniform | 2496 | 0.8841 | 0 |
| balanced | 2064 | 1.7119 | 0 |
| candidate | 1719 | 3.4359 | 0 |

Every policy queries exactly 25 cells per incoming board: 7,500 revealed digits at N300 in each bank. The stored complete boards form a simulated label oracle. The policy receives only past joint/equal observation counts and a fixed tie order; it cannot access the current board, source identity, geometry or oracle handle.

Uniform sampling selects the first 25 cells of the shared random permutation. Exposure balancing weights a pair by the reciprocal of one plus its past joint-observation count. Candidate-focused selection additionally removes the weight of pairs already observed equal. The weighted policies greedily select a high-weight group of cells; all ties use the frozen shared permutation.

All 28,800 mask decisions were independently reconstructed from past queried archives. Masks were fixed before current labels were revealed, premature reveals were rejected, and hidden current-row permutations left those masks unchanged. All 96 prefix supports and 192 hidden-value substitution recounts passed. Geometry appears only in the final audit.

Fresh evaluation uses full-all rows [160000,170000), 32 paired seeds, four chains and 262,144 moves per chain. All 960,000 returned boards were independently checked; there were no wrong stops. The count energy uses literal unit1, zero biases, tau0 and ebar0 without hidden-label calibration.

Secondary balanced minus uniform accuracy: 0.8278 pp, descriptive 95% interval [0.5959375, 1.0718828125].
Secondary candidate minus uniform accuracy: 2.5519 pp, descriptive 95% interval [2.0806171875, 3.0497031249999997].

The earlier six-bank study remains inconclusive and is not pooled into this confirmation. Exposure balancing is descriptive. The acquisition policies adapt to prior observations, so the independent-uniform-mask theorem does not apply directly. This experiment does not establish a universally optimal acquisition policy.

[Protocol](protocol-online-confirm.json) · [Restricted policy](online_policy.py) · [Acquisition harness](online_confirm_acquire.py) · [Independent replay](audit_online_confirm_acquisition.py) · [Search](online_confirm_search.py) · [Final audit](audit_online_confirm.py) · [Public results](../../../results/research/ripper/online_confirm.json)
