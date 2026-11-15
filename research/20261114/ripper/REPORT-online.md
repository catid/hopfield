# H1s: selecting observations before seeing the current labels

The frozen 25% coverage-improvement target is **inconclusive**. Candidate-focused selection reduces total N300 false extra edges by 32.506%, with a paired-bank 95% bootstrap interval [22.323%, 41.556%]. This uncertainty uses only six fixed banks and one frozen tie stream per bank.

| Policy | N300 extras across six banks | Mean test solved (%) | Wrong stops |
|---|---|---:|---:|
| BP AdamW: outside this acquisition study; not run | — | — | — |
| uniform | 77, 80, 66, 78, 73, 69 | 0.9600 | 0 |
| balanced | 74, 57, 62, 55, 60, 57 | 1.9683 | 0 |
| candidate | 56, 53, 38, 40, 49, 63 | 4.3467 | 0 |

Every policy queries exactly 25 cells per incoming board: 7,500 revealed digits by N300, and 75,000 by N3000 in each bank. The stored complete boards form a simulated label oracle. The policy receives only past joint/equal observation counts and a fixed tie order; it cannot access the current board, source identity, geometry or oracle handle.

Uniform sampling selects the first 25 cells of the shared random permutation. Exposure balancing weights a pair by the reciprocal of one plus its past joint-observation count. Candidate-focused selection additionally removes the weight of pairs already observed equal. The weighted policies greedily select a high-weight group of cells; all ties use the frozen shared permutation.

All 54,000 mask decisions were independently reconstructed from past queried archives. Masks were fixed before current labels were revealed, premature reveals were rejected, and hidden current-row permutations left those masks unchanged. All 54 prefix supports and 108 hidden-value substitution recounts passed. Geometry appears only in the final audit.

Fresh evaluation uses full-all rows [150000,160000), six paired seeds, four chains and 262,144 moves per chain. All 180,000 returned boards were independently checked; there were no wrong stops. The count energy uses literal unit1, zero biases, tau0 and ebar0 without hidden-label calibration.

Secondary balanced minus uniform accuracy: 1.0083 pp, descriptive 95% interval [0.5283333333333333, 1.4884583333333314].
Secondary candidate minus uniform accuracy: 3.3867 pp, descriptive 95% interval [1.8933333333333333, 4.991791666666664].

N1000/N3000 graph outcomes and exposure balancing are descriptive. The acquisition policies adapt to prior observations, so the independent-uniform-mask theorem does not apply directly. This experiment does not establish a universally optimal acquisition policy.

[Protocol](protocol-online.json) · [Restricted policy](online_policy.py) · [Acquisition harness](online_acquire.py) · [Independent replay](audit_online_acquisition.py) · [Search](online_search.py) · [Final audit](audit_online.py) · [Public results](../../../results/research/ripper/online.json)
