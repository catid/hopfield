# More corrupted cells expose a sample-size boundary

The fixed count-gap estimator recovers the clean graph from **3,000 training boards with nine free-cell digits corrupted in every board**, in all six source samples. It therefore uses the same binary energy as clean counting and returns identical outputs: **79.9167% accuracy with zero wrong stops** at the fixed short search budget. The frozen 2-point noninferiority primary is **supported** through exact model/settings identity.

At **300 training boards**, the same nine-cell corruption condition recovers the graph in only two of six samples. Mean accuracy falls to **26.6733%**, with 10,000 wrong stops across 60,000 evaluations. No threshold was retuned after that failure.

| Training boards | Corrupted cells per board | Accuracy | Wrong stops |
|---:|---:|---:|---:|
| — | BP AdamW: outside this count-estimator study | — | — |
| 300 | 0, clean reference | 79.9167% | 0 |
| 300 | 1 | 79.9167% | 0 |
| 300 | 3 | 79.9167% | 0 |
| 300 | 9 | 26.6733% | 10,000 |
| 3,000 | 0, clean reference | 79.9167% | 0 |
| 3,000 | 1 | 79.9167% | 0 |
| 3,000 | 3 | 79.9167% | 0 |
| 3,000 | 9 | 79.9167% | 0 |

Only the final row versus its clean reference is primary. Other rows describe the boundary. There are **48 requested model/seed cells and ten unique searches**: identical supports and all inference settings are explicitly deduplicated within seed. Equal scores for matched graphs are not independent accuracy replications. The zero primary difference/interval reflects this identity, not certainty about future absolute accuracy.

Each row's nine corrupted cells are chosen uniformly without replacement from its officially free cells; each receives a different digit. One- and three-cell conditions use prefixes of the same selections. N300 uses the prefix of N3000. All given clues remain correct. Nine changed cells are **11.11% of the 81 cell labels**, not arbitrary corruption of every label. The estimator receives equality counts from the noisy complete boards; it does not receive the true graph or the corruption locations.

The unchanged heuristic sorts distinct equality counts, chooses the first largest adjacent gap, and takes its floored midpoint. A post-fit graph diagnostic explains the small-sample failures. Three of the six N300 samples have overlapping true-peer and nonpeer count ranges, so **no scalar count threshold** can recover the exact graph. Another sample has a separating threshold, but the largest-gap heuristic selects an unrelated gap near the high-count tail. This does not rule out more informative estimators. At N3000, the chosen thresholds are 186–199, while every true peer has at most 104–114 equalities and every nonpeer has at least 268–284, depending on the sample.

All inference uses the same unit weights, zero unary terms, zero stopping/focus offsets, four chains, and 262,144 moves per chain. The new cohort is `full-all[170000,180000)`, six search seeds and 10,000 common inputs. The independent audit replayed every corruption draw, rebuilt counts and selected thresholds, checked all mappings and source/output hashes, and validated **100,000 unique saved boards**, including exact integer fields and energies. This remains the original official corpus, with reused source banks and a new noise RNG/cohort.

[Frozen protocol](noise_boundary.json) · [preparation](prepare_noise_boundary.py) · [runner](noise_boundary.py) · [independent auditor](audit_noise_boundary.py) · [public summary](../../../results/research/foureyes/noise_boundary/summary.json) · [post-fit separation diagnostic](../../../results/research/foureyes/noise_boundary/separation.csv) · [diagnostic source](noise_separation.py)
