# H1l: one base puzzle, spatial augmentation, and a count graph

One valid solved puzzle plus its 99 stored spatial symmetry variants recovers the exact 810-edge Sudoku peer graph in **all six frozen groups**, using only a never-equal counter. The optimizer is unnecessary for this recovery. This does not mean one unaugmented example teaches Sudoku: the augmentation pipeline already encodes the Sudoku symmetry group.

| Stored variants | Training boards | Exact groups (of 6) | Median extra edges | Maximum extra edges |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 2106 | 2106 |
| 1 | 2 | 0 | 1827.5 | 1833 |
| 3 | 4 | 0 | 1367.5 | 1393 |
| 7 | 8 | 0 | 784.5 | 823 |
| 15 | 16 | 0 | 277 | 313 |
| 31 | 32 | 0 | 45.5 | 47 |
| 63 | 64 | 1 | 1 | 1 |
| 99 | 100 | 6 | 0 | 0 |

No prefix loses a true peer edge, because a valid Sudoku solution never assigns equal digits to a peer pair. The problem is extra edges between nonpeers whose equality has not yet appeared. At 63 variants, the same-band-or-stack nonpeer orbit is fully covered in every group; each of the five remaining false edges lies in the different-band-and-stack orbit. Group3 already has no extra edges at that prefix.

All six final supports are byte-identical to the previously measured unit-weight count graph. Their training tau and ebar are zero, and unit, temperature, focus, seed, puzzle cohort and move budget are identical. Existing count-search outcomes on full-all[40,000,50,000) are therefore reused with explicit receipt mappings; **no new GPU searches or independent replications are claimed**.

[Frozen protocol](protocol-onebase.json) · [Counter and audit](onebase.py) · [Tracked results](../../../results/research/ripper/onebase.json) · [Prefix data](../../../results/research/ripper/onebase_prefixes.csv) · [Orbit coverage](../../../results/research/ripper/onebase_orbits.csv)
