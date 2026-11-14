# H1n: count recovery from raw distinct training solutions

In this explicitly exploratory structure stage, direct never-equal counting recovers the exact Sudoku peer graph at 100 raw training solutions in all six frozen samples. No spatial or digit augmentation is applied. At 30 solutions, the graphs retain 39–52 extra edges. No true peer edge is excluded at any studied size.

| Raw solutions | Exact graphs (of 6) | Extra edges per seed |
|---:|---:|---|
| 1 | 0 | 2106 / 2106 / 2106 / 2106 / 2106 / 2106 |
| 3 | 0 | 1588 / 1581 / 1592 / 1572 / 1596 / 1558 |
| 10 | 0 | 624 / 584 / 601 / 582 / 574 / 582 |
| 30 | 0 | 49 / 44 / 40 / 39 / 52 / 51 |
| 100 | 6 | 0 / 0 / 0 / 0 / 0 / 0 |
| 300 | 6 | 0 / 0 / 0 / 0 / 0 / 0 |
| 1000 | 6 | 0 / 0 / 0 / 0 / 0 / 0 |

Six independent seeded permutations select nested prefixes from 3,831,994 distinct complete boards in the official training CSV. The official associated puzzle mask is retained for each selected board. Three pairs of 1,000-board samples share one board each; all other sample pairs are disjoint. The source dataset is not asserted to be a uniform distribution over all valid Sudoku solutions.

This stage uses no held-out inputs and makes no accuracy claim. The results select meaningful sample sizes for a separately frozen original-pseudolikelihood versus count comparison. The earlier largest-gap pruning failure is not retuned away.

[Frozen exploratory protocol](protocol-raw-count.json) · [Raw data preparation](prepare_raw.py) · [Count/audit code](raw_count.py) · [Tracked results](../../../results/research/ripper/raw_count.json)
