# H1 lowdata: validated results

All 18 jobs and 9,216 saved boards validated. Primary projected-minus-original effect **8.2682pp**, paired two-way bootstrap95% interval **[6.9661, 9.5703]pp**; preregistered5pp hypothesis: **supported_for_frozen_protocol**.

Development rows[0, 512], 512 puzzles per run, seeds[0, 1, 2], fixed262144moves/chain and4chains. The frozen protocol bytes are retained; [editorial errata](PROTOCOL_ERRATA.json) explain inherited stale prose. The numerical cohort/arms were unchanged.

| k | Arm | Solved per seed | Mean accuracy (%) | Wrong stops |
|---|---|---|---:|---:|
| — | BP AdamW: outside post-fit protocol | Not run | — | — |
| 1 | original | 0 / 0 / 0 | 0.0000 | 209 |
| 1 | projected | 0 / 0 / 0 | 0.0000 | 1536 |
| 3 | original | 0 / 0 / 0 | 0.0000 | 1528 |
| 3 | projected | 0 / 0 / 0 | 0.0000 | 1536 |
| 10 | original | 6 / 2 / 4 | 0.7812 | 1524 |
| 10 | projected | 143 / 115 / 135 | 25.5859 | 159 |

Intervals are exploratory with few model seeds; these results do not establish performance on an untouched external distribution. Search RNG streams can diverge after model-specific compaction. The saved boards permit full independent validity checks.

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.
