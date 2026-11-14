# H1 confirmation: validated results

All 24 jobs and 49,152 saved boards validated. Primary projected-minus-original effect **48.3073pp**, paired two-way bootstrap95% interval **[43.1065, 53.6541]pp**; preregistered5pp hypothesis: **supported_for_frozen_protocol**.

Development rows[512, 2560], 2048 puzzles per run, seeds[0, 1, 2, 3, 4, 5], fixed262144moves/chain and4chains. The frozen protocol bytes are retained; [editorial errata](PROTOCOL_ERRATA.json) explain inherited stale prose. The numerical cohort/arms were unchanged.

| k | Arm | Solved per seed | Mean accuracy (%) | Wrong stops |
|---|---|---|---:|---:|
| — | BP AdamW: outside post-fit protocol | Not run | — | — |
| 17 | original | 294 / 189 / 144 / 208 / 74 / 192 | 8.9600 | 11104 |
| 17 | projected | 1398 / 1402 / 1458 / 1401 / 1403 / 1413 | 68.9697 | 0 |
| 21 | original | 907 / 563 / 636 / 461 / 357 / 998 | 31.9173 | 7346 |
| 21 | projected | 1404 / 1407 / 1397 / 1395 / 1400 / 1417 | 68.5221 | 0 |

Newly evaluated model seeds3–5 alone show **49.9593pp** mean improvement. Seeds0–2 reuse the discovery models on disjoint rows; all6 are included as preregistered.

Intervals are exploratory with few model seeds; these results do not establish performance on an untouched external distribution. Search RNG streams can diverge after model-specific compaction. The saved boards permit full independent validity checks.

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.
