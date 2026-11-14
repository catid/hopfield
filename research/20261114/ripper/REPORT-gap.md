# H1f: prune weak surplus couplings using the learned coefficient gap

At k10, largest-gap pruning changes solve accuracy by **-0.2116 percentage points**, approximate paired95% interval **[-1.5222, 1.0579]**. The frozen≥20point hypothesis is **falsified_for_frozen_protocol**. All110592saved boards were independently checked.

| k | Method | Mean accuracy (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside post-training intervention protocol; not run | — | — |
| 10 | repulsive | 36.0107 | 0 |
| 10 | gap_pruned | 35.7992 | 0 |
| 10 | gap_uniform | 52.7018 | 0 |
| 17 | repulsive | 70.1335 | 0 |
| 17 | gap_pruned | 70.4346 | 0 |
| 17 | gap_uniform | 81.5348 | 0 |
| 21 | repulsive | 70.6868 | 0 |
| 21 | gap_pruned | 70.2718 | 0 |
| 21 | gap_uniform | 81.8441 | 0 |

The gap rule uses sorted positive learned coefficients only, retains values above the midpoint of the first largest gap, and never queries a Sudoku peer graph. The separate uniform arm replaces retained coefficients with their mean. Both retain the same raw-projection temperature/focus scale and recompute only training-derived stopping/focus statistics.

Post-selection graph audit: all18/18 pruned models recover the exact810-edge Sudoku peer graph, including allsix k10fits. Training-weight inspection motivated the pruning rule before new puzzle evaluation; this is an exploratory graph-recovery finding, not independently preregistered discovery.

Fresh development rows[6656,8704), allsix model/search seeds0–5, k10/17/21,262144moves perchain,fourchains. New transformedmodels additionally passed initial+4check exact25field/capturedboard parity gates against GeneralSearch.

[Frozen protocol](protocol-gap.json) · [Runner](gap.py) · [Audit](audit_gap.py) · [Tracked evidence](../../../results/research/ripper/gap.json)

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.
