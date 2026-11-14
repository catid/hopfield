# H1g: normalize temperature and focus to the retained coupling level

At k10, scale normalization changes accuracy by **30.0283 percentage points**, approximate paired95% interval **[26.2860, 33.9895]**. The frozen≥20point hypothesis is **supported_for_frozen_protocol**. All46656saved boards were independently checked, and every model weight exactly matches its frozen uniformized input.

| k | Thermal/focus unit | Mean accuracy (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside post-training intervention protocol; not run | — | — |
| 10 | inherited_unit | 51.2088 | 0 |
| 10 | normalized_unit | 81.2371 | 0 |
| 17 | inherited_unit | 80.6842 | 0 |
| 17 | normalized_unit | 80.8513 | 0 |
| 21 | inherited_unit | 80.3755 | 0 |
| 21 | normalized_unit | 81.1728 | 0 |

The normalized arm uses the common retained coupling as its temperature/focus unit. The inherited arm retains the earlier raw-projection hbar. Weights, graph, training-only tau/ebar, random seeds and move budget are unchanged. No Sudoku geometry is supplied.

At k17, the effect is 0.1672 points,95% interval[-1.3374, 1.7233], decision: **equivalent_within_5pp** under the frozen±5point equivalence margin.
At k21, the effect is 0.7973 points,95% interval[-0.8359, 2.4566], decision: **equivalent_within_5pp** under the frozen±5point equivalence margin.

The fresh development interval was[8704,10000),1296puzzles permodel. All development rows have now been used across the research sequence.

[Frozen protocol](protocol-scale.json) · [Runner](scale.py) · [Audit](audit_scale.py) · [Tracked results](../../../results/research/ripper/scale.json)

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.
