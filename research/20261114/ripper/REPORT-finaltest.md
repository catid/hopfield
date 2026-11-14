# H1h: independent confirmation of the fixed normalized method

On the new fixed test cohort, normalized k10 improves accuracy by **46.5050 percentage points** over canonical repulsion, approximate paired 95% interval **[43.7033, 49.1300]**. The frozen ≥30-point hypothesis is **supported_for_frozen_protocol**. There were **0 wrong stops** across 180,000 normalized evaluations.

All 540,000 saved boards were independently checked by counting digits in rows, columns and boxes and preserving clues. Saved solutions and clue masks were compared with the specified source dataset slice. All 18 normalized trajectory checkpoints and their capture arrays were verified for later continuation.

| Training puzzles | Method | Mean accuracy (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside post-training intervention protocol; not run | — | — |
| 10 | original_general | 1.0117 | 59359 |
| 10 | canonical_repulsive_sparse | 34.8950 | 0 |
| 10 | normalized_sparse | 81.4000 | 0 |
| 17 | original_general | 9.3867 | 53975 |
| 17 | canonical_repulsive_sparse | 69.8583 | 0 |
| 17 | normalized_sparse | 81.6350 | 0 |
| 21 | original_general | 32.7883 | 35467 |
| 21 | canonical_repulsive_sparse | 69.9700 | 0 |
| 21 | normalized_sparse | 81.3783 | 0 |

The method uses digit-label averaging, nonnegative canonical equal-digit couplings, largest-gap support selection, uniform retained weights, and temperature/focus normalized to that common weight. It uses only learned parameters and training statistics; no Sudoku peer graph is supplied. Every transformation and parameter was fixed before this cohort was evaluated.

All recovered normalized graphs are identical up to a global coupling scale. Repeated source fits do not create independent structural evidence; transformed-model variation is largely search-seed and numerical variation.

Protocol: full-all rows [30,000, 40,000), k10/17/21, six seeds, four chains, 262,144 moves per chain. This is a fixed short-budget test within the same official corpus, not a new external distribution.

[Frozen protocol](protocol-finaltest.json) · [Runner](finaltest.py) · [Audit](audit_finaltest.py) · [Tracked results](../../../results/research/ripper/finaltest.json)

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.
