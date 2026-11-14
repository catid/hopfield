# Stopping-rule comparison

Table 4's CRAM rows are complete on all 412,786 Sudoku puzzles and all 1,000 Maze-Unique mazes, each under three seeds. Both tasks have zero incorrect stops. The baseline rows below are copied from the manually reviewed paper and are **published results only**.

Wrong halts uses halted boards as its denominator. Counts pool the three seeds, which evaluate the same test inputs. No inference was rerun for this report.

## Sudoku-Extreme

| Method / provenance | Halted, % | Wrong halts, % of halted | Solved, % |
|---|---:|---:|---:|
| BP AdamW control: outside local training scope | — | — | — |
| HRM (halting head) (published) | 54.5 | 0.62 | 54.7 |
| TRM (halting head) (published) | 85.8 | 2.37 | 83.8 |
| EqR (halting head) (published) | 87.5 | 1.61 | 86.1 |
| TRMP (halting head) (published) | 93.8 | 0.05 | 93.8 |
| EBR (gradient-norm stop) (published) | 99.35 | 0.21 | 99.31 |
| CRAM (energy threshold) (published) | 99.996 | 0.00 | 99.996 |
| CRAM (energy threshold) (fresh local) | 99.996689 | 0.000000 | 99.996689 |

Fresh counts: 1,238,317 halted, 0 incorrect halts, and 1,238,317 solved out of 1,238,358 seed evaluations.

## Maze-Unique

| Method / provenance | Halted, % | Wrong halts, % of halted | Solved, % |
|---|---:|---:|---:|
| BP AdamW control: outside local training scope | — | — | — |
| HRM (halting head) (published) | 21.4 | 0.16 | 40.3 |
| TRM (halting head) (published) | 99.9 | 3.54 | 96.4 |
| EqR (halting head) (published) | 66.0 | 1.06 | 65.3 |
| TRMP (halting head) (published) | 91.2 | 0.73 | 99.2 |
| EBR (gradient-norm stop) (published) | no stopping rule | no stopping rule | no stopping rule |
| CRAM (energy threshold) (published) | 100.00 | 0.00 | 100.00 |
| CRAM (fixed-point stop) (fresh local) | 100.000000 | 0.000000 | 100.000000 |

Fresh counts: 3,000 halted, 0 incorrect halts, and 3,000 solved out of 3,000 seed evaluations.

The released Maze-Unique solver stops at a fixed point; Table 4 labels its CRAM row generically as an energy threshold. This distinction is retained. The zero-wrong-stop observation does not extend to CRAM+z on Maze-Hard: each fresh seed stops incorrectly on maze 106, as documented in the [maze report](MAZE_REPRODUCTION.md).

Regenerate without scoring any test input: `python3 scripts/export_halting_comparison.py`. Counts and source hashes are in [halting-comparison.json](../results/reproduction/halting-comparison.json).
