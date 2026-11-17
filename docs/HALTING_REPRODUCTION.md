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
| WSP-Transformer (zero-violation certificate) (rule-programmed, k=0) | 10.523613 | 0.000000 | 10.523613 |
| WSP-Det (zero-violation certificate) (rule-programmed, k=0) | 14.163513 | 0.000000 | 14.163513 |
| WSP-Assoc (zero-violation certificate) (rule-programmed, k=0) | 100.000000 | 0.000000 | 100.000000 |

Fresh CRAM counts: 1,238,317 halted, 0 incorrect halts, and 1,238,317 solved out of 1,238,358 seed evaluations. Zero-training WSP counts (`k=0` training examples, supplied Sudoku rules): `WSP-Assoc` halts on and solves all `1,238,358 / 1,238,358` seed evaluations (`0` wrong halts); deterministic `WSP-Transformer` and `WSP-Det` certify `43,440 / 412,786` and `58,465 / 412,786` puzzles with `0` wrong halts.

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
| WSP-DeadEnd (fixed-point stop) (rule-programmed, k=0) | 100.000000 | 0.000000 | 100.000000 |
| WSP-Wavefront (fixed-point stop) (rule-programmed, k=0) | 100.000000 | 0.000000 | 100.000000 |
| WSP-Potential (fixed-point stop) (rule-programmed, k=0) | 100.000000 | 0.000000 | 100.000000 |

Fresh counts: 3,000 halted, 0 incorrect halts, and 3,000 solved out of 3,000 seed evaluations (for both fresh CRAM and all three zero-training WSP maze programs).

The released Maze-Unique solver stops at a fixed point; Table 4 labels its CRAM row generically as an energy threshold. This distinction is retained. The zero-wrong-stop observation does not extend to CRAM+z on Maze-Hard: each fresh seed stops incorrectly on maze 106, as documented in the [maze report](MAZE_REPRODUCTION.md). Under zero-training WSP on Maze-Hard (`1,000` cyclic mazes × `3` seeds), purely local `WSP-DeadEnd` halts at fixed points on all `3,000` evaluations with **`100.000000%` wrong halts** (the graph 2-core), whereas geodesic `WSP-Wavefront` and resistive `WSP-Potential` both halt on **`3,000 / 3,000` (`100.000000%`)** with **`0.000000%` wrong halts** and **`100.000000%` solved** (including maze `106`); see [WSP_BENCHMARK_REPORT.md](WSP_BENCHMARK_REPORT.md).

Regenerate frozen CRAM rows without scoring any test input: `python3 scripts/export_halting_comparison.py`. Counts and source hashes are in [halting-comparison.json](../results/reproduction/halting-comparison.json) and [wsp-comparison-tables.json](../results/wsp/wsp-comparison-tables.json).
