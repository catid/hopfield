# Main results and published baselines

Fresh CRAM measurements are compared with Tables 2 and 8 of the [reviewed paper](../references/2611.17309v1.md). Every baseline value is **published only**. None is a local baseline training, inference or timing measurement.

Published fit times use H200 hardware; fresh fit times use RTX PRO 6000 Blackwell. Fresh times below are seed 0's median of seven complete warm fits, matching CRAM's timing boundary in the paper. The paper's ordinary baseline times include cold initialization and compilation. These columns do not establish a matched local training speedup.

## Sudoku-Extreme

| Method / provenance | Solved, % ± sample SD | Params | Mean FLOPs/puzzle | Fit time |
|---|---:|---:|---:|---:|
| BP AdamW control: outside local training scope | — | — | — | — |
| HRM (published) | 54.62 ±4.18 | 27.0M | 2.20e11 | 1.2 h |
| TRM (published) | 85.73 ±0.74 | 5.0M | 4.11e11 | 2.5 h |
| GRAM (published) | 96.10† | 9.1M | 8.51e12 | 3.2 h |
| EqR (published) | 99.49 ±0.14 | 5.0M | 2.11e14 | 2.0 h |
| FRM (published) | 99.13 ±0.29 | 7.0M | 3.91e12 | 2.9 h |
| TRMP (published) | 99.87 ±0.08 | 32.8K | 4.57e12 | 9.0 h |
| EBR (published) | 99.85 ±0.05 | 1.8M | 3.37e11 | 2.8 h |
| CRAM (published) | 100.00 ±0.00 | 7.7K | 2.02e9 | 0.04 s |
| Hand-set CRAM (published) | 100.00 ±0.00 | 1 | 1.08e8 | none |
| CRAM (fresh local) | 99.996689 ± 0.000779 | 7716 | 2.0298031e+09 | 0.03540196 s |
| Hand-set CRAM (fresh local) | 100.000000 ± 0.000000 | 1 distinct coefficient | 1.0846005e+08 | none |
| WSP-Transformer (rule-programmed, k=0) | 10.523613 (deterministic) | 0 learned (28 distinct; 613,657 entries / 4,216 nnz) | 1.7112189e+09 | none |
| WSP-Det (rule-programmed, k=0) | 14.163513 (deterministic) | 0 learned (3 distinct; 24,836 entries / 2,831 nnz) | 15416844 | none |
| WSP-Assoc (rule-programmed, k=0) | 100.000000 ± 0.000000 | 0 learned (7 distinct; 26,511 entries / 4,485 nnz) | 23814789 | none |

## Maze-Unique

| Method / provenance | Solved, % ± sample SD | Params | Mean FLOPs/puzzle | Fit time |
|---|---:|---:|---:|---:|
| BP AdamW control: outside local training scope | — | — | — | — |
| HRM (published) | 39.97 ±33.90 | 27.0M | 3.01e12 | 7.1 h |
| TRM (published) | 92.23 ±13.37 | 6.8M | 3.84e12 | 15.6 h |
| GRAM (published) | 98.63 ±1.31 | 10.9M | 8.00e13 | 23.2 h |
| EqR (published) | 93.00† | 263.8K | 1.16e14 | 7.2 h |
| FRM (published) | 97.27 ±2.80 | 8.0M | 4.70e12 | 9.7 h |
| TRMP (published) | 99.23 ±0.31 | 60.7K | 1.80e13 | 7.6 h |
| EBR (published) | 89.80 ±6.35 | 1.8M | 1.93e13 | 2.0 h |
| CRAM (published) | 100.00 ±0.00 | 18 | 8.37e5 | 0.13 s |
| Hand-set CRAM (published) | 100.00 ±0.00 | 3 | 8.37e5 | none |
| CRAM (fresh local) | 100.000000 ± 0.000000 | 18 | 836870.85 | 0.10758123 s |
| Hand-set CRAM (fresh local) | 100.000000 ± 0.000000 | 3 distinct coefficients | 837048.85 | none |
| WSP-DeadEnd (rule-programmed, k=0) | 100.000000 ± 0.000000 | 0 learned (3 distinct; 13 entries / 8 nnz) | 1804550.4 | none |
| WSP-Wavefront (rule-programmed, k=0) | 100.000000 ± 0.000000 | 0 learned (5 distinct; 82 entries / 16 nnz) | 18093738 | none |
| WSP-Potential (rule-programmed, k=0) | 100.000000 ± 0.000000 | 0 learned (11 distinct; 84 entries / 19 nnz) | 1.6426682e+08 | none |

## Maze-Hard

| Method / provenance | Solved, % ± sample SD | Params | Mean FLOPs/puzzle |
|---|---:|---:|---:|
| BP AdamW control: outside local training scope | — | — | — |
| HRM (published) | 74.50 | 2.70e7 | 3.01e12 |
| TRM (published) | 85.30 | 6.82e6 | 3.84e12 |
| FPRM (published) | 87.00 | 6.83e6 | 4.78e14 |
| NCA (published) | 89.20 | 4.10e4 | 2.00e12 |
| EBR (published) | 88.30 | 1.77e6 | 1.93e13 |
| CRAM+z (published) | 99.90 ± 0.00 | 2.90e1 | 1.41e7 |
| CRAM+z (fresh local) | 99.900000 ± 0.000000 | 29 | 14058028 |
| WSP-DeadEnd (rule-programmed, k=0) | 0.000000 ± 0.000000 | 0 learned (3 distinct; 13 entries / 8 nnz) | 184291.2 |
| WSP-Wavefront (rule-programmed, k=0) | 100.000000 ± 0.000000 | 0 learned (5 distinct; 82 entries / 16 nnz) | 17110751 |
| WSP-Potential (rule-programmed, k=0) | 100.000000 ± 0.000000 | 0 learned (11 distinct; 84 entries / 19 nnz) | 50624506 |

Sudoku uses all 412,786 test puzzles per seed; both maze tasks use all 1,000 test mazes per seed. All local means and sample standard deviations use seeds 0–2 (`WSP-Transformer` and `WSP-Det` are deterministic forward programs). FLOPs use the paper's arithmetic convention (`FC2`), including unsuccessful searches and all chains. They are logical counts, not measured latency.

Weight-Space Programming (`WSP-*`) rows compile supplied domain rules directly into pure PyTorch tensor weights (`k=0` training examples, `0` learned parameters, `wsp.free_model.static_problems == []`); they belong conceptually with `Hand-set CRAM` rather than data-learned models that infer rules from examples. Warm weight-construction (`build()`) medians on 1 Threadripper CPU thread / 1 Blackwell GPU are `8.65 / 18.79 ms` (`WSP-Transformer`), `1.21 / 4.45 ms` (`WSP-Det`), and `1.06 / 4.47 ms` (`WSP-Assoc`) on Sudoku, and `0.003–0.026 ms` CPU / `0.023–16.26 ms` GPU across the maze programs. Comparison-inclusive `FC2` counts are `1.743e8` for `WSP-Assoc` on Sudoku and `1.816e7` / `5.266e7` for `WSP-Wavefront` / `WSP-Potential` on Maze-Hard. On Maze-Hard, local `WSP-DeadEnd` halts on the cyclic 2-core (`0%`), whereas both `WSP-Wavefront` and `WSP-Potential` solve all `1,000/1,000` mazes including test maze `106`; see [WSP_BENCHMARK_REPORT.md](WSP_BENCHMARK_REPORT.md).

† The paper substitutes the original authors' GRAM Sudoku and EqR Maze-Unique accuracy where its own baseline reproduction scored lower. Other columns still describe the CRAM paper's runs. Its TRMP lattice footnote reports a separate 98.90% Sudoku model, 16k parameters and 13m28s warm fit; the table uses the main ACT model. Hand-set parameter counts mean distinct coefficient values. Fresh Sudoku has 7,716 / 7,750 / 7,761 contributing coefficients by seed; its row shows seed 0.

Maze-Hard baseline accuracies were published by the original authors; counts were reported by the CRAM paper. NCA uses official A* targets, while fresh CRAM+z uses the released BFS target convention. Every local CRAM+z seed has one wrong stop on maze 106; see the [maze audit](MAZE_REPRODUCTION.md).

Exact source cells, measured summaries and SHA-256 hashes are in [main-comparison.json](../results/reproduction/main-comparison.json) (frozen CRAM reproduction) and [wsp-comparison-tables.json](../results/wsp/wsp-comparison-tables.json) (WSP extension, exported by `python3 scripts/export_wsp_comparisons.py`). Regenerate the frozen CRAM table from saved reproduction results with `python3 scripts/export_main_comparison.py`.
