# Ripper research evidence

A simpler control now explains the recovered structure: **no optimizer is needed**. Connecting cells that never share a digit in the augmented training solutions recovers the same graph on all 18 datasets. On a new matched 10,000-puzzle cohort, this count graph solves **80.44%**, only **−0.05 points** from the learned normalized models (approximate paired 95% interval **−0.47 to +0.38**), with zero wrong stops. See the [count-control report](../../../research/20261114/ripper/REPORT-count.md) and [audited inference data](count_search.json). This depends on the inherited augmented data; it does not establish that ten unaugmented puzzles suffice.

The latest fixed-cohort result is **81.40%** solved with normalized k10 models, compared with **34.90%** for canonical repulsion and **1.01%** for unchanged fitted CRAM. Normalization improves over repulsion by **46.51 percentage points** (approximate paired 95% interval **43.70–49.13**). All methods use the same short cap of 262,144 moves per chain and four chains. Six source fits each use **10 unique base puzzles plus 99 stored symmetry variants per base**. See [finaltest.json](finaltest.json), [per-seed data](finaltest_per_seed.csv), and the [detailed report](../../../research/20261114/ripper/REPORT-finaltest.md).

The full method averages over digit labels, removes attractive equal-digit couplings, selects support at the largest learned coefficient gap, makes retained weights uniform, and normalizes temperature/focus to that weight. It receives no additional Sudoku peer graph. All recovered graphs are identical up to scale; repeated source fits do not create independent structural evidence. The inherited augmentation pipeline already uses Sudoku symmetries.

The initial confirmed result was digit-label symmetry projection: **+48.31 percentage points** (approximate paired 95% interval **43.11–53.65**) over unchanged fitted CRAM parameters, on new development rows 512–2559, k17/21 and six model/search seeds. The fixed search budget is 262,144 moves per chain, four chains; this is not the paper main-result budget.

| Method/control | Scope |
| --- | --- |
| BP AdamW | Outside this post-training intervention protocol; not run |
| Original CRAM | Measured matched controls, same models, puzzle rows and move budgets |
| Digit-symmetric CRAM | Mean diagonal/off-diagonal pair tables and digit-mean unary terms; no supplied Sudoku peer graph |
| Canonical repulsive CRAM | Drop the constant energy gauge, then clip negative equal-digit couplings |

The preregistered repulsion hypothesis of at least +10 points was falsified: observed **+1.87 points**, interval **0.86–2.82**. Repulsion removed all observed wrong stops in its study. All 18 repulsive models also satisfy a post-hoc sufficient exact-energy validity condition; this is not a worst-case floating-point drift guarantee. Low-data k1/3 models still solve zero puzzles and projection worsens their stopping behavior.

`summary.json` contains grouped plot inputs, hypotheses, decisions, confidence intervals, scope and hashes. `per_seed.csv` is the common source for per-seed plots. The four `audit-*.json` files preserve compact audited outcomes. `certificate.csv` and `certificate.json` retain the post-hoc energy-gap evidence and numerical caveat.

Frozen protocols and implementation are under [research/20261114/ripper](../../../research/20261114/ripper/). Detailed reports there describe matched controls, saved-board validation and protocol errata. Large saved boards, fitted parameters, training examples and source dataset files remain under ignored `artifacts/` and `upstream/` paths; a source-only checkout does not include these inputs. Reconstruct them using the repository reproduction/data setup and the frozen per-input SHA256 manifests before running the research workers. The research entrypoints intentionally require matching archived inputs and Ripper host ownership.

The compact exports also include the audited sparse-backend and fixed held-out studies described below.

The sparse backend audit now reports **2.824×** geometric-mean warm inference speedup with exact measured state/board parity; see `sparse.json` and `sparse_timings.csv`. The fixed held-out study reports **+47.83pp**,95% interval **42.72–52.85pp**, with **0** repulsive wrong stops. See `heldout.json` and `heldout_per_seed.csv`.

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.

The mechanism studies preserve both positive and negative outcomes. Pruning alone failed the proposed 20-point improvement: **−0.21 points**, interval **−1.52–1.06**. Uniformization helped, and the separate scale intervention then improved k10 by **30.03 points**, interval **26.29–33.99**, while k17/k21 met the frozen ±5-point equivalence criterion. See [gap.json](gap.json), [scale.json](scale.json), and their matching per-seed CSV files.

All 540,000 returned boards in the latest test were independently checked against row, column, box and clue constraints, and matched to the designated source dataset slice. The normalized models made zero wrong stops in 180,000 evaluations. This remains a fixed short-budget study within the official corpus, not an external benchmark or the paper's full-budget accuracy evaluation.

The [one-base recovery audit](../../../research/20261114/ripper/REPORT-onebase.md) sharpens the source of structure: one solved puzzle plus 99 stored spatial variants recovers the full graph in all six groups. At 63 variants, five groups still retain one extra edge. Its final graph and search arguments exactly match the existing count baseline, so inference is reused explicitly instead of rerun.

The [augmentation ablation](../../../research/20261114/ripper/REPORT-augmentation.md) exposes a boundary of the learned largest-gap pipeline: full spatial augmentation reaches 79.12% on its new cohort, while base-only and digit-only fits both solve zero. The fixed pruning rule collapses to a handful of incorrect edges in those arms. This is a failure of that pipeline to transfer, not proof that all learning methods require spatial augmentation.
