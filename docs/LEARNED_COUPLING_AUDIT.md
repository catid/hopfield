# Learned coupling inspection: Figure 2b

Fresh seed-0 weights recover the positive Sudoku constraint graph and the Maze-Unique dead-end pruning field. The raw Sudoku energy also contains weak negative extra couplings and nonuniform positive magnitudes, so it is not exactly the uniform hand-set rule energy. All inspection takes place after training; the rule graph is used only for this audit, and checkpoint hashes confirm the weights remain unmodified.

The [Figure 2b reproduction](../results/reproduction/figures/learned-couplings-fresh.png) displays the Maze-Unique field, all digit-pair costs from one Sudoku cell, and the global mean same-digit cost matrix. [Coefficient distributions](../results/reproduction/figures/sudoku-coupling-distributions-fresh.png) show both the positive rule costs and negative extras. Matching PDF and SVG files are beside the PNGs. The [JSON audit](../results/reproduction/figures/learned-couplings-fresh.json) contains every statistic, extra pair, and neighborhood check; the corresponding NPZ retains the displayed arrays.

## Sudoku

The independent constraint mask marks two cells as conflicting when they share a row, column, or 3×3 box. It contains 810 unordered pairs, with exactly 20 constrained neighbors per cell.

| Fresh seed-0 quantity | Measured result |
|---|---:|
| Positive pair support: true / false positive / false negative | 810 / 0 / 0 |
| Any-nonzero pair support: true / false positive / false negative | 810 / 253 / 0 |
| Positive same-digit entries on rule pairs | 7,290 / 7,290 |
| Same-digit rule costs: minimum / mean / maximum | 0.492814 / 0.826831 / 1.316800 |
| Negative same-digit entries on non-rule pairs | 426 / 21,870 |
| Most negative extra cost | −0.097753 |
| Different-digit nonzeros, all pairs | 0 |
| Unary-cost nonzeros | 0 |
| All learned nonzero coefficients | 7,716 / 263,169 |
| Number of coupled neighbors per cell | 23–30 |

Every rule pair has all nine positive same-digit costs, but their magnitudes vary across pairs and digits. Within each rule pair, the nine-digit maximum-minus-minimum ranges from `0.014504` to `0.081487`. Mean cost per rule pair ranges from `0.517158` to `1.289937`. The largest extra negative cost is 11.82% of the mean positive rule cost. These extra attractions explain why the inference representation has more than the 20 rule neighbors per cell; they were not removed or rounded away.

The source checkpoint is `upstream/experiments/runs/CRAM-S/full/full-aug-s0/model.pt`, SHA256 `4fad7152714b5b684fc0bee337c52e10d95d455efdb97d114300ea8d3f122136`.

## Maze-Unique

Expanding the learned 2×2 pair tables gives a binary field

`E(on) − E(off) = intercept + Σ_neighbor coupling × neighbor_on`.

After division by the mean magnitude of the four cardinal couplings, the interior field is:

```text
−0.012843   −1.001469   −0.023119
−0.998531   +1.236346   −0.998531
−0.023119   −1.001469   −0.012843
```

The center is a constant intercept, not a self-coupling. A negative field prefers on. Enumerating all 256 binary neighborhoods confirms that a free interior cell prefers off with zero or one on cardinal neighbors and on with at least two, regardless of diagonal states. This is the local dead-end pruning rule. The independently expanded affine field matches the source convolution within `2.09e-6` over these patterns.

The finite-grid boundary was checked separately: 128 edge patterns and 32 corner patterns produce 64 disagreements with the same simple rule because missing neighbors contribute neither off-state cost nor coupling. These are out-of-benchmark configurations. All boundary cells in the 1,000 training, 100 development, and 1,000 test mazes are clamped walls, so every free cell in this benchmark has the full 3×3 window. The audit therefore supports the claimed rule on the benchmark's free cells and does not imply unrestricted boundary generalization.

The source checkpoint is `artifacts/fresh_maze/maze_unique/seed0/model.pt`, SHA256 `c2a8a02d69fe3357accb98e1497457721a97f8ab312e25f247ed9d85cddc9e18`.

## Reproduction

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  upstream/.venv/bin/python scripts/inspect_learned_couplings.py
artifacts/plotting-venv/bin/python scripts/render_learned_couplings.py
```

The CPU audit uses the pinned experimental environment; plotting uses the separate task-owned matplotlib environment. Published source and result records are unchanged.

## Hand-set energy figure

The [Figure 5 reproduction](../results/reproduction/figures/hand-energies-fresh.png), also saved as PDF and SVG, reconstructs the official hand energies without training or new inference. Sudoku's identity cost appears on exactly 810 row/column/box pairs, each cell having 20 neighbors. Maze-Unique has effective cardinal/diagonal/intercept values `−1`, `−0.1`, and `+1.5`. The 8×8 N-Queens graph contains 728 attack pairs at cost `+1`, with unary queen cost `−0.5`. Graph-coloring edge tables are the 3×3 identity and non-edge tables are zero. These are 1, 3, 2, and 1 distinct nonzero energy values respectively, rather than counts of stored factor entries.

Supports are checked against independently constructed task-rule masks, and [hand-energies-fresh.json](../results/reproduction/figures/hand-energies-fresh.json) retains source hashes and exact arrays. Reproduce with `CUDA_VISIBLE_DEVICES='' upstream/.venv/bin/python scripts/inspect_hand_energies.py` followed by `artifacts/plotting-venv/bin/python scripts/render_hand_energies.py`. The first command also verifies the lower-empty-board-energy claim on the 100 development mazes for each of the three fresh learned Maze-Unique models.
