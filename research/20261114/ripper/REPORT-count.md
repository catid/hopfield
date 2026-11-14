# H1i/H1j: a no-optimizer equality counter recovers the same graph

Connecting cells that never share a training digit recovers the same 810-edge graph as the learned transformations on all 18 augmented training datasets. **No optimizer is needed for this structural recovery.** On the new fixed 10,000-puzzle cohort, the unit-weight count graph solves **80.4433%**, with **0 wrong stops** across 60,000 unique search evaluations.

Its mean accuracy difference from the learned normalized models is **-0.0478 percentage points**, approximate paired 95% interval **[-0.4745, 0.3845]**. The frozen two-point noninferiority result is **noninferior_within_2pp**. All 240,000 saved boards were independently verified against digit-count/clue constraints and the source dataset slice.

| Base training puzzles | Method | Mean accuracy (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside this intervention/no-fit control protocol; not run | — | — |
| 10 | count_sparse | 80.4433 | 0 |
| 10 | normalized_sparse | 80.5767 | 0 |
| 17 | count_sparse | 80.4433 | 0 |
| 17 | normalized_sparse | 80.4717 | 0 |
| 21 | count_sparse | 80.4433 | 0 |
| 21 | normalized_sparse | 80.4250 | 0 |

The count graph is constructed from equality events in training solutions only. The Sudoku peer graph is consulted only afterward for auditing. Each unique base puzzle has 99 stored symmetry variants, including valid Sudoku spatial transformations. Therefore this finding does not show that 10 unaugmented puzzles recover the graph; augmentation supplies task structure. Digit permutations preserve equality events, so digit-only augmentation cannot change a never-equal graph.

All 18 count graphs are identical. For each search seed, graph, unit, tau/ebar, puzzle cohort and all search arguments are byte-identical across training sizes; only six count searches were executed and explicitly reused across k. Learned normalized weights retain different floating coefficients, so all 18 of those searches were measured. Neither duplicate labels nor repeated source fits are counted as independent structures.

Inference protocol: full-all rows [40,000, 50,000), six seeds, four chains, 262,144 moves per chain. This is a short-budget comparison within the official corpus.

[Count protocol](protocol-count.json) · [Count implementation](count_graph.py) · [Inference protocol](protocol-count-search.json) · [Inference runner](count_search.py) · [Audit](audit_count_search.py) · [Tracked support evidence](../../../results/research/ripper/count.json) · [Tracked inference results](../../../results/research/ripper/count_search.json)
