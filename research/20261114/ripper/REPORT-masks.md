# H1q: conditional randomization of training clue masks

The registered hypothesis that official masks leave unusually fewer false extra edges is **not_supported_in_prespecified_direction**. The six official N300 banks retain 440 extra edges in total; the 1,024 count-matched uniform-mask draws average 407.825. The one-sided finite-simulation p-value is **0.945366** at the frozen 0.01 threshold.

Each complete training board and its original number of given cells is held fixed. Only the chosen cells are randomized, independently across rows. This is conditional mask randomization, not uniform sampling of Sudoku solutions. All 18,432 prefix supports were independently checked using exact integer-valued matrix counts; the full mask RNG sequence was replayed and source clue counts preserved.

| Mask arm, N300 | Mean solved (%) | Wrong stops |
|---|---:|---:|
| BP AdamW: outside this count mechanism study; not run | — | — |
| official | 1.0733 | 0 |
| uniform | 1.3000 | 0 |

The preselected draw0 search gives uniform minus official 0.2267 percentage points, with a descriptive 95% paired bootstrap interval [-0.36166666666666664, 0.86]. These are six seeds on fresh full-all rows [120000,130000), four chains and 262,144 moves per chain. No mask stream was selected using graph or accuracy outcomes. All 120,000 returned boards were independently checked.

For each fixed bank, the exact expected number of surviving nonpeers is the sum of products of unexposed-pair probabilities. The conditional expectation was evaluated independently by direct multiplication and log accumulation. The formula and its union bound require independence between row masks, not between cell pairs. Per-bank N1000/N3000 results and Monte Carlo deviations from exact expectation are descriptive.

At N1000, 5,928 of 6,144 randomized banks recovered the exact graph. At N3000, all 6,144 recovered it. The conditional theorem gives a numerical failure bound of 1.254e-10 for any of the six fixed banks to fail under one fresh ideal uniform-mask draw at N3000. This tiny bound comes from the theorem, not from estimating a rare failure probability using zero observed failures.

Counting used only the 324 equal-digit pairs of each complete board, always gated by observing both cells. An independent masked-feature matrix implementation validated every saved support; dense clue-only recounts checked the first and last draw in every bank at all three prefixes, and hidden-value substitution gates passed. Geometry enters the post-fit audit only.

The one-sided test was frozen before outcomes. A nonsignificant result in its stated direction must not be presented as equality or as a confirmatory opposite-direction effect. Uniform masks may define non-unique training puzzles; the held-out test puzzles are unchanged.

[Protocol](protocol-masks.json) · [Mask sampler](mask_count.py) · [Independent count audit](audit_mask_counts.py) · [Search](mask_search.py) · [Final audit](audit_masks.py) · [Conditional theorem](RANDOM_MASK_COVERAGE_THEOREM.md) · [Public results](../../../results/research/ripper/masks.json)
