# H1k: the fixed recovery pipeline depends on spatial augmentation

Full spatial augmentation exceeds digit-only augmentation by **79.1167 percentage points**, approximate paired 95% interval **[78.4250, 79.7667]**, supporting the frozen ≥20-point hypothesis for this pipeline. All 180,000 saved boards were independently checked.

| Training augmentation | Mean accuracy (%) | Wrong stops |
|---|---:|---:|
| BP AdamW: outside this fixed-fit-recipe ablation; not run | — | — |
| base_only | 0.0000 | 59458 |
| digit_only | 0.0000 | 60000 |
| full_spatial | 79.1167 | 0 |

The same 10 unique base puzzles per seed and the same 32-update fit recipe are used. Base-only has 10 training boards; digit-only and full-spatial have 1,000 each. The post-fit pipeline is fixed: digit projection, nonnegative couplings, largest-gap support selection, uniformization and scale normalization. A fresh full-spatial seed0 refit exactly reproduces the frozen parameters.

The largest-gap rule retains only 1–18 edges for base-only fits and 1–4 edges for digit-only fits, instead of the 810 Sudoku edges. Its incorrect stopping behavior explains the failures. This is a transfer failure of the fixed pipeline, not a theorem that every learning method needs spatial augmentation.

The direct count graph for base-only and digit-only data is exactly identical in every seed, as required because digit permutations preserve equality. No redundant digit-only count-graph search was run. Geometry is consulted only in the audit.

Evaluation uses the new fixed full-all interval [50,000, 60,000), six seeds, four chains and 262,144 moves per chain.

[Frozen protocol](protocol-augmentation.json) · [Runner](augmentation.py) · [Audit](audit_augmentation.py) · [Tracked results](../../../results/research/ripper/augmentation.json)
