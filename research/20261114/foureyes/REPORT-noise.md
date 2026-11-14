# One tolerated equality repairs this label-noise failure

Corrupting **one free-cell digit in three of 300 training solutions** breaks strict never-equal counting: accuracy falls from 82.0083% to 55.1233%, with 16,083 wrong stops in 60,000 evaluations. Retaining an exclusion when its endpoints were equal **at most once** recovers the exact clean graph in all six samples and restores **82.0083% accuracy with zero wrong stops**.

The frozen primary improvement target was +5 percentage points. The measured gain is **+26.8850 points**, with a paired two-sided 95% bootstrap interval [22.4750, 32.1167], so the hypothesis is **supported**.

| Training data and equality threshold | Accuracy | Wrong stops |
|---|---:|---:|
| BP AdamW | Outside this count-estimator comparison | — |
| Clean labels, threshold 0 | 82.0083% | 0 |
| Three corrupted labels, threshold 0 | 55.1233% | 16,083 |
| Three corrupted labels, threshold 1 | 82.0083% | 0 |
| Three corrupted labels, threshold 3 | 82.0083% | 0 |

Threshold 1 is the single primary intervention. The clean and threshold-3 controls are descriptive. These three successful settings produce identical binary supports and inference parameters within every seed. Their searches are explicitly shared; they are not independent accuracy replications. There are **24 requested model/seed cells and 12 unique searches**. All model artifacts and requested-to-actual mappings are retained.

For each of six frozen raw training banks, corruption selects three distinct rows, one officially free cell per row, and a uniformly chosen different digit. It leaves all given clues unchanged. A pair becomes an exclusion when its equality count is at most the prespecified threshold. The constructor receives no row/column/box geometry. All arms use unit weights, zero unary terms, stopping threshold zero, focus offset zero, and the same search settings. In particular, stopping thresholds are not calibrated from corrupted training-label energies.

Evaluation uses `full-all[130000,140000)`: 10,000 shared official puzzles, six search seeds, four chains, and 262,144 moves per chain. The independent auditor replayed every corruption draw, reconstructed every equality count and support, checked all duplicate mappings, and validated **120,000 unique saved boards**. It independently recomputed all saved integer fields and energies and verified that every accepted board has zero energy under its inferred graph. The wrong stops in the strict noisy arm are therefore model errors, not floating-point threshold drift.

This result covers one small, prespecified corruption level on these training banks. It does not establish that threshold 1 is generally robust or optimal. Larger thresholds can add false exclusions when legitimate co-occurrences are rare. The corpus was evaluated in the original reproduction; this is a new research slice rather than an external dataset. Confidence intervals use the frozen 5,000 two-way bootstrap draws over six seed labels and shared puzzle indices.

[Frozen protocol](noise.json) · [corruption and estimator](prepare_noise.py) · [runner](noise.py) · [independent auditor](audit_noise.py) · [public summary](../../../results/research/foureyes/noise/summary.json) · [per-seed rows](../../../results/research/foureyes/noise/per_seed.csv) · [graph diagnostics](../../../results/research/foureyes/noise/graph_diagnostics.csv)
