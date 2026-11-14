# A matched transformation closes the raw-data PL gap

The previously fixed transformation pipeline brings the digit-relabeling PL model trained on **100 raw solutions** within the frozen **2-point noninferiority margin** of direct counting. On 10,000 new follow-up puzzles, transformed PL solves **80.4583%**, compared with **80.7933%** for counts. The paired difference is −0.3350 percentage points, with a two-sided 95% bootstrap interval [−1.2252, +0.4084]. Both have zero wrong stops. The primary hypothesis is **supported**.

This changes the interpretation of the [earlier raw-recipe comparison](../ripper/REPORT-raw-compare.md): its poor PL results do not establish that optimizer training cannot recover a useful energy. Digit symmetry, graph selection and thermal calibration are consequential. Counting still supplies a simpler route to the same structure without optimization.

| Raw training solutions | Method | Mean accuracy | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW | Outside this fixed-PL-recipe comparison | — |
| 30 | Direct counts | 4.5933% | 0 |
| 30 | Transformed PL, no augmentation | 0% | 59,994 |
| 30 | Transformed PL, digit relabeling | 0% | 60,000 |
| 100 | Direct counts | 80.7933% | 0 |
| 100 | Transformed PL, no augmentation | 0% | 60,000 |
| 100 | Transformed PL, digit relabeling | 80.4583% | 0 |
| 300 | Direct counts, reused identical N100 searches | 80.7933% | 0 |
| 300 | Transformed PL, no augmentation | 0% | 60,000 |
| 300 | Transformed PL, digit relabeling | 80.8417% | 0 |

All counts are across six seed runs of 10,000 shared puzzles. Only N100 digit-relabeling PL versus counts is primary. Other rows are descriptive. The N100 transformed graphs contain exactly the true 810 edges in four fits; one has an extra edge and one misses an edge. Zero observed wrong stops is not a general certificate proof for these imperfect graphs. At N300 all six digit-relabeling fits recover the exact graph. The no-augmentation PL fits collapse to 1–8 retained edges; the N30 digit-relabeling fits retain 1–3. The failed settings remain reported.

The experiment reuses the exact prior 32-update PL fits without refitting. Its fixed sequence is digit-label projection, nonnegative canonical couplings, first largest positive-coefficient gap, strict retention above the gap midpoint, uniform retained weights, and thermal/focus scale normalization to that coefficient. Training-only stopping and focus statistics are recomputed. It supplies no Sudoku peer graph to fitting, transformation or search; graph truth is used only afterward to diagnose structure. This is the unchanged pipeline selected during earlier augmented-data studies.

The new cohort is `full-all[110000,120000)`, with four chains, 262,144 moves per chain and six training/search seed labels. No sampled training completion occurs in this cohort. The corpus was used in the original reproduction; this is a new follow-up slice, not an external dataset. The earlier untransformed comparison used a different slice, so its score change is not a paired before/after effect estimate.

Independent checks validated **480,000 saved boards**, input identities, source/output hashes, unchanged original fits, and all five general/sparse state checks per job. The auditor independently reconstructed every learned retained support using direct float64 digit-orbit averages, and checked coefficients and training calibration. Count N300 results reuse byte-identical N100 models/settings with explicit mappings. The frozen uncertainty rule uses 5,000 paired bootstrap draws over six seed labels and 10,000 shared puzzles.

A [metadata erratum](RAW_TRANSFORM_METADATA_ERRATUM.json) identifies four inherited description/hash fields in the derived protocol that still described the earlier untransformed comparison. It was recorded before outcome interpretation. The new hypothesis/transformation fields, executed runner, budgets and decision rule were unchanged; the original protocol remains preserved.

[Frozen protocol](raw_transform.json) · [runner](raw_transform.py) · [independent auditor](audit_raw_transform.py) · [public summary](../../../results/research/foureyes/raw_transform/summary.json) · [per-seed results](../../../results/research/foureyes/raw_transform/per_seed.csv) · [graph diagnostics](../../../results/research/foureyes/raw_transform/graph_diagnostics.csv)
