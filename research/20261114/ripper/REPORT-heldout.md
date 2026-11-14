# H1e: fixed held-out canonical repulsion test

The fixed canonical-repulsive method improves mean solve accuracy by **47.8317 percentage points**, approximate paired95% interval **[42.7248, 52.8475]**. The frozen≥5point hypothesis is **supported_for_frozen_protocol**. Repulsive wrongstops: **0** across120000model-puzzle evaluations.

All250000saved boards were independently checked by digit counts in rows, columns and boxes plus preservation of clues. The prespecified10000-puzzle k17seed0 sparse/general integration pair has identical full returned boards, stop times, and24finalstate fields excluding elapsed walltime.

| k | Method | Mean accuracy (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside post-training intervention protocol; not run | — | — |
| 17 | original_general | 9.6333 | 53877 |
| 17 | canonical_repulsive_sparse | 69.0517 | 0 |
| 21 | original_general | 32.7617 | 35418 |
| 21 | canonical_repulsive_sparse | 69.0067 | 0 |

This uses fixed full-all rows[20000,30000), k17/21, six model/search seeds0–5,262144moves perchain,fourchains. Models, transformations, temperatures and stopping parameters were frozen before cohort results. This cohort is held out from this research development sequence, not claimed to be a pristine external distribution. No test tuning or modelselection occurred.

[Frozen protocol](protocol-heldout.json) · [Runner](heldout.py) · [Audit](audit_heldout.py) · [Public results](../../../results/research/ripper/heldout.json)
