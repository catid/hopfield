# H1aj: oracle diagnosis of the half-budget boundary

This is a separately frozen post-hoc mechanism experiment on the same already-seen H1ai banks and 5,000 puzzles. Its interventions use known Sudoku geometry. It is not a deployable learner or a fresh confirmation.

The three frozen endpoints are evaluated separately; no conjunction or joint dominance claim is made.

- repair_true: **supported_5pp_equivalence**, gain 2.1969 pp, marginal paired two-way 95% interval [1.5443, 2.9475] pp.
- soften_strong_extra: **supported_10pp_gain**, gain 40.6763 pp, marginal paired two-way 95% interval [37.9610, 43.2865] pp.
- delete_weak_extra: **falsified_10pp_gain**, gain 4.1725 pp, marginal paired two-way 95% interval [3.5262, 4.8606] pp.

| Oracle intervention | Canonical accuracy (%) | Correct certified (%) | Correct capped (%) | Canonical compatibility (%) | Given–given infeasibility (%) | Invalid certificates |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — |
| base | 23.5525 | 0.0000 | 23.5525 | 0.0000 | 99.7775 | 0 |
| repair_true | 25.7494 | 0.0000 | 25.7494 | 0.0000 | 99.7775 | 0 |
| soften_strong_extra | 64.2288 | 0.0000 | 64.2288 | 0.0000 | 99.7775 | 0 |
| delete_weak_extra | 27.7250 | 16.6894 | 11.0356 | 18.4669 | 15.1744 | 0 |

The repair arm restores exactly 82 evidence-weak true edge-bank pairs from 1/16 to 1. The softening arm changes exactly 503 strong extra edge-bank pairs from 1 to 1/16. The deletion arm removes exactly 19,445 weak extra edge-bank pairs. An independent row/column/box construction exhaustively checked all 414,720 model weights and verified zero true-edge deletions.

Repair and softening preserve the positive support and zero set. Deleting weak extras expands the zero set and may increase canonical compatibility. All 810 true Sudoku edges per bank remain positive in every arm, so exact zero-energy returns remain sound. These claims do not bound positive-energy capped-return accuracy or establish solution uniqueness.

Observed invalid certificates: 0; unique-search certificates: 26703. The zero-invalid gate is nonvacuous.

All arms retain the same unit1, temperature .4→.31, theta .45, tau0/ebar0, four chains, 262,144-move cap and paired search seeds. Archived baseline trajectories and any identical complete weight/input/settings cells are reused exactly. No new labels, source banks or held-out cohort were consumed.

Of 128 requested cells, 93 require new searches; 35 cells reuse archived baseline results. All 625,000 unique returned boards were independently audited, including 465,000 new boards. New GPU allocation is 8.5113 GPU-min. Every new search passed 150 general/sparse, 132 hidden-target and six independent integer field checks, including 127 unrefreshed moves.

All three intervals use 5,000 paired two-way resamples over 32 banks and the same 5,000 puzzle indices. They are marginal, approximate, and not adjusted for choosing this mechanism study after H1ai. Earlier study results are not pooled as new evidence.

[Protocol](protocol-confidence-oracle.json) · [Oracle policy](confidence_oracle_policy.py) · [Model audit](audit_confidence_oracle_models.py) · [Results](../../../results/research/ripper/confidence_oracle.json)
