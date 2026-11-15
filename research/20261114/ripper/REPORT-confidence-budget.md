# H1ai: half-budget confidence weighting on fresh puzzles

The frozen accuracy conjunction is **falsified_conjunction**. The earlier source-only zero-weak-true-edge hypothesis failed separately and did not filter banks.

- Evidence minus uniform, target 10 pp: **supported_10pp_gain**, mean 23.5525 pp, marginal two-way 95% interval [20.8600, 26.3450] pp.
- Evidence minus shuffled, target 10 pp: **supported_10pp_gain**, mean 23.5525 pp, marginal two-way 95% interval [20.8600, 26.3450] pp.
- Evidence accuracy, target 70%: **falsified_70percent_floor**, mean 23.5525%, marginal two-way 95% interval [20.8600, 26.3450]%.
- Zero invalid certificates: **passed**, observed 0. No certificates occurred in any arm, so this gate is vacuous in this study.

Primary accuracy includes all canonical exact matches, both certificates and capped returns. Each interval has marginal nominal 95% coverage; no simultaneous-coverage claim.

| Weights | Canonical accuracy (%) | Correct certified (%) | Correct capped (%) | Canonical compatibility (%) | Given–given infeasibility (%) | Invalid certificates | Noncanonical valid returns |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — | — |
| uniform | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 99.7775 | 0 | 0 |
| evidence | 23.5525 | 0.0000 | 23.5525 | 0.0000 | 99.7775 | 0 | 0 |
| shuffled | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 99.7775 | 0 | 0 |

One shared M1000/N1110 fit costs 29,750 queries (2,000 calibration and 27,750 training). Each of the 32 fresh banks is evaluated with three positive weight assignments: uniform 1; evidence 1 if exact final LR≤1 else 1/16; shuffled with the same multiset and a label-independent permutation fixed before observations.

The earlier source-only audit retained every true edge but weakened 82 true edge-bank pairs. It weakened 19,445 of 19,948 extra edge-bank pairs (97.4784%), leaving 503 extras strong. These postfit structural findings neither tuned the weights nor removed banks. Positive weights preserve the zero set; weak true edges do not invalidate zero-energy certification while all true edges remain present.

Calibration coverage is 32/32 with 0 fallbacks; every outcome is retained. Source roles are globally disjoint; all 1,729,093 prior Ripper source rows were excluded. Source audit verified 66,498 distinct complete boards, and test/source complete-board overlap totals 0.

The source stage independently replayed 35,520 masks, 115,084,800 LR values, 103,680 first-crossing paths and 96 exact weight assignments. Each fresh search checked 150 general/sparse comparisons, 132 hidden-target comparisons and six independent integer field/energy/focus checks, including 127 unrefreshed moves. Every certificate was independently checked in scaled integer energy.

The same 5,000 fresh puzzles [400000,405000) are used across all 32 bank/search seeds, not 160,000 distinct inputs. All 96 requested cells reduced to 96 unique searches; 480,000 returned boards were audited. GPU allocation is 10.0741 GPU-min; no new standalone GPU preflights were run. H1ah preflights are reused unchanged, and all per-search checks are charged in this study.

Temperatures .4→.31, strong unit1, theta .45, tau0/ebar0, four chains and the 262,144-move cap remain unchanged. No adaptive extension, per-model normalization or outcome-based tuning occurred. Canonical compatibility and given–given infeasibility restrict the relevant certificate outcomes, not positive-energy capped-return accuracy. Uniqueness was not independently established.

[Inference protocol](protocol-confidence-budget-inference.json) · [Source protocol](protocol-confidence-budget-source.json) · [Source-only report](REPORT-confidence-budget-source.md) · [Results](../../../results/research/ripper/confidence_budget.json)
