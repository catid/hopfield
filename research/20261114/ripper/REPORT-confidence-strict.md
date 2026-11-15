# H1ak: a stricter model-derived confidence rule

The frozen fresh-puzzle conjunction is **falsified_conjunction**.

- Strict minus old_rule, required 20 pp: **falsified_20pp_gain**, mean 12.5106pp, marginal paired two-way 95% interval [9.3700,15.4782] pp.
- Strict minus shuffled, required 20 pp: **supported_20pp_gain**, mean 36.1569pp, marginal paired two-way 95% interval [33.6123,38.7417] pp.
- Strict accuracy, required 50%: **falsified_50percent_floor**, mean 36.1569%, marginal two-way 95% interval [33.6123,38.7417]%.
- Invalid certificates: 0. The zero-invalid gate is vacuous because no certificates occurred.

| Weight rule | Canonical accuracy (%) | Correct certified (%) | Correct capped (%) | Canonical compatibility (%) | Given–given infeasibility (%) | Invalid certificates |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — |
| old_rule | 23.6462 | 0.0000 | 23.6462 | 0.0000 | 99.8044 | 0 |
| strict_rule | 36.1569 | 0.0000 | 36.1569 | 0.0000 | 99.8044 | 0 |
| shuffled | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 99.8044 | 0 |

The strict rule assigns weight 1 only when the exact final LR≤1/16, and weight 1/16 otherwise. Excluded pairs stay 0. The old rule uses LR≤1; shuffled preserves the strict weight multiset and uses the same label-independent permutation generated before H1ai observations. No geometry enters assignment.

All 32 H1ai source fits remain fixed and included, costing 29,750 original queries each. No new labels or fits were acquired. The single new threshold was chosen after H1aj, then frozen without a sweep before these fresh puzzle outcomes. This tests transfer to fresh puzzles conditional on the same fits; it is not a fresh-source-bank replication.

Before inference, source-only audits independently classified 45,868 retained pair-bank ratios with Fraction and Decimal80, checked boundary and LR=1 cases, reproduced the old rule, replayed the pre-observation permutation, and checked 311,040 weights. Every structural outcome was retained.

All three rules preserve positive support and the zero set. All observed true edges stay positive. Canonical compatibility and given–given infeasibility restrict certificate capacity, not positive-energy capped-return accuracy. No uniqueness assumption is made.

All bank/search seeds share the same 5,000 fresh puzzles [405000,410000). Source/test complete-board overlap totals 0. 96 requested cells reduce to 96 new unique searches; 480,000 boards audited. New GPU allocation 9.9605 GPU-min, below the 20 GPU-min ceiling.

Unit 1, temp .4→.31,theta .45, tau0/ebar0, four chains and 262,144 moves remain unchanged. Every new search passed 150 general/sparse, 132 hidden-target and 6 independent integer checks, including 127 unrefreshed moves. Three intervals are marginal 95%; no simultaneous coverage or adaptive extension.

[Protocol](protocol-confidence-strict.json) · [Rule](confidence_strict_policy.py) · [Source gate](audit_confidence_strict_models.py) · [Results](../../../results/research/ripper/confidence_strict.json)

Postfit structure: the stricter rule weakens 342 true edge-bank pairs versus 82 under the old rule, and leaves 145 extras strong versus 503. All 19,948 extra edge-bank pairs stay in support. These diagnostics did not filter banks or change the recipe.

Audit correction: the frozen original final auditor stopped on five misspelled schema reads before decoding boards or aggregate outcomes. The original file is preserved. A separately hashed corrected auditor restores those five keys and adds provenance only; an independent text/AST verifier confirms no numerical or statistical changes. No search was rerun. See the [hashed addendum](confidence-strict-audit-correction.json), [exact diff](confidence-strict-audit-correction.diff), [captured error](confidence-strict-audit-original-error.txt), and [public verifier](verify_confidence_strict_correction.py).
