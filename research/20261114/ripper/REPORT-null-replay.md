# H1ar: known equality nulls on unchanged permutation-noise evidence

The target of at least 29 exact-law graphs retaining every true edge is **supported_finite_bank_target**: 32/32 banks retain all true edges.

| Equality null | All true edges retained | Exact graphs | Missing true edges | Extras | Final-only reinstated pairs |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — |
| calibrated | 11/32 | 4/32 | 41 | 45 | 966 |
| exact_law | 32/32 | 0/32 | 0 | 14229 | 2385 |
| nominal_iid | 32/32 | 0/32 | 0 | 30864 | 2332 |

All 32 H1ap acquired masks and observed labels are unchanged. The replay changes only the equality-null probability: archived calibrated q0 = 42288940519/64000000000000; exact permutation-law q0 = 9/1600; or nominal iid eta = 0.05, giving q0 = 311/25600. The alternative remains 1/9 and the crossing bound 648000. These known-law values are supplied oracle information, not newly estimated calibration.

Pairs are removed permanently at the first crossing. Final-only threshold differences are reported separately; they do not reinstate sequentially excluded pairs. Every unexposed pair remains retained. The original acquisition masks are replayed rather than reselected using counterfactual exclusions.

For this supplied corruption law, the current mask is fixed before the independent board permutation. Conditional on history and the current clean source board, a queried true pair is equal with probability p = 9/1600. Its LR multiplier has expectation p(q1/q0) + (1-p)(1-q1)/(1-q0). This is exactly one at q0 = p and below one at nominal iid q0 > p; an unqueried pair has multiplier one. This is a conditional martingale/supermartingale statement for this law and predictable masks, not a universal correlated-noise guarantee.

Thresholds use exact rational powers, independently checked by 3,333 Decimal80 endpoint/predecessor tests. Independent reconstruction checked all 115,084,800 cumulative pair states and matched all 103,680 archived calibrated first-crossing paths. Every bank and outcome is retained.

This post-hoc mechanism comparison uses no new observations, calibration fits, source boards, puzzle inputs or GPU work. Exact graph recovery and extra-edge costs are descriptive; the primary concerns true-edge retention only.

[Protocol](protocol-null-replay-corrected.json) · [Independent replay](audit_null_replay_corrected.py) · [Results](../../../results/research/ripper/null_replay.json)

Execution used 19.1671 CPU-seconds and 3.2252 wall seconds within the frozen 120 CPU-second / 60-second ceilings.

A synthetic matrix-indexing preflight failed before any counterfactual outcomes were accessed. The original producer, auditor and inadvertently frozen protocol are preserved. The corrected producer selects the 25 labels first, then forms their explicit outer equality matrix. Separate checked calls verified 1,875 literal matrix entries and 1,488 threshold cases before the corrected protocol was frozen and the first run launched. The scientific recipe and budgets did not change. [Correction and lineage](null-replay-pre-run-correction.json) · [Exact diff](null-replay-pre-run-correction.diff) · [Preserved failure](null-replay-original-preflight-failure.txt) · [Independent correction check](verify_null_replay_correction.py).
