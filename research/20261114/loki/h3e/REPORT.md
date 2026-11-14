# H3e: fixed-recipe transfer to 12-Queens

**Frozen primary hypothesis: supported.** No-decay Adam at the fixed 250-update prefix was compared with preconditioned SGD at 4,000 updates using certificate-first selection. The n=10 learning rates, 4,000-update cosine horizon, normalized temperature and 20,000-move budget were transferred without n=12 tuning. All arms were fitted from scratch on the same 100,000 unique partial inputs and all their compatible completion labels.

| Method | Updates | Original N20 (%) | Certificate-first N20 (%) | Coverage (%) | Wrong stops | Capped chains | Mean moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW, decay 0.01 | 4,000 | 99.788411 | 99.788411 | 54.172327 | 0 | 212,163 | 13320.23 |
| BP Adam, no decay | 250 | 100.000000 | 100.000000 | 71.566672 | 0 | 19,955 | 4878.40 |
| BP Adam, no decay | 1,000 | 100.000000 | 100.000000 | 72.323870 | 0 | 15,655 | 4508.89 |
| BP Adam, no decay | 4,000 | 100.000000 | 100.000000 | 71.632876 | 0 | 32,272 | 6000.44 |
| SGD + geometry preconditioner | 4,000 | 99.926758 | 99.926758 | 72.493537 | 0 | 14,569 | 4322.40 |

Primary accuracy difference: +0.073242 percentage points, one-sided 95% bootstrap endpoints [0.024414, 0.122070]. Coverage difference: -0.926865 points, endpoints [-1.076949, -0.769524]. The exact upper probability of any paired accuracy regression is 0.036562% (0 affected inputs). Margins are 0.1 point accuracy and 2 points coverage. Inputs, with all three search repeats, are the resampling units.

The 8,192 held-out inputs are disjoint from every fitting input. They contain 6 one-clue, 305 two-clue and 7,881 three-clue inputs, proportionally sampled from the enumerated universe. The full universe has 206,628 completable inputs and 14,200 distinct complete solutions. Two independent enumeration algorithms and direct row/column/diagonal checks agreed on all solutions. Complete labels can repeat across distinct partial inputs, matching the GRAM task; this is not a split by complete solution.

As a post hoc scale reference, independent uniform sampling from each input's valid complete solutions would have expected pooled coverage 76.371168%, because duplicate samples count once. This is an offline oracle reference, not an upper bound or a learned competitor. It did not affect the protocol or decision.

Each selected input contributes every compatible completion, totaling 2,049,002 fitting rows. Exact per-solution free-cell counts reduce the mathematical PL objective to 14,200 solution rows. An independent float64 full-row fixture matched loss and gradients; the full-data GPU check against uncompressed upstream PL passed before training. Float32 reduction order changes, so bitwise equality of complete training trajectories is not claimed.

The bounded optimization check measured median forward/backward times 79.262040 ms uncompressed and 0.393050 ms aggregated (201.66× for this operation only). Loss error was 1.19e-07; maximum gradient error was 7.45e-07. Three measured calls followed one warmup. This is not an end-to-end fit or search speedup; aggregate-path memory figures include still-resident full-data buffers.

The independent audit checked all 2,457,600 saved boards against the exact completion set and givens, recomputed original and certificate-first selection identities, reconstructed every training free-cell count, and checked all 8,192 completion counts. Calibration was independently recomputed in float64. Full per-clue counts and certificate-margin diagnostics are retained.

This is a new generated n=12 domain and fixed-recipe transfer study, not an official GRAM n=12 benchmark reproduction. There is one deterministic trajectory per optimizer and three inference seeds. No held-out setting selection occurred. Certificate-first is a derivative output policy; original-selection outcomes remain separately visible.

[Frozen protocol](protocol.json), [data generator](prepare.py), [runner](driver.py), [auditor](analyze.py), [tracked results](../../../../results/research/loki/h3e/summary.json), [per-seed counts](../../../../results/research/loki/h3e/per_seed.csv), [per-clue counts](../../../../results/research/loki/h3e/per_clue_seed.csv).
