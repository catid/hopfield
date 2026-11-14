# H3d: learned-certificate-first output selection

**Frozen primary hypothesis: supported.** The primary gain target was at least 0.2 percentage points for the 4,000-update AdamW-decay model. This reanalysis preserves every existing chain, weight, stopping threshold and search operation. It selects the lowest-energy stopped chain if one exists, otherwise the original lowest-energy returned chain. It uses learned flags and energies only; puzzle rules enter independent evaluation afterward.

| Method | Updates | Original N20 (%) | Certificate-first N20 (%) | Gain (pp) | Recovered / regressed input-seed pairs |
|---|---:|---:|---:|---:|---:|
| BP AdamW, decay 0.01 | 4,000 | 99.605916 | 100.000000 | +0.394084 | 77 / 0 |
| BP Adam, no decay | 250 | 99.984646 | 100.000000 | +0.015354 | 3 / 0 |
| BP Adam, no decay | 500 | 99.984646 | 100.000000 | +0.015354 | 3 / 0 |
| BP Adam, no decay | 1,000 | 99.984646 | 100.000000 | +0.015354 | 3 / 0 |
| BP Adam, no decay | 2,000 | 100.000000 | 100.000000 | +0.000000 | 0 / 0 |
| BP Adam, no decay | 4,000 | 100.000000 | 100.000000 | +0.000000 | 0 / 0 |
| SGD + geometry preconditioner | 250 | 99.595680 | 99.877169 | +0.281488 | 55 / 0 |
| SGD + geometry preconditioner | 500 | 99.738984 | 99.877169 | +0.138185 | 27 / 0 |
| SGD + geometry preconditioner | 1,000 | 99.877169 | 99.923230 | +0.046062 | 9 / 0 |
| SGD + geometry preconditioner | 2,000 | 99.984646 | 100.000000 | +0.015354 | 3 / 0 |
| SGD + geometry preconditioner | 4,000 | 99.984646 | 100.000000 | +0.015354 | 3 / 0 |

Primary one-sided 95% bootstrap endpoints for the gain are [0.271252, 0.527151] percentage points. Unique inputs are resampled with their three search repeats clustered inside; these are not independent training seeds.

Every selection identity matched a literal row-by-row stopped-subset implementation. Original identities reproduced saved N20 correctness, and every no-stopped-chain fallback retained the exact original identity. The independent audit checked 4,298,580 returned boards against the complete 724-solution set and givens. Coverage, N1, search moves and search FLOPs remain unchanged because no search or returned chain changed.

Float64 full-board recomputation found 0 stopped flags outside the predefined 0.001 absolute-energy allowance; the largest stopped-certificate deficit was 0. Strict negative-gap counts and extrema are retained per record in the JSON. Capped outputs can be earlier best-energy boards, so their flags are not equated with a direct certificate on the returned state. The learned condition is not a general puzzle-correctness proof.

There is no additional search. Selection itself has a measured CPU cost: original median 0.134293 ms; certificate-first median 0.417701 ms; overhead 0.283408 ms for all 6,513 inputs with 20 chains (101 alternating measurements after ten warmups each). This excludes I/O and the diagnostic checker and is not a search-speedup claim.

This is an adaptive follow-up on already-used official test outputs. The policy differs from the paper convention, so its results are a derivative experiment rather than a replacement reproduction score. No performance-based setting or checkpoint selection was performed inside this frozen reanalysis.

[Frozen protocol](protocol.json), [selector](driver.py), [independent auditor](analyze.py), [tracked results](../../../../results/research/loki/h3d/summary.json), [per-seed counts](../../../../results/research/loki/h3d/per_seed.csv).
