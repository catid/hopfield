# H1ag: canonical compatibility and certified versus capped returns

Post hoc diagnosis using saved H1ag models and boards; no new labels, queries, inference trajectories, tuning or hypothesis tests.

No independent uniqueness proof is established by this diagnostic. Report canonical-solution compatibility, not exact capacity for all valid solutions.

| Evidence law / bound | Canonical zero-energy compatibility (%) | Given–given infeasibility (%) | Correct certified (%) | Correct capped, compatible target (%) | Correct capped, incompatible target (%) | Compatible target misses (%) |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — | — | — |
| fixed_exact | 46.6369 | 6.8016 | 40.2575 | 0.0000 | 9.5853 | 6.3794 |
| mixture_exact | 40.5194 | 8.0288 | 35.3525 | 0.0000 | 10.6419 | 5.1669 |
| fixed_learned | 10.2041 | 21.6944 | 9.4341 | 0.0000 | 10.3053 | 0.7700 |
| mixture_learned | 7.1300 | 25.4328 | 6.5747 | 0.0000 | 9.0944 | 0.5553 |

Canonical compatibility bounds correct canonical zero-energy certificates. Positive given-given conflicts preclude every zero-energy completion respecting the clues. Neither bounds returned-best-board accuracy at positive energy.

Strictly positive reweighting preserves the zero set and all given-given obstructions, but can change positive-energy best-board rankings and capped-return accuracy.

Each arm has 32 bank/search seeds on the same 10,000 puzzles. Percentages average over 320,000 bank–puzzle instances, not 320,000 distinct inputs. A compatible target miss means the search did not return the stored target within the frozen budget; no uniqueness or universal search claim is implied.

[Study report](REPORT-mixture-law.md) · [Diagnostic script](diagnose_mixture_compatibility.py) · [Diagnostic data](../../../results/research/ripper/mixture_law_compatibility.json)
