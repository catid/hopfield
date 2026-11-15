# A count-gap rule adapts to the tested noise levels

The fixed largest-gap rule on equality counts recovers the exact clean Sudoku graph at every tested noise level in all six source banks. At the primary endpoint, **all 300 training boards each have one free-cell digit corrupted**. The adaptive count energy and clean count energy are identical, so their inference outputs are shared exactly: **82.0617% accuracy with zero wrong stops** on the new 10,000-puzzle cohort.

The frozen 2-percentage-point noninferiority hypothesis is **supported**. Its measured difference and paired interval are both zero because the models and search settings are byte-identical. This is structural equivalence for these recovered models, not a confidence claim that future test accuracy is known without uncertainty.

| Corrupted training boards | Exclusion threshold | Accuracy | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside this count-estimator study | — | — |
| 0/300 | Strict count, clean reference | 82.0617% | 0 |
| 3/300 | 0 | 56.2467% | 13,959 |
| 3/300 | 1 | 82.0617% | 0 |
| 3/300 | 3 | 82.0617% | 0 |
| 3/300 | Fixed largest-count-gap rule | 82.0617% | 0 |
| 30/300 | 0 | 0.0783% | 59,952 |
| 30/300 | 1 | 76.6750% | 1,932 |
| 30/300 | 3 | 82.0617% | 0 |
| 30/300 | Fixed largest-count-gap rule | 82.0617% | 0 |
| 300/300 | 0 | 0.0000% | 60,000 |
| 300/300 | 1 | 0.0000% | 60,000 |
| 300/300 | 3 | 53.8233% | 17,529 |
| 300/300 | Fixed largest-count-gap rule | 82.0617% | 0 |

The rule sorts the distinct pair-equality counts, takes the first largest adjacent gap, and uses the floored midpoint as the exclusion threshold. There is no edge-count, rank, geometry or percentile restriction. The selected thresholds range from 9 to 12 in this study. These values come from training counts only; no test accuracy or true peer graph enters selection. Lower noise levels and the fixed thresholds are descriptive controls.

Each corrupted row changes exactly one randomly chosen officially free cell to a different digit. The 3/30/300-row conditions use prefixes of one frozen corruption permutation. Thus the strongest condition changes **300 of 24,300 cell labels (1.23%)**, despite affecting every training board. It is not 100% cell corruption. All given clues remain unchanged.

All energies use unit weights, zero unary terms, zero stopping/focus offsets and the same fixed inference settings. Evaluation uses `full-all[140000,150000)`, six search seeds, four chains and 262,144 moves per chain. This is a new follow-up slice of the previously evaluated official corpus.

There are **78 requested model/seed cells and 42 unique searches**. Exact support/settings matches are deduplicated within seed, with explicit mappings. The independent audit replays every corruption draw, reconstructs thresholds and supports, checks all mappings and source/output hashes, and validates **420,000 unique saved boards**, including exact integer fields and energies.

The failure of threshold 1 at larger corruption levels matters: the earlier successful one-equality repair was not universally robust. The gap heuristic also has no general guarantee; overlap between true-constraint and legitimate co-occurrence counts can break it. The next boundary is more corrupted cells per board and larger training samples.

[Frozen protocol](noise_curve.json) · [preparation](prepare_noise_curve.py) · [runner](noise_curve.py) · [auditor](audit_noise_curve.py) · [public summary](../../../results/research/foureyes/noise_curve/summary.json) · [per-seed results](../../../results/research/foureyes/noise_curve/per_seed.csv)
