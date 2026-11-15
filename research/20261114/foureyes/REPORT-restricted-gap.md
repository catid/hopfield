# Restricting the count-gap search helps, but misses its frozen improvement target

**The ≥1-percentage-point improvement hypothesis is inconclusive.** Across 4,096 independent 600-board source-and-corruption draws, restricting the gap selector to gaps below the median pair count improves exact graph recovery from **3,977 to 4,008 banks**: **31 gains, zero losses**, or **+0.756836 points**. The separate conservative one-sided 95% net endpoints are **+0.424776 and +1.072562 points**. The lower bound fails to reach the frozen +1-point target; the upper does not fall below it.

| Estimator | Exact graphs / 4,096 | Recovery |
|---|---:|---:|
| Original largest gap | 3,977 | 97.0947% |
| Largest gap below the median count | 4,008 | 97.8516% |

These are two threshold rules for one count estimator, with no optimizer. The experiment is not a new BP AdamW comparison. Earlier [optimizer controls](REPORT-raw-transform.md) remain separate.

## Prospective design

The [protocol](restricted_gap.json), SHA-256 `50edf1119fd96db604cb2e077eea8f609c4b77eb3e2350b06a7f06673a45b3e4`, was frozen before new bank sampling. Each bank independently samples 600 distinct rows from the same fixed 3,785,228-row eligible corpus as [H4n](REPORT-noise-source-replication.md). Independent RNG streams vary source sampling and corruption. Cross-bank overlap, including overlap with H4n samples, is allowed; these are new draws, not globally disjoint or never-used individual boards. There are 2,457,600 row occurrences and 1,807,228 unique rows.

Every row has nine distinct originally free digits independently changed to other digits; original clues remain unchanged. Both estimators see identical counts. The original rule chooses the first largest adjacent gap in sorted distinct counts. The restricted rule chooses the first largest gap whose upper endpoint is at or below the median of all 3,240 pair counts. Both use the floored midpoint, with threshold zero when no eligible gap exists.

The restriction is an **additional rank prior**: it assumes the forbidden-pair cluster lies below the median count. It uses neither cell geometry nor the exact number of true edges, but it is not prior-free arbitrary graph learning.

An explicitly exploratory check on H4n's previously inspected 512 N600 fits gave 493 versus 502 exact graphs, nine gains and no losses. That check selected the new rule and informed the +1-point target. Its stronger observed effect does not carry into this prospective trial and is not pooled with it. The frozen protocol retains the feasibility artifact's hash and its full result.

## Uncertainty and audit

The paired net lower bound subtracts an upper bound for the loss probability from a lower bound for the gain probability, each using exact Clopper–Pearson α=0.025. A union bound gives at least 95% one-sided coverage without assuming that gains and losses are independent. Reversing the endpoints gives a separate one-sided upper bound. The pair is not a joint two-sided 95% interval. [Independent SciPy verification](../../../results/research/foureyes/restricted_gap/statistics-verification.json) checks the direct binomial-tail inversion.

The [audit](audit_restricted_gap.py) replays every source-bank draw, rereads all selected rows from the official CSV using independent parsing, validates the source solutions and clues, reconstructs every corrupted row, and rebuilds all **4,096 count arrays / 8,192 thresholded models** by direct CPU digit equality. It checks **22,118,400 changed cells**. All thresholds, model supports and identities match. The audit took 36.66 seconds; source preparation took 60.26 seconds. All 64 GPU producer shards completed without failures. Retained live counters progressed from 4 to 53 to 64 completed shards in successive samples; these short count jobs do not require long GPU occupancy.

No solver search was added: the scientific endpoint is exact graph recovery, and exact recovered energies already have audited inference behavior. The result concerns a fixed corpus, independent nine-cell corruption and N600. It neither establishes the 1-point target nor proves the restriction harmless on other data or graph families.

[Public summary](../../../results/research/foureyes/restricted_gap/summary.json) · [Every paired bank](../../../results/research/foureyes/restricted_gap/per_bank.csv) · [Arm totals](../../../results/research/foureyes/restricted_gap/recovery.csv) · [Frozen producer](restricted_gap.py) · [Independent audit](audit_restricted_gap.py) · [Statistical verifier](verify_restricted_statistics.py)
