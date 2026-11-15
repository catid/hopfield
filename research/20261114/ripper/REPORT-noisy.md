# H1y: noisy adaptive acquisition

The frozen >=10pp exponential-e-process versus uniform-e-process accuracy target is **supported_at_least_10pp_gain**. The gain is36.0209pp,paired two-way95%bootstrap interval[33.0065,39.1550]pp.

| Policy | Mean solved (%) | Wrong stops / 320,000 repeated evaluations | Exact graphs / 32 | Missing true edges | Extra edges |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this mechanism study; not run | — | — | — | — | — |
| uniform_e | 42.6272 | 0 | 0 | 0 | 291 |
| exponential_e | 78.6481 | 0 | 20 | 0 | 12 |
| exponential_naive | 0.0000 | 320000 | 0 | 12065 | 0 |

The32independent1500-board source-bank draws contain47,691distinct complete boards among48,000rowinstances. Cross-bank overlap is allowed; all260,724previoussourceboards are excluded. Each bank has its own fixed source/noise/tie stream.

The label oracle consists of100%complete stored boards. Each policy receives only25queriednoisylabelsperrow (37,500perbank). Every current mask is finalized before any current-row noisy label is exposed. Each queried label independently flips with eta=.01 to a uniformly different digit. The known law is supplied; eta and q1=1/9 are not fit.

The e-process excludes pairs only at their first likelihood-ratio crossing of324000. Its predictable-mask martingale and Ville/union proof bounds deleting any trueedge by1%perlearnedmodel. This does not control extras,exactrecovery,solveaccuracy,or arbitrary learned-model stopping certificates. The exposure score remains(8/9)^m until exclusion. The naive control instead excludes after its first observed equality and lacks this protection.

All144,000masks,466,560,000log-LRvalues, noise realizations,96supports and311,040first-crossing paths were independently checked before/after search. Exact rational crossing decisions are independently validated with70-digit Decimal calculations.

Inference uses the same10,000fresh full-all inputs[210000,220000) across32paired bank/searchseeds; these are repeated evaluations,not320,000distinctpuzzles. Exact support/settings/seed/input dedup reduces96requestedcells to96searches. All960,000returnedboards were independently checked. Fourchains,262,144movesperchain,unit1,B0,tau0,ebar0 remain fixed. Wrong stops are retained as measured outcomes.

The old six-bank pilot selected theN1500endpoint only and is not pooled. Structural and other accuracy comparisons are descriptive. A support containing all trueedges has a valid integer zero-energy certificate under preserved clues; budgeted best-board returns have no such guarantee.

[Protocol](protocol-noisy.json) · [Filtration proof](NOISY_ACQUISITION_EPROCESS.md) · [Policy](noisy_policy.py) · [Acquisition](noisy_acquire.py) · [Replay](audit_noisy_acquisition.py) · [Search](noisy_search.py) · [Audit](audit_noisy.py) · [Public results](../../../results/research/ripper/noisy.json)
