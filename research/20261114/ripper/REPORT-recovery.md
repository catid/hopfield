# H1w: sample-count graph recovery confirmation

The frozen >=90% exact-recovery-at-N600 target is **inconclusive**: exponential recovered 61/64 exact graphs. Exact binomial one-sided 95% lower and upper limits are 0.883283 and 0.987103, respectively. These are separate one-sided limits, not a 95% two-sided interval.

| Policy | Exact graphs at N600 / 64 | Mean N600 test solved (%) | Wrong stops |
|---|---:|---:|---:|
| BP AdamW: outside this acquisition study; not run | — | — | — |
| uniform | 2 | 65.5506 | 0 |
| candidate | 49 | 79.0320 | 0 |
| exponential | 61 | 80.5077 | 0 |

This is a sample-count recovery confirmation, not a superiority claim over candidate. Exponential was selected by the stated H1v noninferiority decision rule; its decay8/9 and25queriesperboard remain fixed. The exploratory six-old-bank feasibility study was used only to choose a realistic endpoint and is not pooled.

The study draws 64 independent 1000-board banks from the same prior-excluded finite source pool. Sampling is without replacement within each bank; overlap across banks is allowed. There are 63,454 distinct source boards among64,000 bank rows; 546 row instances repeat across banks. There is no prior-bank or within-bank overlap. Binomial uncertainty assumes the independent sampling and tie-stream mechanism.

Every policy queries25cellsperincomingboard, so N600 costs15,000 revealed digits per bank. Only past queried counts enter selection. All192,000 masks were independently replayed, all768prefixsupports and1,536hidden-valuerecounts passed. Stored completions serve only as the simulated oracle; geometry enters the final audit.

Inference evaluates N600 only on the same10,000 new full-all inputs[200000,210000) across64pairedbank/searchseeds. Exact support/settings/seed/input dedup reduces192requestedcells to143unique searches. All1,430,000returnedboards were independently checked, with zero wrong stops. Repeated inference inputs are not counted as independent puzzles. Four chains,262,144movesperchain,unit1,B0,tau0,ebar0 and the frozen search configuration are unchanged.

| Policy | N | Exact graphs / 64 | Total extra edges |
|---|---:|---:|---:|
| uniform | 300 | 0 | 4740 |
| candidate | 300 | 0 | 3332 |
| exponential | 300 | 0 | 1759 |
| uniform | 450 | 0 | 973 |
| candidate | 450 | 1 | 543 |
| exponential | 450 | 16 | 95 |
| uniform | 600 | 2 | 183 |
| candidate | 600 | 49 | 19 |
| exponential | 600 | 61 | 3 |
| uniform | 1000 | 62 | 2 |
| candidate | 1000 | 64 | 0 |
| exponential | 1000 | 64 | 0 |

All construction curves outside the registered exponential-N600 primary and all accuracy comparisons are descriptive and unadjusted.

Secondary candidate minus uniform accuracy: 13.4814 pp, descriptive 95% interval [11.637644531249999, 15.31023046875].
Secondary exponential minus uniform accuracy: 14.9570 pp, descriptive 95% interval [13.10024609375, 16.82800390625].
Secondary exponential minus candidate accuracy: 1.4756 pp, descriptive 95% interval [0.652328125, 2.3650468749999995].

The inherited RNG prose range was clarified before acquisition to cover the64seeds explicitly listed in the frozen protocol. No outcomes or settings were changed. See[pre-acquisition clarification](RECOVERY_PROTOCOL_CLARIFICATION.md).

[Protocol](protocol-recovery.json) · [Acquisition](recovery_acquire.py) · [Replay](audit_recovery_acquisition.py) · [Search](recovery_search.py) · [Audit](audit_recovery.py) · [Public results](../../../results/research/ripper/recovery.json)
