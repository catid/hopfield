# H3bx: fresh matched-prior confirmation of deeper handoff

**The frozen 2% total-membership-call primary is supported.** Across all 4,096 fresh paired cases, the fixed deeper policy used **9,975 actual calls versus 10,238** for the separately executed original baseline, saving **263 calls / 2.568861%**. The exact test is `997,500 <= 1,003,324`. The passing maximum derived from the freshly executed baseline is 10,033, leaving a 58-call margin. Every paired exact flag agrees: 4,083 exact identifications and 13 failures retained. The fresh success/failure count was reconstructed from the new routes and was not assumed from H3bw.

![Fresh paired total-call endpoint](../../../../results/research/loki/h3bx/fresh-paired-total-cost.png)

| Method | MAP calls | Tree calls | Total membership calls | Exact cases |
|---|---:|---:|---:|---:|
| BP AdamW | Not measured | Not measured | Not measured | Not measured |
| Original H3br four-MAP/depth-four policy, newly executed | 9,853 | 385 | 10,238 | 4,083/4,096 |
| Fixed H3bw deeper handoff policy, newly executed | 2,185 | 7,790 | 9,975 | 4,083/4,096 |

The deeper arm replaces 7,668 MAP calls with 7,405 additional tree calls. Both arms together pay 20,213 membership queries, including 6,746 independently reacquired same-case queries. No common MAP prefix, tree label, historical answer or source duplicate creates a free oracle call. The endpoint counts MAP/tree membership calls; the common eight-positive source budget is reported separately.

H3bw fixed this policy using the earlier examined H3br cohort and achieved a 2.143892% saving there. H3bx freezes the same policy before a new cohort, rather than reusing that comparator total. The earlier 10,355-call denominator and 10,147-call passing threshold play no role in the new primary. H3bu/H3bv remain incomplete at their original caps. This new finite-cohort confirmation does not retroactively complete them.

The supplied class retains all 1,541 ordinary-queen models with zero, one or two D4 leaper orbits from 1 <= a < b <= 11. The sampling prior chooses orbit count k uniformly from {0,1,2}, then a model uniformly among C(55,k), conditioned on its complete N12 positive population having M >= 8. The 1,522 eligible models receive exact integer prior weights 1485/27/1, totaling 4,436 tickets; all 19 ineligible rows and zero-frequency models remain in the reports. The query universe remains all 73,712 supplied N13 boards.

For case i in 0..4095, `default_rng(2026900000+i).integers(4436)` selects the truth via exact cumulative ticket intervals. An independent `default_rng(2027000000+i).choice(M,8,replace=False)` selects eight distinct positives. All 4,096 truth/source pairs are drawn and durably logged before any prospective membership query. The 8,192 seed values avoid both explicitly recorded historical seeds and an expanded conservative union of 507,476 prior truth/source seed values. The audit states its known-stream boundary; it does not certify undocumented RNG streams.

Each pair shares one source bank: 32,768 newly drawn positive appearances, 65,536 arm appearances and 12,412 globally distinct N12 boards. All 3 within-cohort duplicate-bank cases and all 6 cases matching historical eight-board banks are retained. Repeated truths, ties, successes and failures are neither redrawn nor excluded.

The deeper arm executes every case first. After m paid MAP calls, a nonsingleton set of at most sixteen candidates may use a posterior-optimal tree of requested depth 8-m. Larger sets continue unchanged MAP and remain in the experiment. Every hypothetical original-baseline branch is evaluated through its remaining 4-m MAP calls and terminal depth-four tree. Every candidate cost and exact flag is retained. An infeasible terminal baseline contributes zero additional tree calls and exact=False. Any false branch vetoes handoff; otherwise the new exact expected-cost comparison must pass, with equality accepted. Each arm retains its maximum of eight actual queries per case.

The baseline then starts every case again from its eight positives and an empty query history. It executes up to four original MAP calls, then the original full-enumeration posterior-optimal depth-four tree using the original firstlex feasibility compiler and lexicographic ties. Its compiler cache is separate from the deeper arm and counterfactual evaluator. Every baseline label is acquired again after selection. Query records are committed and fsynced before each separate oracle read and immediate hash-linked answer. Only true-source generation, oracle evaluation and result checking consume the actual truth identity; selection and compilation receive candidates, declared predictions, weights, depths and query history.

Both arms use the unchanged exact reciprocal source score `1 / (C(55,k) C(M,8))` over every compatible supplied model. With all baseline branches identifying, that complete baseline belongs to the deeper tree class, so the optimized tree must satisfy its prescribed-posterior expected-cost guard. Expected nonincrease under the matched prior and preservation of every exact flag are conditional correctness properties. They are separate from the measured finite-cohort 2% endpoint and do not establish a uniform population effect size.

| MAP calls before deeper handoff | Cases |
|---:|---:|
| 0 | 2,446 |
| 1 | 1,252 |
| 2 | 278 |
| 3 | 76 |
| 4 | 18 |
| No tree handoff | 26 |

The all-inline compiler preserves normalized depth `min(depth, subset_size-1)`, exact Huffman lower-bound ordering, and lexicographic ties. Every lower bound is recomputed, every merge is charged, complete inline traces are retained and the persistent bound registry remains empty. Full proofs precede compact routing caches. Canonical tree reuse preserves distinct requested-depth inputs and binds the complete original proof/result identities. These computation rules are unchanged from H3bw.

Before any new draws, the monitored run repeats 17,008 fixed inline/reference cases, 4,252 original-baseline depth-four comparisons, cost/admissibility and cache gates, four baseline-wrapper fixtures, source/prior checks and ledger durability tests. Runtime gates charge 409,517 states / 856,035 work steps. Independent prefreeze replay also rejects ten rehashed semantic mutations involving candidate sets, labels, boards, arm identities and inherited queries. No prospective outcome was used to choose the new seeds or gate fixtures.

| Monitored resource, both arms and all runtime gates | Used | Limit |
|---|---:|---:|
| Memo states | 1,023,451 | 2,000,000 |
| Charged work steps | 36,779,912 | 50,000,000 |
| Aggregate CPU seconds | 293.477040 | 600 |
| Wall seconds | 361.055631 | 900 |
| Actual membership calls | 20,213 | 65,536 |
| GPU seconds | 0 | CPU exact search |

The deeper compiler charges 565,888 states / 35,841,053 work steps, including 29,001,238 recomputed merges. It records 46 canonical reuse hits. Counterfactual evaluation charges 28,654 nodes / 13,394 MAP partitions; the separately executed original compiler charges 19,392 states / 69,430 work. Full process-lifetime monitor CPU, initial source hashing, selector descendants, preparation, serialization and both realized arms are included. Four bounded monitor probes verified initialization, descendant CPU and both resource guards before the run. No cap was extended. Post-run compression and verification are separately timed. This is a query-efficiency result, not a measured end-to-end speedup claim.

The complete independent scientific audit passes **9,534,965 main checks in 291.970541 seconds**, plus its embedded 1,241,564 inline and 666,061 original-baseline checks. It reconstructs every fresh prior ticket and sampled source index; replays all 20,213 separately paid calls; checks every case, exact flag, duplicate and cost; and independently verifies every original firstlex/weighted certificate, deeper certificate, counterfactual branch, input binding and resource counter. The verifier does not invoke any scientific tree compiler or policy evaluator.

A separate lossless export audit then matches all 81,907 immutable scientific files to their exported representation. The full archive retains 5,046 proofs totaling 30,494,351,089 raw bytes / 3,883,008,269 deterministic gzip bytes, plus every counterfactual document and the exact original ledger. Large proof payloads remain outside Git on Loki and in local artifacts. The compact package publishes all raw/compressed identities, source/protocol seals, complete cases and the full audit receipt. **Portable checkout verification reconstructs sources and replays paid labels, filters and MAP choices, but does not replay omitted optimality proofs.** Full proof replay requires the separately retained archive.

```bash
upstream/.venv/bin/python research/20261114/loki/h3bx/verify_public.py --root SNAPSHOT
upstream/.venv/bin/python research/20261114/loki/h3bx/verify_public.py --root SNAPSHOT --archive-dir FULL_EXPORTED_ARCHIVE
artifacts/plotting-venv/bin/python research/20261114/loki/h3bx/plot_public.py
```

Protocol SHA-256: `bb832478a3b226f89f204955e629ed438e830aeb86f01d17cf1b2612eb0be1fc`. The ledger has 75,563 records and raw SHA-256 `c4660652b671d663fc8d182b991284b87535ce9cce8f164b2f15232e1fa6a901`. The experiment concerns a supplied finite class, complete model predictions and a noiseless membership oracle. It establishes neither arbitrary-rule learning, positive-only learning, learned-optimizer superiority nor a uniform population saving. A prior-shift stress test is a separate prospective study.

[Frozen protocol](protocol.json), [prospective design](DESIGN.md), [summary](../../../../results/research/loki/h3bx/summary.json), [all paired costs](../../../../results/research/loki/h3bx/case-costs.csv), [full audit receipt](../../../../results/research/loki/h3bx/full-archive-audit.json), [external archive boundary](../../../../results/research/loki/h3bx/archive-boundary.json).
