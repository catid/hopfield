# H1bm: dense recovery holds at matched pair coverage; advantage target fails

The dense recovery target is **supported**: querying 72 cells on 124 boards recovers **31/32 certified integral exact graphs**, meeting the required 31. Sparse acquisition, querying 32 cells on 639 boards, recovers **30/32**. Both arms provide exactly **316,944 pair coqueries per bank**. The separately specified dense-advantage target is **falsified**: its observed advantage is **+1**, below the required +2. Every bank is resolved, with no selected fractional solution. This is not an equivalence result or proof that total pair coverage explains recovery.

| Procedure on the fresh paired cohort | Exact graphs | Degree 20 selected | Labels per bank |
|---|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — |
| Sparse: uniform 32 cells on 639 boards | 30/32 | 32/32 | 20,448 |
| **Dense: uniform 72 cells on the first 124 boards** | **31/32** | **32/32** | **8,928** |
| Sparse supplied-degree-20 grid diagnostic | 30/32 | supplied | same observations |
| Dense supplied-degree-20 grid diagnostic | 31/32 | supplied | same observations |
| Sparse top-810 control | 0/32 | not imposed | same observations |
| Dense top-810 control | 2/32 | not imposed | same observations |

Dense acquisition wins on banks 6 and 21; sparse acquisition wins on bank 8. Both succeed on the other 29 banks. Each of the three failed arm/bank combinations misses two true edges and adds two false edges. The top-810 controls miss 137 edge-bank pairs under sparse acquisition and 105 under dense acquisition, with matching extras. Degree 20 is the unique exact-curvature maximum on all 64 arms.

H1bl motivated this prospective test: at equal label cost, dense acquisition recovered 32/32 versus sparse 6/32 while receiving more than twice as many pair coqueries. H1bm matches total pair volume instead: `639 × C(32,2) = 124 × C(72,2) = 316944`. These are different fresh cohorts and budgets. The earlier effect size is unpaired context, not an estimate of how changing pair volume affects this cohort.

The protocol was frozen before new identities, masks or outcomes. It excludes **2,324,738 source IDs** from the declared combined Ripper/H4ak ledger through H1bl. Each of 32 banks draws 639 distinct identities from the remaining **1,507,256** official rows. The cohort has **20,313 new unique IDs**, 20,448 row instances and **135 repeated instances across banks**, with no within-bank repeats or declared-exclusion overlap. Freshness refers to that ledger, not globally untouched data or an external source distribution.

All identities are sealed first, then all **64 complete mask streams**, before selected CSV labels are materialized. Each row receives a fresh uniform order of all 81 cells. Sparse acquisition uses its first 32 cells on all 639 rows; dense acquisition uses its first 72 cells on the fixed first 124 rows. Sparse masks are subsets of dense masks on shared rows. The prefix is fixed before labels, without row selection, donors or label feedback. Both arms share mask-independent whole-board corruption on those rows: a fresh uniform draw below 9/160 triggers a uniform cell permutation. Dense acquisition uses the literal prefix of clean boards and latent noise; it never uses the 515-row suffix. Neither estimator receives the corruption probability.

Matching total pair counts does not match the exposure vector or observation dependence. Before the official draws, mask combinatorics gave the following rationale: for one fixed pair, `J ~ Binomial(n, m(m−1)/(81×80))`. Both arms have expected exposure **4402/45 = 97.8222**, but mask-induced variance is **82.84697** for sparse acquisition and **20.65136** for dense acquisition. These are per-pair mask marginals; pair observations within a board are dependent.

The preregistered descriptive audit measures mean within-bank exposure variance across the 3,240 pairs: **82.74968 sparse versus 21.64042 dense**. Across all banks, individual-pair exposures range from **59 to 139** sparse and **78 to 115** dense. Lower exposure dispersion, fewer independent corrupted boards and different within-board dependence accompany dense acquisition. Neither the finite outcome nor these diagnostics isolate a unique causal explanation.

Recovery retains exact half-even Laplace rate costs on all 3,240 pairs. Fresh LPs solve every degree 1 through 40; degree zero uses an analytic zero witness. Selection maximizes the exact second difference of doubled objectives over degrees 1 through 39 with lower-degree ties. A common-degree prior and bounded grid remain supplied. Degree 20, the 810-edge count, geometry and donor information are not supplied to the selected procedure.

Independent source auditing reproduces every draw and checks all selected answers against the original CSV and Sudoku constraints. Independent replay verifies every mask, queried value, shared value, equality/coquery count, cost and proof. All **2,560 exact LP certificates**, **64 analytic zero witnesses**, **7,680 altered-proof rejection tests** and **64 complete-label substitution gates** pass. The grid has 1,710 fractional solutions (850 sparse, 860 dense); they provide valid objective evidence, while all selected solutions are integral. No primal is snapped or rounded. Certificates establish optimality, not uniqueness. Any uncertified grid degree would make its entire arm unresolved, and separate conservative endpoint bounds retain all such cases.

| Acquisition accounting | Charged label instances |
|---|---:|
| Sparse complete alternative deployment | **654,336** |
| Dense complete alternative deployment | **285,696** |
| Shared queries on the first 124 rows | 126,976 |
| **Actual new paired research union** | **813,056** |
| New donor queries | **0** |

The dense deployment uses **368,640 fewer labels**, a **56.34% reduction** relative to sparse acquisition, while meeting its finite 31/32 target. The research union is `32 × (639 × 32 + 124 × 72 − 124 × 32) = 813056`. Cross-bank repeats carry separate latent instances and are charged separately. The H1az–H1bm sequence physical union is **8,190,009**, not a total for all historical Ripper research. The updated declared Ripper ledger has **2,108,626 unique IDs**; the combined Ripper/H4ak union has **2,345,051**. Future exclusions must include this study's selected IDs and `all_source_manifest.json`. No puzzle-solver input is used.

All **68 commands** completed in **34.376556 aggregate CPU-seconds and 11.352452 wall seconds**, within the frozen 900 CPU-second / 180-second ceilings. Eight workers handle masks and paired fit/audit jobs. At 1.04 seconds, live counters showed identity selection plus all 32 paired-mask commands complete, with the CSV materializer at 101% CPU. Serial CSV materialization was the observed bottleneck; this study retained H1bl's verified positional parser optimization. All four Ripper GPUs were at 0% utilization and 2 MiB allocation because this structural inference is CPU-only. The external sample is preserved in `live-sample.json`; final progress records no active owned process.

Synthetic preflight tested exact matched pair counts, nested masks, prefix values/noise, invariance to dense-unused suffix changes, complete-label substitutions, literal unions, four conservative endpoint cases and excluded-source toy draws. The lead agent reviewed the new source before freeze. Public verification replays identities, exclusions, masks, query accounting, exposure diagnostics, rate costs, all proofs/mutations and both endpoints. Two frozen registry paths map to byte-identical public files through an adapter; scientific files remain unchanged. Original CSV/recipient-label replay and full-label gates require the raw archive. Local, Ripper and public-only-checkout verification pass; deterministic PNG/SVG/provenance rerender is checked byte for byte.

[Frozen protocol](protocol-matched-coquery.json) · [Study](matched_coquery_study.py) · [Independent audit](audit_matched_coquery.py) · [Portable verifier](verify_matched_coquery_public.py) · [Results](../../../results/research/ripper/matched_coquery/summary.json) · [Figure](../../../assets/figures/matched_coquery.png)
