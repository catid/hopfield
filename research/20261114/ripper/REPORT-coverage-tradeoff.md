# H1bl: dense acquisition improves exact recovery at equal label cost

The dense procedure target is **supported**: querying 72 cells on 200 boards recovers **32/32 certified integral exact graphs**, exceeding the required 31. Querying 32 cells on 450 boards recovers **6/32** at the same 14,400 labels per bank. The separately specified dense-advantage target is also **supported**: the exact-recovery difference is **+26**, exceeding +2. Every bank is resolved; sparse acquisition returns four selected fractional solutions, which remain procedure failures.

| Procedure on the fresh paired cohort | Exact graphs | Degree 20 selected | Labels per bank |
|---|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — |
| Sparse: uniform 32 cells on 450 boards | 6/32 | 32/32 | 14,400 |
| **Dense: uniform 72 cells on the first 200 boards** | **32/32** | **32/32** | **14,400** |
| Sparse supplied-degree-20 grid diagnostic | 6/32 | supplied | same observations |
| Dense supplied-degree-20 grid diagnostic | 32/32 | supplied | same observations |
| Sparse top-810 control | 0/32 | not imposed | same observations |
| Dense top-810 control | 25/32 | not imposed | same observations |

Dense acquisition wins on 26 paired banks, with no sparse wins. Both succeed on banks 1, 3, 4, 11, 13 and 16. Sparse selected primals are fractional on banks 2, 6, 17 and 18; the other 22 sparse failures are integral but nonexact, totaling 159 missing and 159 extra edge-bank pairs. Those totals exclude the fractional outputs. The top-810 controls miss 502 edge-bank pairs under sparse acquisition and seven under dense acquisition, with matching extras. Degree 20 is the unique exact-curvature maximum on all 64 arms.

The protocol was frozen before new identities, masks or outcomes. It excludes **2,310,410 source IDs** from the declared combined Ripper/H4ak ledger through H1bk. Each of 32 banks draws 450 distinct identities from the remaining **1,521,584** official rows. The cohort has **14,328 new unique IDs**, 14,400 row instances and **72 repeated instances across banks**, with no within-bank repeats or declared-exclusion overlap. Freshness refers to that ledger, not globally untouched data or an external source distribution.

All identities are sealed first, followed by all **64 complete mask streams**, before selected CSV labels are materialized. Each row receives a fresh uniform order of all 81 cells. Sparse acquisition uses its first 32 cells on all 450 rows; dense acquisition uses its first 72 cells on the fixed first 200 rows. Sparse masks are therefore subsets of dense masks on shared rows, while both marginal recipes remain uniform. The 200-row prefix is fixed before labels, without row selection. There are no donors or label feedback.

Both arms share mask-independent whole-board corruption on their common rows. A board is replaced by its uniform random cell permutation when a fresh uniform draw is below 9/160; dense acquisition uses the literal first 200 clean/noise rows and never the 250-row suffix. Neither estimator receives that corruption probability. The preflight explicitly tests invariance to changes in dense-unused suffix labels and latent noise.

At equal label cost, dense acquisition provides **511,200 pair coqueries per bank**, versus **223,200** for sparse acquisition, while using fewer independent corrupted boards. The result tests this combined acquisition tradeoff. It does not establish that coquery count alone causes the improvement. H1bk's 32/32 uniform-32 recovery used 900 boards and twice the label budget on a different cohort; it is unpaired context only.

Recovery retains exact half-even Laplace rate costs on all 3,240 pairs. It solves a fresh LP at every degree 1 through 40 and uses an analytic zero witness at degree zero. Selection maximizes the exact second difference of doubled objectives over degrees 1 through 39, breaking ties toward the lower degree. The method still supplies a common-degree prior and bounded grid; it does not supply degree 20, the 810-edge count, geometry or donor information.

Independent source auditing reproduces every draw, checks every selected answer against the original CSV and Sudoku constraints, and checks identity-to-mask-to-label ordering. Independent replay verifies every mask, queried value, shared value, equality/coquery count, cost and proof. All **2,560 exact LP certificates**, **64 analytic zero witnesses**, **7,680 altered-proof rejection tests** and **64 complete-label substitution gates** pass. The grid includes 1,711 fractional solutions (865 sparse, 846 dense); these remain valid objective evidence. No primal is snapped or rounded. Certificates prove optimality, not uniqueness or the absence of integral co-optima. Any uncertified grid degree would make its entire arm unresolved, and separate conservative endpoint bounds retain all such cases.

| Acquisition accounting | Charged label instances |
|---|---:|
| Sparse complete alternative deployment | **460,800** |
| Dense complete alternative deployment | **460,800** |
| Shared queries on the first 200 rows | 204,800 |
| **Actual new paired research union** | **716,800** |
| New donor queries | **0** |

The physical union is `32 × (450 × 32 + 200 × 72 − 200 × 32) = 716800`. Repeated source IDs across banks carry separate latent instances and are charged separately. The H1az–H1bl sequence physical union is **7,376,953**, not a total for all historical Ripper research. The updated declared Ripper ledger has **2,088,313 unique IDs**; the combined Ripper/H4ak union has **2,324,738**. Future exclusions must include this study's selected IDs and `all_source_manifest.json`. No puzzle-solver input is used.

All **68 commands** completed in **33.656554 aggregate CPU-seconds and 11.190022 wall seconds**, within the frozen 900 CPU-second / 180-second ceilings. Eight workers handle mask generation and paired fit/audit jobs. The execution wrapper recorded process CPU and completion counters during execution. The external process/GPU sample arrived after completion: no owned worker remained, and all four Ripper GPUs showed 0% utilization and 2 MiB allocation. This is CPU-only structural inference; no GPU computation was required.

Before freezing, a bounded parser optimization replaced dictionary allocation with positional `csv.reader` in this new study's materializer. On the first 200,000 archived official rows, restricted to 1,449 previously examined H1bk IDs, three alternating trials produced byte-identical outputs: median 0.214419 seconds before and 0.148642 after (**1.44× on that warm-cache subset only**). This is not a general end-to-end speed guarantee. The independent source auditor was unchanged. The full synthetic gate passed again after the change, including variable-width counts, prefix coupling, complete-label substitutions, literal unions, endpoint bounds and source exclusions. The initial and refreshed receipts are preserved.

Public verification reproduces fresh-ID draws, exclusions, prefix-nested masks, literal query accounting, rate costs, every exact proof and mutation test, endpoints and paired outcomes. A publication adapter maps two frozen registry paths to byte-identical public files; frozen science remains unchanged. Original CSV/recipient-label replay and complete-label substitution require the raw archive. The portable package is verified locally, on Ripper, and from a checkout without an `artifacts` directory. The figure has a fixed SVG hash salt and omitted date metadata for deterministic rendering.

[Frozen protocol](protocol-coverage-tradeoff.json) · [Study](coverage_tradeoff_study.py) · [Independent audit](audit_coverage_tradeoff.py) · [Portable verifier](verify_coverage_tradeoff_public.py) · [Results](../../../results/research/ripper/coverage_tradeoff/summary.json) · [Figure](../../../assets/figures/coverage_tradeoff.png)
