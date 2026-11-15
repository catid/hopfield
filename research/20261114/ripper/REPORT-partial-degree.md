# H1bi: partial-observation curvature infers degree 20 and retains recovery

Both separate prespecified aligned targets are **supported**. The LP-curvature selector chooses degree 20 on **32/32 banks**, exceeding the required 31, and returns **28/32 certified integral exact graphs**, exceeding the required 27. All 64 aligned and descriptive permuted banks have a unique curvature maximum at 20. The procedure supplies a common-degree grid, but does not supply degree 20 or the 810-edge count.

| Procedure on the examined H1bh banks | Degree 20 selected | Certified exact | Integral nonexact | Fractional | Unresolved |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — |
| Aligned fixed feedback, inferred degree | **32/32** | **28/32** | 4 | 0 | 0 |
| Permuted fixed feedback, inferred degree; descriptive | **32/32** | **25/32** | 5 | 2 | 0 |
| Archived aligned, supplied degree 20 | supplied | 28/32 | 4 | 0 | 0 |
| Archived permuted, supplied degree 20 | supplied | 25/32 | 5 | 2 | 0 |
| Archived aligned top-810 control | not imposed | 0/32 | — | — | — |
| Archived permuted top-810 control | not imposed | 0/32 | — | — | — |

This is an adaptive follow-up on the **examined H1bh observations**, not another fresh-recipient confirmation. The two aligned endpoints were frozen after seeing H1bh's 28/32 recovery. The exact-graph target allows at most one fewer aggregate exact recovery; it does not bound paired losses. The permuted arm has no new inferential endpoint. The result removes the supplied numerical degree on these inputs, without improving graph accuracy or reversing H1bh's failed 29/32 and +4 targets.

The input costs remain unchanged: for every one of 3,240 pairs, use exact half-even rounding of `1000000 * (equality_count + 1) / (coquery_count + 2)`. For each bank and each degree `d=1,...,40`, a fresh deterministic HiGHS instance minimizes the linear cost with each node's degree equal to `d` and edge weights in `[0,1]`. Degree zero uses an analytic all-zero primal/dual witness. With certified doubled objectives `V[d]=2F(d)`, select the largest `V[d+1] - 2V[d] + V[d-1]` over degrees 1 through 39; ties choose the smaller degree. The degree-20 fits are new solves, not copied ancestral solutions.

The common-degree assumption and bounded grid remain supplied priors. No cell geometry or preferred degree enters selection. Any additive cell cost `b[i]+b[j]` shifts the objective by `d * sum(b)` and leaves its exact curvature unchanged; this identity does not establish universal degree identification. The inherited acquisition still uses historical donor labels and shared indexed topology.

All **2,560 LPs** have valid exact scale-two primal/dual certificates, and all **64 degree-zero witnesses** pass. Independent checking rejects **7,680 altered certificates**. Every grid degree must certify before a bank can be resolved; none is unresolved. There are **1,731 fractional grid solutions**, including all 1,280 fits at odd degrees on 81 nodes. Those intermediate objectives remain valid evidence. Only a fractional selected solution is a graph failure; the two selected fractional permuted solutions are retained without rounding. Certificates prove optimality, not uniqueness or absence of integral co-optima.

All 64 selected primal arrays exactly match their archived supplied-degree-20 counterparts. Aligned failures remain banks 2, 3, 20 and 23, each with two missing and two extra edges. The permuted arm retains five integral failures with 14 missing and 14 extra edge-bank pairs, plus fractional banks 10 and 12. The unique maximum at degree 20 exceeds the runner-up by 864,915 to 2,310,454 in doubled-objective curvature units. These margins are diagnostics, not a calibrated confidence bound.

No labels or source IDs were acquired. Each alternative deployment still requires **960,000 historical labels**: 720,000 recipient queries and 240,000 donor warmup queries. Actual incremental acquisition is zero. The declared Ripper source ledger stays at **2,045,471 unique IDs**, and the H1az–H1bh sequence union stays at **5,148,181 physical labels**; the latter is not all historical Ripper research. No puzzle-solver input is used.

Execution completed all **65 commands** using **24.277323 aggregate CPU-seconds and 3.164344 wall seconds**, within the frozen 900 CPU-second / 180-second budget. Eight direct child processes each fit and independently audit a bank, and the queue refills immediately. A live sample showed 16 completed banks, eight active workers, and sampled workers at 100–102% CPU. This work is CPU-only; the four Ripper GPUs were idle at 0% utilization and 2 MiB each.

Before freezing, an exact audit-parity gate on eight archived H4aj banks checked 320 proofs and rejected 960 mutations in each mode. Serial replay took 0.822026 seconds; eight-process replay took 0.118258 seconds, a **6.95× improvement on that archived subset**, not a general speed guarantee. The outputs matched each other and the ancestor's summary exactly. Synthetic four- and five-node cases checked curvature selection, lower-degree ties, selected fractionality, additive-cell invariance and 42 mutated-proof rejections. A subsequent text-only correction clarified the aggregate endpoint wording; the initial receipt is preserved and the refreshed gate records its hash.

Public verification checks package hashes, reconstructs the unchanged rates from exported counts, replays every exact proof and mutation test, selects degrees independently, checks graph outcomes and archived controls, verifies the 64 identical selected primals and unique maxima, and validates resource receipts. Original-label and acquisition provenance remain covered by the frozen H1bh raw audit. Both local and Ripper public proof replays passed.

[Frozen protocol](protocol-partial-degree.json) · [Study](partial_degree_study.py) · [Independent audit](audit_partial_degree.py) · [Public verifier](verify_partial_degree_public.py) · [Results](../../../results/research/ripper/partial_degree/summary.json) · [Figure](../../../assets/figures/partial-degree.png)
