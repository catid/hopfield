# H1bq: balanced masks meet recovery target but miss advantage target at 7.5% noise

The balanced-mask recovery target is **supported: 26/32 exact graphs exceeds the required 24**. The separate advantage target is **falsified: uniform masks recover 25/32, leaving a one-graph advantage instead of the required four**. Every arm selects degree 20 as its unique curvature maximum. No selected output is fractional or unresolved. Neither equivalence nor a general benefit of balanced masks follows from the one-graph difference.

| Procedure: 180 boards ×72 queried cells, common iid7.5% noise | Exact graphs | Integral nonexact failures | Missing /extra edge-bank pairs |
|---|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — |
| Independent uniform masks | 25/32 | 7 | 22 /22 |
| Balanced affine-line complement masks | **26/32** | 6 | 17 /17 |
| Uniform top-810 control | 0/32 | 32 | 241 /241 |
| Balanced top-810 control | 0/32 | 32 | 281 /281 |

The fixed-degree-20 diagnostics recover the same 25 and 26 graphs. Balanced masks win on banks 1, 21, 27 and 29, while uniform masks win on banks 8, 10 and 14. Twenty-two banks succeed under both, and banks 0, 16 and 20 fail under both. All failures and regressions remain in the result.

The fixed 7.5% law was chosen prospectively after H1bp's 180-row 5% ceiling result, before H1bq identities, masks or labels. There is no outcome-based noise tuning or extension. H1bp is unpaired historical context. A fresh common latent corrupted board is generated for each H1bq row by adding a uniformly sampled nonzero offset 1 through 8 modulo 9 when `U < 0.075`. The **implemented binary64 threshold is `0x1.3333333333333p-4`**. Under ideal rational `p=3/40`, true-edge equality probability is `1839/102400` and the affine signal scale is `85849/102400`; these are algebraic ideal-law constants, distinguished from the floating threshold.

Both arms use the same 180 clean rows and latent noise. Uniform masks independently query 72 cells per row. Balanced masks query complements of the 90 affine lines in `GF(9)²` twice, with independent cell permutations and line-order permutations in each cycle. `GF(9)=GF(3)[t]/(t²+1)`. Each line omits nine cells, each point belongs to ten lines and each pair belongs to one, yielding exactly **142 coqueries per pair and 160 observations per cell** over two cycles. The estimator receives no field or Sudoku geometry. Uniform masks have expected pair exposure 142 and marginal variance `1349/45`; their observed mean within-bank variance is 29.739660, versus exactly zero for balanced masks. This changes higher-order mask dependencies jointly with exposure variance and does not isolate a single mechanism.

All source IDs and both complete mask streams for every bank are sealed before selected CSV labels are materialized. Source, uniform-mask, balanced-mask and noise streams have separate frozen seeds. No label feedback, donors, filtering or resampling is allowed. The new cohort draws outside the **2,362,676-ID declared combined Ripper/H4ak ledger** through H1bp, from 1,469,318 eligible official rows. It uses **5,750 new unique IDs**, 5,760 source instances and ten cross-bank repeated instances, with no within-bank repeats or exclusion overlaps. Independent auditing verifies selected answers against the original CSV and Sudoku constraints. Updated ledgers contain **2,132,001 Ripper IDs** and **2,368,426 combined IDs**. Freshness is relative to this ledger; puzzle-solver inputs are untouched.

| Acquisition and data-construction accounting | Cell-label instances |
|---|---:|
| Each standalone deployment | **414,720** |
| Common latent observations shared across masks | 368,719 |
| New physical noisy research union | **460,721** |
| Separate clean-source oracle: 5,750 unique boards ×81 | **465,750** |
| Clean materialized instances, including repeats | 466,560 |

Shared `(bank,row,cell)` observations are charged once; repeated source IDs with distinct noisy instances remain separate. The clean-source oracle is separate from noisy acquisition. The H1az–H1bq noisy-label sequence union is **10,540,119**, excluding clean-oracle access and older studies outside that sequence. The common noise changes 35,136/466,560 full-board cells; queried errors are 31,148/414,720 under uniform masks and 31,289/414,720 under balanced masks. These realized diagnostics do not change endpoints. Shared full-board error counts must not be added across arms. Future exclusions must include these selected IDs and `all_source_manifest.json`.

Inference remains exact half-even Laplace rate costs, fresh degree-1-through-40 LP solves, an analytic zero witness and exact curvature selection over degrees 1 through 39 with lower-degree ties. A common degree and bounded grid are supplied priors; degree 20, edge count 810, geometry and noise parameters are not supplied. All **2,560 exact LP certificates, 7,680 altered-proof rejections, 64 zero witnesses and 64 complete-label substitution gates pass**. Intermediate fractional LP counts are 871 uniform and 885 balanced, retained as valid objective evidence without snapping. Certificates prove optimality, not uniqueness. Any uncertified degree would make the whole arm unresolved; selected fractional/nonexact outputs are resolved failures.

Endpoints are separate and conservative. Recovery bounds are `S_balanced` through `S_balanced+U_balanced`, with threshold 24. Advantage bounds are `S_balanced−S_uniform−U_uniform` through `S_balanced+U_balanced−S_uniform`, with threshold four. With no unresolved arms the exact bounds are [26,26] and [1,1].

Execution completed **68 commands in 34.559714 aggregate CPU-seconds and 11.134154 wall seconds**, below 900 CPU-second /180-second limits. Eight workers overlap eligible fits and audits. A live sample at 1.14 seconds showed 33 completed commands and CSV materialization at 101% CPU. The verified faster positional parser is retained; serial materialization and source auditing remain dependencies. Four Ripper GPUs each showed 0% utilization and 2 MiB allocation because these are CPU LPs. Final progress has no active owned workers.

Synthetic preflight passes independent scalar `Fraction` noise replay, threshold boundaries, exact ideal-law probability enumeration, the full finite-field incidence audit, independent mask streams, shared noisy values, complete-label substitutions, realized-noise counts, literal union accounting, six endpoint-bound cases and toy exclusions. The lead agent reviewed final source and gate before freeze. Frozen science remains unchanged.

Portable public verification passes locally, on Ripper and in a public-only checkout. It reconstructs identities, masks, costs, certificates/mutations, endpoints and query accounting, mapping two frozen registry paths to byte-identical public files. Original CSV/noise replay and full-label gates require the raw archive. PNG/SVG/provenance rerenders are byte-identical and the figure is visually inspected.

[Frozen protocol](protocol-higher-noise.json) · [Study](higher_noise_study.py) · [Independent audit](audit_higher_noise.py) · [Portable verifier](verify_higher_noise_public.py) · [Results](../../../results/research/ripper/higher_noise/summary.json) · [Figure](../../../assets/figures/higher_noise.png)
