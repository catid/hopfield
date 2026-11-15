# H1bp: exact mask balance gives no recovery advantage at 180 rows

The balanced-mask recovery target is **supported at 32/32 exact graphs**, but the separate advantage target is **falsified: uniform masks also recover 32/32, for a difference of zero instead of the required two**. All 64 arms select degree 20 as their unique curvature maximum. There are no selected fractional or unresolved outputs. This is a ceiling result for these banks, not evidence that the mask designs are equivalent.

| Procedure: 180 boards ×72 queried cells, common iid5 noise | Exact graphs | Degree 20 selected | Mean within-bank pair-exposure variance |
|---|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — |
| Independent uniform 72-cell masks | 32/32 | 32/32 | 30.114757 |
| Balanced affine-line complement masks | **32/32** | 32/32 | **0** |
| Uniform top-810 control | 0/32 | not imposed | same uniform masks |
| Balanced top-810 control | 0/32 | not imposed | same balanced masks |

Every paired bank succeeds under both designs. The supplied-degree-20 grid diagnostics also recover 32/32 for each. Top-810 controls miss 91 and 117 edge-bank pairs, respectively, with matching extras. These controls and paired outcomes are descriptive; the primary recovery threshold is 31/32 and the separate advantage threshold is two exact graphs.

The new balanced acquisition uses the 90 affine lines in `GF(9)²`, with `GF(9)=GF(3)[t]/(t²+1)`. Its 81 abstract points index cells. A row queries the 72-cell complement of one nine-point line. Each point belongs to ten lines and each distinct pair belongs to one line, so one complete 90-row cycle observes every cell 80 times and every pair 71 times. Two cycles give **160 observations per cell and exactly 142 coqueries per pair**. Each cycle independently randomizes both all 81 cell identities and all 90 line positions. These random permutations do not use Sudoku geometry or labels.

Uniform masks query the same 72 cells per row in distribution, independently across rows. Their expected pair exposure is also 142, with marginal variance `1349/45 ≈29.977778`. The observed average within-bank variance is 30.114757. Balanced masks reduce that variance to zero, while also changing higher-order mask dependencies. This is a joint acquisition-design intervention, not a single-cause test of exposure variance. The lack of recovery improvement does not mean balanced acquisition is never useful.

Each of the 32 banks uses the same 180 clean source rows and **one common latent iid5 noisy stream** for both mask designs. Cell corruption independently adds an offset from 1 through 8 modulo 9 with archived binary64 probability threshold 0.05. Both arms see identical values wherever their masks overlap. All identities and both complete mask streams in every bank are sealed before selected CSV labels are materialized. Masks, source IDs and noise use separate frozen random streams. There is no label feedback, donor acquisition, resampling or rescue. Noise parameters and mask geometry are not supplied to the graph estimator.

Fresh rows come from the 1,475,064 official IDs outside the **2,356,930-ID declared combined Ripper/H4ak ledger** through H1bo. H1bp uses **5,746 new unique IDs**, 5,760 source instances and 14 repeated instances across banks. There are no within-bank repeats or declared-exclusion overlaps. Independent auditing checks every selected answer against the original CSV and Sudoku constraints. Updated ledgers are **2,126,251 Ripper IDs** and **2,362,676 combined IDs**. Freshness is relative to that ledger, not globally untouched data. Future exclusions must include the selected IDs and `all_source_manifest.json`; puzzle-solver inputs are untouched.

| Acquisition and data-construction accounting | Cell-label instances |
|---|---:|
| Either standalone deployment: 32×180×72 | **414,720** |
| Literal observations shared across the two masks | 368,531 |
| New physical noisy research union | **460,909** |
| Separate clean-source oracle: 5,746 unique boards ×81 | **465,426** |
| Clean materialized instances, including repeats | 466,560 |

The two masks share the same latent noisy boards, so shared queried cells are charged once. Separate noisy instances of repeated source IDs remain charged. The clean-source oracle is separate from noisy deployment acquisition. The H1az–H1bp noisy-label sequence union becomes **10,079,398**, excluding clean-oracle access and older research outside that sequence. The common full-board noise has 23,397 wrong cells out of 466,560; queried errors are 20,877/414,720 for uniform masks and 20,778/414,720 for balanced masks. These are descriptive realized diagnostics; common full-board errors must not be added across arms.

Inference is unchanged: exact half-even Laplace rate costs, fresh degree-1-through-40 LP solves, an analytic degree-zero witness, and exact second-difference selection over degrees 1 through 39 with lower-degree ties. The common-degree prior and bounded grid remain assumptions; degree 20, edge count 810, Sudoku geometry and the noise parameter are not supplied. All **2,560 exact LP certificates, 7,680 altered-proof rejections, 64 zero witnesses and 64 complete-label substitution gates pass**. Intermediate fractional LP counts are 858 for uniform and 879 for balanced masks. They remain valid objective evidence; no primal is snapped. Certificates prove optimality, not uniqueness.

Each endpoint is assessed separately. Recovery bounds are `balanced successes` through `balanced successes + unresolved`. Advantage bounds are `balanced successes − uniform successes − uniform unresolved` through `balanced successes + balanced unresolved − uniform successes`. Selected fractional/nonexact outputs are resolved failures; any uncertified grid degree would make the arm unresolved. All banks and grid outcomes are retained.

Execution completed **68 commands in 34.451140 aggregate CPU-seconds and 11.109837 wall seconds**, below the frozen 900 CPU-second /180-second ceilings. Eight workers handle masks and bank fits/audits. At 1.14 seconds the live counter showed 33 commands complete and the CSV materializer at 101% CPU. This confirms the serial materialization dependency; the previously measured faster positional parser is retained. All four Ripper GPUs were at 0% utilization and 2 MiB allocation because this is CPU LP inference. Final progress records 68 completed commands and no active owned workers.

Synthetic preflight independently implements field multiplication by polynomial convolution/reduction, checks all 729 associativity/distributivity triples and verifies exact line incidence. Three randomized mask schedules replay identically under independent implementations. It also tests independent mask streams, noisy-value sharing, threshold boundaries, realized-noise counts, full-label substitutions, literal union accounting, six conservative endpoint cases and toy exclusions. The lead agent reviewed the producer, independent auditor, gate and bounded runner before freeze. Frozen science remains unchanged.

Portable public verification replays identities/exclusions, masks, incidence, costs, proofs/mutations, endpoints and accounting. Two registry paths resolve to byte-identical public copies. Original CSV/noise replay and complete-label gates require the raw archive. Local, Ripper and a public-only checkout verification pass; deterministic PNG/SVG/provenance rerenders are byte-identical and visually inspected.

[Frozen protocol](protocol-balanced-masks.json) · [Study](balanced_masks_study.py) · [Independent audit](audit_balanced_masks.py) · [Finite-field masks](balanced_mask_math.py) · [Independent incidence audit](audit_balanced_mask_math.py) · [Portable verifier](verify_balanced_masks_public.py) · [Results](../../../results/research/ripper/balanced_masks/summary.json) · [Figure](../../../assets/figures/balanced_masks.png)
