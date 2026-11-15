# H1bn: both iid corruption targets fail

Both preregistered targets are **falsified**. With independent errors on 5% of cells, dense acquisition recovers **15/32 certified integral exact graphs**, below the required 31. Reducing the independent error rate to match the bursty law's true-edge equality probability recovers **30/32**, also below 31. The descriptive bursty baseline recovers **30/32**. Every bank is resolved, and all **96 arms select degree 20**. Equal aggregate recovery for the bursty and pair-matched laws is not an equivalence result.

| Procedure: 72 queried cells on 124 boards | Exact graphs | Degree 20 selected | Selected fractional outputs |
|---|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — |
| Bursty whole-board permutation, descriptive baseline | 30/32 | 32/32 | 0 |
| Independent 5% cell errors | **15/32** | 32/32 | 3 |
| Independent errors matching true-edge equality rate | **30/32** | 32/32 | 0 |
| Bursty top-810 control | 3/32 | not imposed | — |
| Independent 5% top-810 control | 0/32 | not imposed | — |
| Pair-matched independent top-810 control | 0/32 | not imposed | — |

The supplied-degree-20 grid diagnostics recover the same 30, 15 and 30 graphs. The bursty failures are banks 11 and 30, each with two missing and two extra edges. Independent 5% noise yields 14 integral nonexact failures totaling 73 missing and 73 extra edge-bank pairs, plus fractional outputs on banks 7, 15 and 22. Those error totals exclude fractional outputs. Pair-matched independent noise fails on banks 14 and 20 with six missing and six extra edge-bank pairs in total. Top-810 controls miss 113, 394 and 140 edge-bank pairs respectively, with matching extras.

Paired contrasts are descriptive. Independent 5% noise wins on one bank and loses on 16 against the bursty baseline; 14 banks succeed under both and one fails under both. Pair-matched independent noise wins on two banks and loses on two, with 28 shared successes. Neither comparison had a contrast endpoint. The two 31/32 targets remain failed regardless of these patterns.

The three corruption laws were frozen before fresh source identities or masks. Bursty noise uniformly permutes all 81 positions of a board with probability 9/160. Because each clean board has nine copies of every digit, this gives **5% marginal wrong-cell probability** and **0.005625 true-edge equality probability**. Independent noise adds a uniformly sampled nonzero offset from 1 through 8 modulo 9 to each selected cell. With error probability `p`, a true edge becomes equal with probability `p/4 − 9p²/64`.

| Corruption law | Marginal wrong-cell probability | True-edge equality probability |
|---|---:|---:|
| Bursty permutation mixture | 0.05 | 0.005625 |
| Independent errors at p = 0.05 | 0.05 | 0.0121484375 |
| Independent errors at p = 0.022792210226441033 | 0.022792210226441033 | 0.005625, within 3.65×10⁻¹⁹ |

The pair-matched probability is the smaller root of `9p² − 16p + 0.36 = 0`, frozen as binary64 `0x1.756d7561abf76p-6`. The independent audit enumerates the nine possible shared observed digits with exact rational probabilities; it also enumerates all 6,480 distinct ordered source-position pairs for bursty noise. Matching true-edge equality does not match pair covariance, nonpeer equality rates or whole-board digit counts. This experiment tests these declared corruption distributions, without isolating a single causal mechanism or establishing universal robustness.

All arms share the same clean boards and masks but use **independently seeded corruption streams**. A single fresh uniform 72-cell mask is drawn per row and reused across the three laws. All source IDs and all **32 common mask streams** are sealed before selected CSV labels are materialized. Noise is independent of the masks; no label feedback, donor policy or resampling of difficult corruptions is allowed. Corruption parameters are used by the data generator and auditor, and are not supplied to the graph estimator.

The new cohort draws 124 distinct rows per bank outside **2,345,051 declared combined Ripper/H4ak source IDs** through H1bm. It uses **3,963 new unique IDs**, 3,968 row instances and five repeated instances across banks from the remaining **1,486,943** official rows. There are no within-bank repeats or declared-exclusion overlaps. This is freshness relative to that ledger, not globally untouched data or an external distribution. Independent source auditing checks every selected answer against the original CSV and Sudoku constraints.

Realized full-board wrong-cell counts are **15,895 bursty, 16,192 independent 5%, and 7,440 pair-matched**, each out of 321,408 materialized cell instances. Queried wrong-cell counts are 14,122, 14,327 and 6,617 out of 285,696 per arm. True-edge equality counts among queried pairs are 13,963, 30,801 and 14,503. These are descriptive realized counts; difficult draws remain in the experiment.

Inference retains the common-degree prior, bounded degree grid 0 through 40 and exact half-even Laplace rate costs. Every degree 1 through 40 receives a fresh LP solve; degree zero uses an analytic zero witness. Selection maximizes the exact second difference of doubled objectives over degrees 1 through 39 with lower-degree ties. Degree 20, the 810-edge count, geometry and corruption parameters are not supplied. Degree 20 is the unique curvature maximum in every arm.

All **3,840 exact LP certificates**, **96 analytic zero witnesses**, **11,520 altered-proof rejection tests** and **96 complete-label substitution gates** pass. The grid includes 2,645 fractional solutions (882 bursty, 893 independent 5%, 870 pair-matched), which remain valid objective evidence. No primal is snapped or rounded. A selected fractional solution is a graph-procedure failure; any uncertified degree would make the entire arm unresolved. Each target separately uses lower `exact` and upper `exact + unresolved` bounds. Certificates prove optimality, not uniqueness or the absence of integral co-optima.

| Acquisition and data-construction accounting | Cell-label instances |
|---|---:|
| Each complete alternative deployment | **285,696** |
| All three noisy acquisition arms | **857,088** |
| Noisy labels reused across corruption laws | **0** |
| Separate clean-source oracle: 3,963 unique boards × 81 cells | **321,003** |
| Clean materialized instances, including cross-bank repeats | 321,408 |

Noisy acquisitions belong to different latent corrupted instances and are charged separately even when their realized values agree. The clean-source oracle supports simulation/data construction; it is reported separately from algorithm deployment labels. Repeated clean source IDs reuse the same clean answer, while separate latent noisy instances remain charged. The noisy-label sequence union for H1az–H1bn becomes **9,047,097**, not a total for all historical Ripper research and not a clean-oracle-inclusive total. Updated source ledgers contain **2,112,589 Ripper IDs** and **2,349,014 combined IDs**. Future exclusions must include these selected IDs and `all_source_manifest.json`. No puzzle-solver input is used.

Execution completed all **68 commands** in **43.731149 aggregate CPU-seconds and 12.357799 wall seconds**, within the frozen 900 CPU-second / 180-second ceilings. Eight workers process masks and per-bank fits/audits. A live external sample at 1.04 seconds showed 33 commands complete and CSV materialization at 100% CPU, confirming the observed serial data-loading bottleneck. The existing verified positional CSV parser was retained. All four Ripper GPUs showed 0% utilization and 2 MiB allocation; the study is CPU-only. Final progress has no active owned worker.

Synthetic preflight checks exact pair probabilities, the frozen probability hex value, threshold boundaries, nonzero offsets, independent noise replay, mask/latent invariance, complete-label substitutions, distinct-law query accounting, all 96 arm indices, source exclusions and conservative endpoint bounds. Its first implementation hit an integer overflow because `Fraction` retained a NumPy integer numerator; converting toy counts to Python integers fixed the gate before freeze. The failed receipt is preserved as implementation history, not a scientific outcome. The lead agent reviewed the complete new source and passed gate before freeze.

Public verification reproduces source draws/exclusions, common masks, costs, proofs/mutations, exact endpoints and paired outcomes. Two frozen registry paths resolve to byte-identical public copies; frozen science remains unchanged. Original CSV, corruption replay, realized noise diagnostics and full-label gates require the raw audit archive. The portable package is verified locally, on Ripper and in a public-only checkout without `artifacts/`. PNG/SVG/provenance are byte-identical on rerender and visually inspected.

[Frozen protocol](protocol-corruption-law.json) · [Study](corruption_law_study.py) · [Independent audit](audit_corruption_law.py) · [Portable verifier](verify_corruption_law_public.py) · [Results](../../../results/research/ripper/corruption_law/summary.json) · [Figure](../../../assets/figures/corruption_law.png)
