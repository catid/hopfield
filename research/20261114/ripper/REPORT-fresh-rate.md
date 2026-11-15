# H1az: N900 recovery confirms on sources disjoint from the Ripper ledger

The prospective hypothesis is **supported**: all **32/32 graphs are recovered exactly** at N=900, each with an independently verified exact primal-dual certificate. There are no certified nonexact, fractional, or unresolved outcomes. The required threshold was at least 29 exact graphs.

| Procedure | Exact graphs | Missing true edge-bank pairs | Extras | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Original acquisition support | 0/32 | 22 | 242 | 0 |
| Same-cost top-810 | 2/32 | 66 | 66 | 0 |
| Smoothed-rate degree-20 LP | **32/32** | **0** | **0** | **0** |

This confirms the N=900 result from the adaptive H1ay prefix study using **new source identities disjoint from the archived Ripper completion-source ledger**, new calibration observations, new predictable mask/tie streams, and new position-permutation corruptions. It does not claim that these identities were globally untouched by every other repository activity. The exact official training CSV, ledger, eligible pool rule, seeds, code, primary, and budgets were frozen before selecting any new identity. The selected identities were separately sealed before their labels were read. No held-out puzzle input or solver row was used.

The ledger consists of 19 archived manifests and 1,190 completion files covering **1,928,477 distinct source IDs**. The official training pool has 3,831,994 distinct complete boards, leaving 1,903,517 eligible IDs. A seeded permutation divides eligible IDs into disjoint calibration and training pools. Each of 32 banks samples 1,000 calibration and 900 training rows without replacement within the bank. Cross-bank repeats are permitted and measured.

| Source accounting | Calibration | Training | Combined |
|---|---:|---:|---:|
| Row appearances | 32,000 | 28,800 | 60,800 |
| Distinct newly selected IDs | 31,452 | 28,367 | **59,819** |
| Repeated appearances across banks | 548 | 433 | 981 |
| Overlap with prior declared ledger | 0 | 0 | 0 |

There is no within-bank repetition or calibration/training overlap. The updated Ripper ledger union is **1,988,296 IDs**. Future exclusions must include both roles, using this study's `all_source_manifest.json`. The independent source audit reconstructs the prior union, replays every seeded selection, checks disjointness, reads every selected answer from the CSV independently, checks all Sudoku constraints, and verifies unique complete answer strings. The public export contains the sealed excluded and selected ID arrays so sampling and declared-ledger disjointness are independently checkable; rebuilding historical exposure and verifying actual CSV labels require the raw archive.

The acquisition and recovery recipes are unchanged. Each training board has 25 queried cells, with the mask fixed before the current corruption. With probability 9/160 the complete board is uniformly permuted over its positions; otherwise it stays clean. This preserves digit histograms and makes repeated calibration labels agree, so the inherited scalar calibration remains misspecified. Its original first-crossing exclusions influence future query allocation; the final recovery LP nevertheless considers all 3,240 pairs.

Costs are `roundEven(1e6*(equalities+1)/(coqueries+2))`, computed with exact integer arithmetic and independently checked using `Fraction`. Every cell is constrained to degree 20. The top-810 control sorts the same costs with lexicographic pair-index ties. Degree 20 and its implied 810-edge cardinality are explicit structural priors; row, column, and box identities are not optimizer inputs. This does not measure a learned substitute for those priors.

All 28,800 masks, current-noise draws, observed labels, count updates, likelihood trajectories, first crossings, calibration endpoints, and retained acquisition supports are independently replayed. The recovery auditor independently recounts observed pairs and checks all 32 exact integer optimality proofs. The public verifier rejects 96 altered certificates. No primal is snapped or rounded. Certificates establish optimality, not uniqueness. A fractional returned optimum would have remained a procedure failure, without excluding an integral co-optimum. No extra prefix or replacement bank was allowed.

The experiment consumes **784,000 new queries**: 720,000 training and 64,000 calibration, or 24,500 per bank. The recovery costs require no supplied noise probabilities, but acquisition still uses calibration; this is not end-to-end calibration-free acquisition. This fresh study's spending is distinct from the nested historical budget alternatives in H1ax/H1ay.

Execution used **58.496781 aggregate CPU-seconds and 29.407572 wall seconds**, including identity selection/sealing, source materialization, independent CSV audit, all acquisitions and independent replays, 32 LPs/proofs, and summary. This is within frozen 900 CPU-second / 180-second ceilings. Eight one-thread workers process independent banks, with thread limits set before numerical imports; no GPU or puzzle solver work was performed. Synthetic preflight, prior-ledger registration, and publication packaging are separate.

The new acquisition auditor materializes its calibration NPZ arrays once, removing repeated decompression inside the replay loop while preserving the inherited calculation. Prefreeze lineage checks verify that the collector and auditor match their archived recipes after the declared N900 and eager-loading changes. Median acquisition time is 0.630 seconds per bank; the 32 LP calls together take 0.037154 solver CPU-seconds. Source preparation and independent validation dominate this small inference workload. These timings are not a matched-compute comparison with learned baselines.

The result supports a finite, prospective ledger-disjoint confirmation of this specific noise law, acquisition strategy, and structural recovery prior. It is not a population reliability bound or a guarantee under arbitrary correlated noise.

[Frozen protocol](protocol-fresh-rate.json) · [Independent source/recovery audit](audit_fresh_rate.py) · [Acquisition replay](audit_fresh_rate_acquisition.py) · [Public verifier](verify_fresh_rate_public.py) · [Results](../../../results/research/ripper/fresh_rate.json) · [Figure](../../../assets/figures/fresh-rate.png)
