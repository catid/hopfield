# H1bb: balancing exposure alone does not reach the N900 target

The frozen hypothesis is **falsified**: label-blind exposure balancing produces **25/32 certified integral exact graphs**, below the required 29. Seven returned optima are certified integral nonexact; there are no fractional or unresolved new runs. All-pair exposure variance falls by **98.505% relative to uniform masks**, so making coverage nearly uniform does not suffice for this fixed recovery procedure and budget.

| Procedure | Certified integral exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Archived adaptive masks + degree-20 LP | 32/32 | 0 | 0 | 0 |
| Archived uniform masks + degree-20 LP | 23/32 | 8 | 1 | 0 |
| Exposure-balanced masks + degree-20 LP | **25/32** | **7** | **0** | **0** |

The balanced same-cost top-810 control recovers 0/32 graphs, with 250 missing true edge-bank pairs and 250 extras. The balanced degree-20 LP has 20 missing and 20 extra edge-bank pairs across its 32 integral outputs. Six banks that failed with uniform masks become exact, while four uniform successes become nonexact. The net increase from 23 to 25 is descriptive; it is not a separate statistical superiority claim.

This is an adaptive follow-up after H1ba's uniform-mask failure, using the same H1az source rows, archived mask-independent corruption latents, and H1ba tie permutations. There are no fresh source identities or latent draws. The new rule and primary were frozen before generating any new-rule masks or counts. It uses the existing greedy mask rule with weights `(8/9)^joint_count`, but keeps every exclusion flag permanently false and updates only coquery exposures. No observed equality, calibration estimate, likelihood-ratio state, or geometry enters the mask generator. **All 900 masks are generated and saved before source labels or noise arrays are loaded.** The rule is not changed after outcomes.

Recovery remains the all-3,240-pair LP with exact half-even rounded Laplace equality-rate costs and degree exactly 20 at every cell. Degree 20 and its implied cardinality are supplied structural priors. The balancing rule is one particular fixed design for the known nine-symbol alphabet, not a guarantee for arbitrary acquisition rules or noise laws. The adaptive comparator still uses calibration and label-dependent exclusion feedback. The result does not show that calibration, or that particular feedback mechanism, is necessary for all successful procedures.

| Acquisition | Global observed pair-exposure range | Mean within-bank all-pair exposure variance |
|---|---:|---:|
| Archived adaptive | 28–155 | 595.377643 |
| Archived uniform | 49–124 | 74.258314 |
| Label-blind balanced | **79–89** | **1.109877** |

Each bank has exactly 270,000 pair coqueries, averaging 83⅓ per pair. Diagnostics use the exact integer dispersion `3240 * sum(m²) - sum(m)²`; variance divides this by `3240²`. Every bank's minimum balanced exposure lies between 79 and 81, and its maximum between 87 and 89. These after-fit diagnostics establish that exposure balancing achieved its intended coverage effect. They do not alone identify the cause of individual recovery failures or prove a mediation mechanism. The adaptive arm's larger all-pair variance is compatible with allocating observations selectively.

Independent replay checks all 28,800 balanced masks using a separate greedy implementation and outer-product count updates, reconstructs the same archived noise and selected labels, independently recounts costs, and verifies every reused answer. All **32 new and 64 archived exact optimality certificates** pass, including the archived uniform fractional optimum. No archived LP is refitted and no primal is rounded. A fractional returned optimum does not rule out an integral co-optimum; certificates establish optimality, not uniqueness. The public verifier replays masks, all three exposure diagnostics, cumulative overlap accounting and exact proofs, rejecting 288 altered certificates. Raw equality-label and original acquisition provenance remain separate archive checks.

| Query accounting | Count |
|---|---:|
| Alternative balanced deployment: training | 720,000 |
| Alternative balanced deployment: calibration | 0 |
| Alternative uniform deployment: total | 720,000 |
| Alternative adaptive deployment: total including calibration | 784,000 |
| Previously acquired union, adaptive + uniform + old calibration | 1,281,927 |
| Balanced answers reused from either previous mask set | 382,775 |
| **Newly exposed balanced answers** | **337,225** |
| **Cumulative physical union after this study** | **1,619,152** |

The new-query count excludes answers observed by either earlier acquisition, not just the adaptive arm. Reused plus newly exposed answers equal 720,000. Alternative deployment budgets are separate scenarios and are not summed as research spending.

Execution used **11.852775 aggregate CPU-seconds and 2.366740 wall seconds**, within the frozen 120 CPU-second / 60-second ceilings. Eight one-thread workers cover mask construction, acquisition, 32 LPs, independent replay, new/archived proofs, diagnostics, accounting and summary. Synthetic preflight and packaging are separate. The final live process receipt records all 34 commands complete and no active owned PID. No new source ID, latent draw, puzzle-solver input, or GPU work is used; the Ripper source ledger remains **1,988,296 IDs**.

[Frozen protocol](protocol-balanced-mask.json) · [Mask generator](balanced_mask_features.py) · [Independent audit](audit_balanced_mask.py) · [Public verifier](verify_balanced_mask_public.py) · [Results](../../../results/research/ripper/balanced_mask.json) · [Figure](../../../assets/figures/balanced-mask.png)
