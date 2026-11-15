# H1be: feedback from the first 300 rows meets the fixed 900-row recovery target

Freezing only the acquisition policy's equality feedback after 300 rows recovers **29/32 certified integral exact graphs**, meeting the prespecified target of at least 29. Continuing feedback through all 900 rows also recovered 29/32 in the archived paired H1bd arm. This supports the sufficiency of this fixed prefix intervention on these examined streams; equal aggregate success does not imply bankwise equivalence.

| Procedure | Certified integral exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Archived feedback updated through all 900 rows | 29/32 | 3 | 0 | 0 |
| Policy feedback frozen after row 300 | **29/32** | **3** | **0** | **0** |

The frozen arm fixes the three archived failures at banks 5, 11, and 20, but fails on three previously successful banks: 10, 15, and 26. Each new failure has two missing and two extra edges. Thus the paired comparison has three wins and three losses. Same-arm top-810 recovery remains 0/32 exact, with 234 missing and 234 extra edge-bank pairs, compared with 243 of each for the archived full-feedback costs. Per-cell degree constraints remain consequential.

The only acquisition change is the fixed equality-feedback snapshot. For the first 300 rows, the greedy policy uses its actual past equality counts. After observing row 300, it copies that matrix and uses the immutable snapshot for every subsequent mask. Its weights remain `(8/9)^actual_joint_count / (1 + policy_feedback)`. **Actual joint and equality counts continue updating through all 900 rows**, and final LP recovery uses all actual observed counts. The post-prefix labels are therefore still acquired and used for recovery; they cease to influence future acquisition masks.

Independent replay confirms the first 300 masks and labels match the archived aligned arm, the frozen matrix equals those first 300 observed equality counts, and all 600 later policy-feedback digests are identical. The final actual equality-count sum is **835,864**, while the sum of frozen feedback snapshots is **266,281** across 32 banks. Synthetic substitution of later labels leaves every later mask unchanged while changing actual final equality counts. The 32 raw future-suffix gates substitute sources and noise from row 450 onward and preserve the first 450 masks, labels, counts, and frozen snapshot.

This is one frozen follow-up cutoff on the now-examined H1bd sources, mask-independent corrupted boards, and tie streams. There was no cutoff sweep, replacement bank, new source identity, latent draw, calibration, or puzzle-solver input. The result does not establish a minimum useful feedback prefix, universal necessity or irrelevance of continued feedback, a population success probability, or a unique causal mediator. In particular, the different failure banks show that continued feedback changes outcomes even when aggregate success is equal.

Recovery retains all 3,240 candidate pairs, exact half-even rounded Laplace equality-rate costs, and degree exactly 20 at every cell. Degree and implied 810-edge cardinality remain supplied priors; geometry is used only for evaluation. All **32 new and 32 archived exact integer primal-dual proofs** pass, with no fractional or unresolved outputs. Archived fits are not rerun. Certificates establish optimality, not uniqueness; no primal snapping is allowed.

| Physical query accounting on the H1bd streams | Count |
|---|---:|
| Frozen arm training queries | 720,000 |
| Reused from the union of both H1bd arms | 492,215 |
| **Newly exposed label instances** | **227,785** |
| Prior H1bd paired union | 1,216,156 |
| **H1bd stream union after this study** | **1,443,941** |
| New calibration queries | **0** |

This alternative still costs 720,000 training queries and zero calibration; it offers **no query saving**. Actual new spending is `frozen_mask AND NOT (old_aligned_mask OR old_permuted_mask)`, counted over bank, row, and cell. The H1az–H1be acquisition sequence totals **3,465,890** physical labels, which is an accounting total for that sequence, not all historical Ripper experiments. The Ripper source-ID ledger stays at 2,016,891.

Execution used **19.394482 aggregate CPU-seconds and 3.265744 wall seconds**, within frozen 240 CPU-second / 120-second limits. The 34 wrapper commands cover preparation, 32 acquisitions and LPs, independent sequential replay, 32 suffix gates, new and archived proof checks, query accounting, and summary. Synthetic preflight, freeze registration, and packaging are separate. Eight one-thread workers were observed at 100–102% CPU while the command counter advanced. Final accounting has no active owned PID. The eligible study is CPU-only, so the four Ripper GPUs remained idle.

The public verifier checks the first 300 mask matches, all mask exposures and physical unions, snapshot totals, actual-count rate costs, proofs, graph outcomes, and the finite endpoint; it rejects **192 altered certificates**. It does not independently replay labels, prove snapshot fidelity, or rerun suffix substitutions without the raw observations. Those checks were performed by the independent raw audit.

[Frozen protocol](protocol-frozen-feedback.json) · [Policy](frozen_feedback_policy.py) · [Independent audit](audit_frozen_feedback.py) · [Public verifier](verify_frozen_feedback_public.py) · [Results](../../../results/research/ripper/frozen_feedback.json) · [Figure](../../../assets/figures/frozen-feedback.png)
