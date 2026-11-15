# H1bf: a fixed feedback snapshot transfers between source-disjoint banks

A 300-row equality snapshot from another bank supports **30/32 certified integral exact graphs** without recipient-label feedback during acquisition, exceeding the prespecified target of at least 29. The archived same-bank prefix procedure recovered 29/32. Transfer has two paired wins and one loss, but this is not a superiority result: the deployment requires additional donor acquisition, and the comparison was not an independent fresh confirmation.

| Procedure | Certified integral exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Archived same-bank feedback frozen after row 300 | 29/32 | 3 | 0 | 0 |
| Fixed other-bank feedback throughout acquisition | **30/32** | **2** | **0** | **0** |

Transfer fixes archived failures 10 and 15, loses previously successful bank 19, and both procedures fail on bank 26. The transfer failures have two and four missing edges respectively, with equal numbers of extra edges. The same-arm top-810 control remains 0/32 exact, with 190 missing and 190 extra edge-bank pairs. Degree constraints still contribute beyond ranking costs.

The donor assignment was frozen before new recipient masks. A deterministic augmenting-path matcher processes recipients and donor candidates in increasing index order, forbids self-donation, and rejects any donor whose first 300 source IDs intersect the recipient's 900 source IDs. It assigns each of 32 donors exactly once. Every assigned donor-prefix/recipient pair is source-disjoint. The fixed map uses only source identities; no recovery outcome, feedback quality, or mask result enters matching. Failure to find a valid map would have left the study incomplete rather than relaxing the rule.

Each donor's snapshot is reconstructed from its actual archived aligned labels in the first 300 rows, identical to the H1be snapshot. The recipient uses the immutable matrix for all 900 masks, with weights `(8/9)^recipient_coquery_count / (1 + donor_equality_count)`. Only recipient coquery counts update during mask generation. The implementation saves all 900 masks before loading that recipient's clean source boards or latent arrays, and it uses no recipient equality feedback at any point. Final recovery then uses all 900 rows of actual queried labels.

Independent audit reconstructs each donor snapshot from observed labels, reproduces the source-only assignment, and replays every recipient mask. In each of 32 complete-label substitution gates, every recipient source value is changed. All queried values and final equality counts change, while **every acquisition mask and coquery count remains identical**. This validates the declared absence of recipient-label feedback; it does not remove the legitimate label information already learned from donors.

This study uses the same examined H1bd source rows, corrupted boards, and tie streams. All banks share the same indexed structural graph. The donor permutation couples their outcomes, so the 30/32 result is a finite target, not an independent-trial population estimate. It does not establish transfer to a different graph or distribution, a cost-free prior, superiority over same-bank adaptation, or the minimum donor prefix. There is one fixed 300-row prefix and one deterministic assignment, with no donor or cutoff sweep.

Recovery retains all 3,240 pairs, exact half-even rounded Laplace equality-rate costs, and degree exactly 20 at each cell. Degree and implied 810-edge cardinality are supplied priors; geometric relations appear only in evaluation. All **32 new and 32 archived exact integer primal-dual proofs** pass. Every returned solution is integral; there are no unresolved outputs or snapped primals. Proofs establish optimality, not uniqueness. Archived fits are not rerun.

| Standalone transfer deployment accounting | Label instances |
|---|---:|
| Recipient training queries | 720,000 |
| Donor warmup queries | 240,000 |
| Warmup/recipient overlap, counted once | 74,072 |
| **Exact deployment union** | **885,928** |

The deployment first acquires all donor warmups, then uses their snapshots for recipient masks. Since every donor is used once, each source bank contributes one 300-row warmup. Exact accounting takes the union of that bank's warmup and recipient masks over `(bank, row, cell)`: `720000 + 240000 - 74072 = 885928`. This is **165,928 more labels** than the 720,000-query same-bank prefix deployment. No calibration is added, but donor acquisition is a real cost and is not omitted.

| Actual incremental research accounting | Label instances |
|---|---:|
| Reused recipient answers from the three preceding acquisitions | 445,624 |
| **New recipient answers** | **274,376** |
| Prior union on the H1bd streams | 1,443,941 |
| **Updated union on the H1bd streams** | **1,718,317** |

All donor warmup labels were already present in the preceding union. Actual new spending counts only recipient cells absent from the union of H1bd aligned, H1bd permuted, and H1be frozen masks. The H1az–H1bf acquisition sequence now totals **3,740,266** physical labels; that is not a total for all historical Ripper research. No source identity, latent draw, or puzzle-solver input is new, and the Ripper source ledger remains 2,016,891 IDs.

Execution used **20.042212 aggregate CPU-seconds and 3.624059 wall seconds**, within frozen 240 CPU-second / 120-second limits. All 34 wrapper commands completed. Eight one-thread workers were observed at 100–102% CPU while counters advanced; final accounting has no active owned PID. These bounds cover donor/source checks, mask generation and acquisition, 32 new LPs, independent replay, 32 complete-label gates, new and archived proofs, accounting, and summary. Synthetic preflight, registration, and packaging are separate. The eligible study is CPU-only, so all four Ripper GPUs remained idle.

The public verifier independently reconstructs the source-disjoint donor assignment, checks snapshot lineage, and replays **all 28,800 recipient masks** from exported fixed feedback. It checks complete deployment and incremental unions, rate costs, proofs, graph outcomes, and the finite endpoint, and rejects **192 altered certificates**. Original donor and recipient labels, their counts, and complete-label substitution gates require the raw archive; those were checked by the independent raw auditor.

[Frozen protocol](protocol-transfer-feedback.json) · [Policy](transfer_feedback_policy.py) · [Independent audit](audit_transfer_feedback.py) · [Public verifier](verify_transfer_feedback_public.py) · [Results](../../../results/research/ripper/transfer_feedback.json) · [Figure](../../../assets/figures/transfer-feedback.png)
