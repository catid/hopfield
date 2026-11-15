# H1bg: fixed donor-feedback alignment meets the four-bank contrast target

The prespecified alignment contrast is **supported exactly at its threshold**: archived aligned donor feedback recovers **30/32** certified integral exact graphs, while the prospective cell-permuted control recovers **26/32**, a difference of **+4**. The control also has four certified integral nonexact outputs and two certified fractional outputs, all retained as resolved failures. There are no unresolved proofs, so both contrast bounds equal +4.

| Procedure | Certified integral exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Archived aligned fixed donor feedback | **30/32** | 2 | 0 | 0 |
| Cell-permuted fixed donor feedback | **26/32** | 4 | 2 | 0 |

Aligned feedback wins on six paired banks and loses on two. The control's integral failures are banks 13 and 14 with two missing and two extra edges each, bank 22 with four of each, and bank 30 with three of each. Fractional banks 10 and 28 contain 12 and 8 half-integral variables respectively; they are not converted into graphs. Control top-810 recovery is 0/32 exact, with 249 missing and 249 extra edge-bank pairs, versus 190 of each for archived aligned costs.

This control changes only the cell coordinates of the fixed matrix supplied to acquisition: `control_feedback = donor_feedback[permutation, permutation]`. Each of 32 permutations uses seed `202611860000 + recipient` and was embedded in the protocol before control masks or recovery outcomes. Donor assignment, donor-prefix size, underlying source/noise streams, tie streams, mask policy, and recovery recipe match H1bf. The original and permuted matrices have **identical full off-diagonal value histograms**, and both remain immutable for all 900 rows.

The static matrices address a limitation of the earlier evolving-feedback comparison: there is no difference in how their global feedback-value distribution develops over time, because neither matrix changes. The finite +4 advantage supports this aligned fixed-feedback procedure over this fixed permutation control. It does not identify a unique mediator: the location of feedback still interacts with the recipient coquery counts and changes the selected-cell trajectory. It also does not establish population significance or universal alignment benefits.

The aligned count of 30 was already known when the contrast was designed. It is an archived comparator, not a new procedure endpoint. The only primary target was `aligned_exact - control_exact >= 4`, with bounds `[30 - control_success - control_unresolved, 30 - control_success]`. The design uses one fixed control permutation per recipient, without searching for a favorable permutation. Bank outcomes are coupled through the donor assignment, and the source/noise streams were already examined.

As in H1bf, each donor's first 300 source IDs are disjoint from its recipient's 900 source IDs, every donor is used once, and the matrix is reconstructed from legitimate donor warmup labels. All 900 recipient masks are generated and saved before loading recipient source boards or latent arrays. Recipient labels never feed back into mask selection. All 32 complete-label substitution gates change every queried value and final equality counts while leaving every mask and coquery count unchanged.

Recovery uses all 3,240 pairs, exact half-even rounded Laplace equality-rate costs, and degree exactly 20 at each cell. Degree and the implied 810-edge cardinality remain supplied priors. All **32 new and 32 archived exact integer primal-dual proofs** pass, without primal snapping or archived refits. The two certified fractional optima count as procedure failures; their existence does not rule out integral co-optima. Proofs establish optimality, not uniqueness, and integral edge-error totals exclude fractional outputs.

| Alternative deployment accounting | Aligned archived | Permuted control |
|---|---:|---:|
| Recipient training queries | 720,000 | 720,000 |
| Donor warmup queries | 240,000 | 240,000 |
| Warmup/recipient overlap | 74,072 | 74,108 |
| **Exact deployment union** | **885,928** | **885,892** |

Both procedures have the same gross recipient and donor budgets. The exact label unions differ by 36 because their masks overlap the warmup differently. Donor acquisition is included, and no query-saving claim is made.

The control reuses **529,877** recipient answers from the four preceding acquisitions and exposes **190,123 new label instances**. The previous union on the H1bd streams was 1,718,317; the updated union is **1,908,440**. All donor warmups were already acquired. The H1az–H1bg acquisition sequence totals **3,930,389** physical labels, explicitly not all historical Ripper research. There is no new source identity, latent draw, calibration, or puzzle-solver input; the source ledger remains 2,016,891 IDs.

Execution used **20.603275 aggregate CPU-seconds and 3.894048 wall seconds**, within frozen 240 CPU-second / 120-second ceilings. All 34 wrapper commands completed, covering source/donor checks, 32 control mask streams and LPs, independent mask/count replay, 32 complete-label gates, new and archived proofs, accounting, and summary. Synthetic preflight, registration, and packaging are separate. Eight one-thread workers were observed at 102–104% CPU while command counters advanced. Final accounting has no active owned PID. The eligible study is CPU-only, so all four Ripper GPUs stayed idle.

The public verifier checks all fixed cell-permutation seeds, exact feedback histograms, donor assignment, and **all 28,800 control masks**, as well as deployment/incremental query unions, rate costs, new and archived proofs, graph outcomes, and contrast bounds. It rejects **192 altered certificates**. Original donor/recipient labels and complete-label substitution gates require the raw archive and were checked by the independent raw auditor.

[Frozen protocol](protocol-permuted-transfer.json) · [Study](permuted_transfer_study.py) · [Independent audit](audit_permuted_transfer.py) · [Public verifier](verify_permuted_transfer_public.py) · [Results](../../../results/research/ripper/permuted_transfer.json) · [Figure](../../../assets/figures/permuted-transfer.png)
