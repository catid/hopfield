# H1bd: soft feedback confirms both finite targets on new Ripper-ledger sources

Both prespecified endpoints are **separately supported**. Aligned soft feedback recovers **29/32 certified integral exact graphs**, meeting the procedure target of at least 29. Permuted feedback recovers **20/32**, giving an exact-bank difference of **+9**, above the separate target of +4. There are no unresolved proofs. The control's one certified fractional optimum is retained as a resolved procedure failure.

| Procedure | Certified integral exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Aligned soft equality feedback | **29/32** | **3** | **0** | **0** |
| Cell-permuted soft equality feedback | **20/32** | **11** | **1** | **0** |

Aligned feedback wins on ten paired banks and loses on one. Its failures are banks 5 and 11, each with two missing and two extra edges, and bank 20 with three of each. The permuted arm's eleven integral failures have 37 missing and 37 extra edge-bank pairs; its fractional bank 9 has six half-integral variables and is not converted into a graph. Same-arm top-810 controls recover 0/32 exact graphs in both arms, with 243 missing/extra pairs for aligned costs and 254 for permuted costs. These controls use the same costs, so per-cell degree constraints remain consequential.

The frozen protocol drew 32 independent banks of 900 official training rows, without replacement within each bank, from the 1,843,698 identities outside **all 1,988,296 IDs in the declared archived Ripper completion-source ledger**. Source identities were sealed before reading selected labels. There are **28,595 new unique IDs**, 28,800 bank-row instances, 205 repeated instances across banks, and no within-bank or declared-ledger overlap. The updated Ripper ledger contains **2,016,891 unique IDs**. These identities are disjoint from that declared ledger, not necessarily from all other repository experiments; this is not external-distribution evidence. All 28,595 selected CSV answers and their Sudoku constraints were independently checked. No puzzle-solver input was used.

Both arms retain H1bc's fixed policy and recovery recipe. Before querying the current row, the greedy policy uses

\[
w_{ij}=\frac{(8/9)^{m_{ij}}}{1+c_{ij}},
\]

where m and c are past actual coquery and equality counts. The control replaces only the policy's equality feedback with `c[permutation, permutation]`, using one independently seeded cell permutation per bank fixed in the protocol before source selection. Each arm has its own real observation history; joint counts and recovery costs always use true coordinates. There is no calibration, hard exclusion, likelihood-ratio state, or supplied noise probability. The alphabet-dependent 8/9 rule and exponent one remain fixed design choices.

Each bank-row has a fresh mask-independent corrupted board: draw U and a uniform permutation of 81 positions, and permute the full clean board iff U < 9/160. The paired arms share this underlying board and tie stream. Source, noise, tie, and feedback-permutation seeds are all new. Independent sequential replay checks every state digest, mask, noise draw, selected label, and count update. Each of 64 stream audits also substitutes sources and noise from row 450 onward and confirms that the first 450 masks, labels, and count state remain unchanged.

Recovery uses all 3,240 pairs, exact half-even rounded Laplace equality-rate costs, and an LP imposing degree exactly 20 at each cell. Degree and the implied 810-edge cardinality remain supplied structural priors; Sudoku geometry is used only for evaluation. All **64 exact integer primal-dual proofs** pass. No primal values are snapped. Certificates establish optimality, not uniqueness; the control's fractional optimum does not rule out an integral co-optimum.

The procedure endpoint is exactly 29/32, and the alignment endpoint is exactly +9, because every proof resolved. This supports these two finite targets on the sealed banks; it does not establish a population confidence bound or identify a unique causal mediator. The permutation intervention changes feedback location and the subsequent trajectories, including exposure and feedback magnitude. The earlier H1bc result was 31/32 versus 26/32 on examined sources; it is not pooled into these fresh endpoint decisions.

| New physical paired-query accounting | Count |
|---|---:|
| Aligned training label instances | 720,000 |
| Permuted training label instances | 720,000 |
| Shared between arms, counted once | 223,844 |
| **Actual new union** | **1,216,156** |
| Calibration label instances | **0** |

Each alternative deployment uses 720,000 queries and zero calibration. Physical spending is the union over `(bank, row, cell)`, so `1440000 - 223844 = 1216156`; shared labels count once. Cross-bank repeats of a clean source ID have different latent instances and remain separately charged. The preceding H1az–H1bc acquisition sequence used 2,021,949 physical labels, so that sequence plus this disjoint study totals **3,238,105**. That is an accounting total for this sequence, not all historical Ripper studies.

Execution used **47.616919 aggregate CPU-seconds and 20.723922 wall seconds**, within frozen 900 CPU-second / 180-second ceilings. This includes identity selection and seal, source materialization, independent CSV/ledger audit, 64 acquisitions and LPs, replay, 64 suffix gates, proofs, accounting, and summary. Registration, synthetic preflight, and packaging are separate. The main measured bottleneck was serial source materialization/audit: all eight one-thread bank workers then ran near 100% CPU each. All 36 wrapper commands completed and no owned child remained active. The eligible computation was CPU-only; all four Ripper GPUs stayed idle, with no filler GPU work.

The public verifier replays source-ID draws and disjointness from the exported exclusion set, permutation seeds, mask exposures and query unions, rounded costs, all 64 proofs, outcomes, and separate endpoint bounds. It rejects **192 altered certificates**. Reconstructing the historical ledger from completion archives, checking original CSV labels, sequential policy/latent replay, and future-suffix substitution require the raw archive; these were checked by the independent raw auditors.

[Frozen protocol](protocol-fresh-soft-feedback.json) · [Study](fresh_soft_feedback_study.py) · [Independent audit](audit_fresh_soft_feedback.py) · [Public verifier](verify_fresh_soft_feedback_public.py) · [Results](../../../results/research/ripper/fresh_soft_feedback.json) · [Figure](../../../assets/figures/fresh-soft-feedback.png)
