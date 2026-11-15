# H1aw: smoothed equality rates recover the graph without supplied noise probabilities

The frozen hypothesis is **supported**. The Laplace-smoothed equality-rate degree-20 LP recovers **32/32 exact graphs**, each with an independently checked exact optimality certificate. It matches the archived exact-law LP. Keeping the same-rate 810 lowest-cost edges recovers 16/32 graphs.

| Procedure | Exact graphs | Missing true edges | Extras | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Original H1ap support | 4/32 | 41 | 45 | 0 |
| Archived exact-law degree-20 LP | 32/32 | 0 | 0 | 0 |
| Smoothed-rate top-810 | 16/32 | 23 | 23 | 0 |
| Smoothed-rate degree-20 LP | 32/32 | 0 | 0 | 0 |

Each pair with m archived coqueries and c equal observed labels receives the integer cost

\[
w_e=\operatorname{roundEven}\!\left(10^6\frac{c_e+1}{m_e+2}\right).
\]

The producer computes this with exact integer quotient and remainder. The independent auditor uses exact `Fraction` rounding. Both odd and even half ties are tested; an unexposed pair receives cost 500000. This fixed cost uses no supplied q0 or q1. It replaces the likelihood and exposure weighting and introduces Laplace smoothing, so the experiment does **not** isolate only the effect of removing a noise-law parameter.

All 3,240 pairs are eligible. The LP minimizes total cost under bounds 0 ≤ x ≤ 1 and degree exactly 20 at every cell. The top-810 control sorts the same costs, breaking ties by lexicographic unordered-pair index. Degree 20 and the implied graph size 810 remain explicit structural priors. Sudoku row, column, and box identities are not supplied to the optimizer. All 32 already-examined H1ap banks are retained, with no new labels or acquisition replay.

The result demonstrates that **this specific smoothed-rate recovery procedure does not require supplied noise probabilities on these fixed streams**. It is not a fresh-source claim or a guarantee for arbitrary correlated noise. Had it failed, that would not have established that calibration was necessary. The same-rate cardinality control still loses 23 true edge-bank pairs and retains 23 extras, showing that its graph-size information alone is insufficient on 16 banks.

The exact-law comparator is an archived result from H1av. Its counts, raw primal/dual outputs, costs, and exact optimality certificates are reverified; its LPs are not rerun. The new LP recipe remains one-thread HiGHS 1.12.0 dual simplex, presolve off, fixed bank seeds, and a one-second solver limit. No primal value is rounded or snapped. Independent integer checks verify degree/bound feasibility, scale-two dual inequalities, complementarity, and equal primal/dual objectives. All 32 new solutions are integral; there are no fractional, nonexact, or unresolved results. Certificates establish optimality of returned solutions, not uniqueness.

The primary required at least 29 certified integral exact graphs. Its lower and upper success bounds are both 32. The public verifier independently checks all 32 new certificates and all 32 archived comparator certificates without invoking an LP solver, and rejects 192 altered certificates.

Execution used **2.653717 aggregate CPU-seconds and 0.603261 wall seconds**, within the frozen 120 CPU-second / 60-second ceilings, with thread limits set before numerical imports. The 32 new LP calls total 0.040543 solver CPU-seconds, with at most 157 simplex iterations. These costs include preparation, cost and certificate audits, comparator re-verification, and summary generation. Synthetic preflight and publication packaging are separate.

The archived streams represent 888,000 historical training queries and 64,000 historical calibration queries. The new costs use only training counts and the fixed smoothing rule; calibration is not refitted. This study consumes zero new observations, queries, source rows, solver puzzle inputs, and GPU work.

[Frozen protocol](protocol-laplace-rate.json) · [Independent audit](audit_laplace_rate.py) · [Public verifier](verify_laplace_rate_public.py) · [Results](../../../results/research/ripper/laplace_rate.json) · [Figure](../../../assets/figures/laplace-rate.png)
