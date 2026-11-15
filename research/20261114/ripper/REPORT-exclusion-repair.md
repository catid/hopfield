# H1av: repairing premature exclusions with the same observations

The frozen hypothesis is **supported**. Allowing all 3,240 pairs in the degree-20 LP recovers the exact graph in **32/32 banks**, each with an independently checked exact optimality certificate. Keeping the original candidate restrictions recovers 11/32; all remaining 21 banks have direct integer proofs that the restricted LP is infeasible.

| Procedure | Exact graphs | Missing true edges | Extras | Certified infeasible | Unresolved |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — |
| Original H1ap support | 4/32 | 41 | 45 | — | 0 |
| Top-810, all pairs eligible | 16/32 | 24 | 24 | — | 0 |
| Degree-20 LP, original candidate restrictions | 11/32 | 0 in 11 feasible outputs | 0 in 11 feasible outputs | 21/32 | 0 |
| Degree-20 LP, all pairs eligible | 32/32 | 0 | 0 | 0/32 | 0 |

H1ap used a deliberately misspecified scalar calibration under shared board-permutation corruption. Its permanent exclusions lost 41 true edge-bank pairs across 21 banks. H1av does not acquire or replay observations. It uses the same archived final coquery/equality counts and supplied exact equality null q0 = 9/1600. Both LP arms receive the explicit degree-20 prior; the cardinality control receives the implied graph size, 810. The unrestricted LP can reinstate previously excluded pairs, while the hard LP cannot.

Both arms use the unchanged H1au integer costs: round-half-even of 10⁶ times c log(1600/81) + (m−c) log(12800/14319), the original q1 = 1/9 likelihood ratio against supplied exact q0. Decimal80 produces costs and an independent Decimal110 audit reconstructs them. Top-810 sorts by cost, then lexicographic pair index. The LP minimizes the same weighted edge sum with bounds 0 ≤ x ≤ 1 and exactly 20 incident edges at every cell. Only candidate availability differs between the two LP arms. No Sudoku row, column, or box identities enter optimization, but degree, cardinality, and the noise law are explicitly supplied information.

In every bank that had lost a true edge, at least one cell has fewer than 20 allowed candidate neighbors. The archived witness identifies that cell and its exact allowed degree. Selecting every allowed incident edge still cannot satisfy degree 20, which certifies LP infeasibility and permits skipping that redundant solve. This is stronger than the separate observation that a missing true edge makes exact truth recovery impossible under the hard restrictions. A missing true edge alone would not establish LP infeasibility.

The 11 feasible hard LPs and all 32 unrestricted LPs return integral exact graphs. The unrestricted solutions restore all 41 original missing edge-bank pairs and remove all 45 original extras. The unrestricted top-810 control retains 24 missing and 24 extra pairs, so known cardinality plus these costs does not suffice in half the banks.

All actual LPs use the frozen one-thread HiGHS 1.12.0 dual-simplex recipe with presolve off, fixed seeds, and a one-second limit. No primal is rounded or snapped. Independent integer checks verify primal degree/bound constraints, scale-two dual inequalities, complementarity, and equal objectives. The public verifier checks 43 exact optimality certificates and 21 degree-deficit infeasibility certificates without invoking HiGHS; 129 altered optimality certificates are rejected. These proofs establish optimality of the returned solutions, not uniqueness. An uncertified solver-infeasible status would have remained unresolved.

The primary required at least 29 independently certified integral exact graphs in the unrestricted arm. All 32 succeeded, with no numerical, fractional, timing, or certification gaps. The frozen decision would be falsified only if successes plus unresolved cases could not reach 29; otherwise it would remain inconclusive. All banks and both arms are retained.

Execution used **2.581840 aggregate CPU-seconds and 0.582185 wall seconds**, within the frozen 120 CPU-second / 60-second limits. There were 43 solver calls; 21 hard calls were skipped using exact infeasibility witnesses. Actual LP calls total 0.027660 solver CPU-seconds, with at most six simplex iterations. Thread limits are set before numerical imports, and the wall timer starts before those imports. Synthetic preflight and publication packaging are separate from the scientific execution budget.

The inputs represent 888,000 historical training queries and 64,000 historical calibration queries. The repair costs use the training counts and supplied exact law; no calibration is refitted. This study adds zero observations, queries, source rows, solver puzzle inputs, or GPU work. The banks were already examined in H1ap. The result tests repair on these fixed observations, not fresh-source generalization or arbitrary correlated-noise robustness.

[Frozen protocol](protocol-exclusion-repair.json) · [Independent audit](audit_exclusion_repair.py) · [Public verifier](verify_exclusion_repair_public.py) · [Results](../../../results/research/ripper/exclusion_repair.json) · [Figure](../../../assets/figures/exclusion-repair.png)
