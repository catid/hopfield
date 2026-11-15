# H1au: an explicit degree prior closes the residual recovery gap

The frozen hypothesis is **supported**: all 32 degree-constrained LP outputs are integral, recover the exact Sudoku graph, and have independently verified exact optimality certificates. Keeping the same-cost 810 lowest-cost edges recovers only 7 of 32 graphs.

| Method and supplied information | Exact graphs | Missing true edges | Retained extras | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Original H1as support | 0/32 | 0 | 2,588 | 0 |
| Same-cost top-810; graph size supplied | 7/32 | 45 | 45 | 0 |
| Degree-20 LP; every cell's degree supplied | 32/32 | 0 | 0 | 0 |

This intervention supplies structural information: each of 81 cells has exactly 20 neighbors, which also implies 810 edges. The cardinality control receives 810 directly. The optimization does not receive the row, column, or box identities, but the degree and cardinality are explicit priors. This is not recovery without structural prior information.

Each bank uses only candidate edges retained by H1as and their archived final coquery counts m and equal-label counts c. The cost is

\[
w_e=\operatorname{roundEven}\!\left(10^6\left[c_e\log(1600/81)+(m_e-c_e)\log(12800/14319)\right]\right).
\]

These are quantized original q1 = 1/9 log-likelihood ratios against supplied q0 = 9/1600. Decimal80 computes the costs; an independent Decimal110 reconstruction verifies them. Top-810 sorts by integer cost and then lexicographic unordered-pair index. The LP minimizes Σ wₑxₑ subject to 0 ≤ xₑ ≤ 1 and Σₑ∋ᵢ xₑ = 20 for each cell. It cannot reinstate previously excluded edges. All source banks, costs, quantization, ties, solver options, certificates, and budgets were frozen before any real LP output.

HiGHS 1.12.0 uses one-thread dual simplex, presolve off, a fixed bank-indexed seed, and a one-second solver limit. No primal value is rounded or snapped. The fixed certificate scheme accepts only literal primal values in {0, 1/2, 1}; all 32 real outputs are integral. Node duals are reconstructed at scale two, within a frozen 1e-6 reconstruction tolerance. With y₂ and z₂ = min(0, 2w − y₂,i − y₂,j), an independent checker uses integer arithmetic to verify primal bounds and degrees, dual inequalities, both complementarity conditions, and identical primal and dual objectives. Acceptance therefore does not depend on trusting the solver's optimal status. The public checker can verify these certificates without invoking HiGHS; it also rejects 96 altered certificates.

An exact certificate proves optimality of the returned graph for the frozen LP. It does not establish uniqueness. A certified fractional output would have been a resolved failure of this fixed procedure, without proving that no integral optimum exists. Numerical, timing, or certification gaps would have remained unresolved. The primary required at least 29 successes; it is supported at 32 successes, zero resolved failures, and zero unresolved outcomes. No bank was replaced or filtered. Integral graph recovery and practical certification status are retained separately in the results.

The top-810 control's 45 missing and 45 extra edge-bank pairs show that known cardinality plus these costs is insufficient on 25 banks. The degree-constrained procedure succeeds on those banks as well. This supports the utility of the additional per-cell degree information for this fixed study; it does not establish fresh-source generalization or universal superiority of the procedure.

The study uses the existing observations representing 888,000 historical training queries, with zero new queries, calibration fits, source rows, solver puzzle inputs, or GPU work. It does not rerun acquisition. Execution recorded **7.529170 aggregate process/child CPU-seconds** and **0.414825 seconds for the monitored execution phase**, within the frozen 120 CPU-second / 60-second ceilings. These clocks have different origins: CPU accounting includes wrapper startup/import work, while the wall timer begins after imports and initial verification. They should not be divided to infer parallel speedup. The 32 LP calls themselves total 0.011523 solver CPU-seconds; no call requires more than seven simplex iterations.

A separate post-run import-only diagnostic found that setting BLAS thread variables before Python reduces import CPU consumption from 1.604 seconds to 0.028 seconds in that measurement, with similar import wall time. The frozen scientific run was not rerun or modified. The diagnostic is separate from its execution budget; future launch commands should set thread limits before Python starts. Synthetic preflight and publication packaging are also separate costs.

The public archive includes final counts, costs, raw primal/dual outputs, exact certificates, original and top-810 supports, and all bank classifications. The count export binds each bank to the archived H1as fit hash. Original observation and acquisition provenance still requires the preserved raw replay.

[Frozen protocol](protocol-degree-prior.json) · [Exact checker](audit_degree_prior.py) · [Public verifier](verify_degree_prior_public.py) · [Results](../../../results/research/ripper/degree_prior.json) · [Figure](../../../assets/figures/degree-prior.png)
