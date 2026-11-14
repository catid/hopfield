# Long-budget Sudoku and the no-optimizer control

**A binary energy built from training counts solves 99.9567% of the reserved Sudoku evaluations, with zero wrong stops.** It matches the transformed learned energy within the frozen 0.1-percentage-point noninferiority margin. This result uses a much longer search budget than the short-budget intervention figures.

The count rule connects two cells if they **never share a digit in the training solutions**, then assigns weight one to each retained pair and zero unary costs. It runs the same focused Gibbs search. No optimizer, trained coefficients or supplied row/column/box graph enters this construction. The data does carry task knowledge: the input contains ten unique base solutions and 99 stored Sudoku-symmetry variants per base. The [augmentation controls](../ripper/REPORT-augmentation.md) and [count study](../ripper/REPORT-count.md) explain this distinction. Counting is still a data-dependent estimator; it is not evidence that the task was solved without prior information.

## Frozen test and results

Both energies use the same 10,000 official test puzzles at `full-all[30000,40000)`, six paired search seeds, four chains, fixed temperatures and focus, 4,096-move checks, cyclic annealing with period 524,288, and no restarts. The primary comparison is at **16,777,216 moves per chain**. This corpus was used in the original reproduction; the slice was reserved for follow-up, and the count comparison was designed after seeing the learned method's curve. It is an adaptive comparison on reused inputs, not independent external confirmation.

| Method | Mean accuracy | Puzzles solved in all six seeds | Wrong stops |
|---|---:|---:|---:|
| BP AdamW | Outside this inference-only comparison | No matched local result | — |
| Transformed learned energy | 99.9417% (59,965/60,000) | 9,975/10,000 | 0 |
| No-optimizer count energy | **99.9567% (59,974/60,000)** | **9,981/10,000** | **0** |

The count-minus-learned difference is **+0.0150 points**, with approximate paired one-sided 95% endpoints **[−0.0151, +0.0467]**. The lower endpoint exceeds the frozen −0.1-point margin, supporting noninferiority. Twenty-two input–seed cases regress and 31 improve; this is not trajectory identity between integer and floating-point coefficient scales. The count method's all-six-seed joint success is **99.81%**, with exact one-sided 95% lower bound **99.7213%**, supporting the separate 99% joint-success claim.

All six runs use the same recovered graph. They are search-seed replications, not six independently learned structures. The count comparison resamples paired search seeds and shared puzzle indices; the joint-success bound treats one puzzle as the sampling unit.

## The earlier primary hypothesis was inconclusive

Before testing counts, the learned-energy continuation froze a stronger endpoint: at least 99% joint success across all six seeds by **4,194,304 moves per chain**. It observed 99.11%, but the one-sided 95% lower bound was **98.9394%**. That primary result remains **inconclusive**. At the separately prespecified 16,777,216-move endpoint, joint success rose to 99.75%, lower bound 99.6510%. The secondary result does not replace the earlier primary decision.

The learned continuations restored all search fields and separate best/certificate-board captures from the 262,144-move prefix. Every seed passed a byte-exact next-check comparison across Ripper and Foureyes before continuation. No completed prefix was rerun for production. The count runs were fresh searches from zero on this cohort. Each method saved one trajectory through every budget, rather than rerunning each point.

## A source-specific stopping certificate

For this **unit-weight, zero-bias, audited 810-edge graph**, the stopping energy is the integer number of equal-digit peer conflicts. Each cell has 20 distinct neighbors. Initial fields are integers in [0,20], and energy is an integer in [0,810]. A changed digit subtracts and adds one in distinct neighbor fields and changes energy by an integer in [−20,20]. When the digit does not change, the update is zero. Periodic refresh sums 0/1 products; the selected-field sum is at most 1,620 and division by two is exact. These quantities and operations are exactly representable in FP32.

Thus, under the pinned kernel's specified operations, exact integer fields and energy are preserved by induction. Temperature and Gumbel calculations choose the next digit but do not enter the integer energy accumulator. Compaction and board capture copy values. The threshold is `0 + 0.001 < 1`, so an accepted energy must be zero. Given the separately verified peer support and clamped clues, zero conflicts implies a valid Sudoku solution.

This is a conditional, implementation-specific soundness argument, **not a convergence guarantee** or a generic floating-point theorem. It does not apply to the unrounded learned coefficients. The source review covers [the derived backend](../ripper/sparse_backend.py), the [pinned sparse move kernel](https://github.com/Prefrontal-AI/CRAM/blob/62bbe830c7d6f87f2ba18f30a28039258b986454/cram/cram_sudoku_sparse.py), and its [general refresh](https://github.com/Prefrontal-AI/CRAM/blob/62bbe830c7d6f87f2ba18f30a28039258b986454/cram/cram_sudoku_general.py). Independent audits also recomputed every saved count field and energy from the boards with integer arithmetic.

## Reproducibility

- Frozen [learned budget protocol](budget.json), [count protocol](count_budget.json), and their [clarifications](BUDGET_ERRATA.md).
- Public [learned curve and hashes](../../../results/research/foureyes/budget/summary.json), [count curve and comparison](../../../results/research/foureyes/count_budget/summary.json), and [per-seed count rows](../../../results/research/foureyes/count_budget/per_seed.csv).
- [Continuation runner](budget.py), [count runner](count_budget.py), [learned auditor](audit_budget.py), [count auditor](audit_count_budget.py).

Each audit independently checked **420,000 saved boards**, their official puzzle identities, clues, budgets, source/output hashes and preservation of all earlier stopped boards. All seven exact-binomial endpoint pairs were also checked against independent SciPy beta inversion. Raw board and state archives remain outside Git; public receipts retain their hashes. Compute timings exclude setup, validation and serialization; the count trajectory starts at zero, while learned continuation timing excludes its previously completed prefix. No cross-host speedup is inferred.
