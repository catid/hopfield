# Conditional soundness of the fixed exclusion certificate

This is a post hoc mathematical result about the fixed H3f control. It is not a convergence guarantee, a claim about arbitrary learned CRAM weights, or evidence of generalization to unseen complete solution patterns.

Assume a 12×12 board, at most three mutually nonattacking clamped givens, and an inferred forbidden-pair graph that equals the true queen-attack graph. The last assumption was independently checked for this experiment after construction; it does not follow from the count rule on every possible dataset. The constructor itself did not use attack categories.

H3f uses `J_ij = 12 * forbidden_ij - 1` for distinct cells and zero unary energy. The training-derived minimum one-cell margin is 11. In exact integer arithmetic, the implemented margin threshold `11 - 0.001` therefore accepts precisely those free-cell flip costs that are at least 11.

The pinned `cram_nqueens.solve` code implements local stopping as `other_field - own_field >= model['delta'] - margin` at every free cell. The temperature unit enters only `T = temperature * model['unit']`, used by proposals; it does not normalize the stopping comparison. For this saved model, delta is exactly 11 and margin is 0.001. The float32 comparison threshold is 10.99899959564209, strictly between 10 and 11.

All interaction entries are exactly representable integers, −1 or 11, and states are binary. A conservative absolute field bound is 11×143=1,573. Absolute total energy is at most 11×binom(144,2)=113,256, and the energy-update intermediate is at most 114,829. These integer bounds are far below float32's exact-integer limit 2^24. The initial doubled-energy sum is also below that limit. Thus the discrete fields and energies remain exactly representable under the implemented additions and subtractions; proposal division by the noninteger temperature changes which state is visited, not the integer stopping comparison. TF32 was disabled in these runs. A synthetic sequence of 20,000 random single-cell flips independently checked the incremental float32 fields and energies against direct int64 recomputation every 100 steps. [The arithmetic validation receipt](../../../../results/research/loki/h3f/integer-arithmetic-validation.json) records the model hash, bounds and actual threshold. This check is not a convergence or task-accuracy measurement.

Let a candidate board contain M queens, of which k≤3 are clamped. Let d_i be the number of other occupied cells attacking an occupied free cell i. Deleting that queen changes energy by

\[
\Delta E_i^{\rm remove}=M-1-12d_i.
\]

Consequently, a stopped board must satisfy

\[
d_i\le M/12-1
\]

for every free queen. Consider the possible values of M.

1. **M<12 and at least one free queen.** Its deletion cost is at most M−1<11, contradicting the certificate.
2. **M=k≤3, with no free queen.** A queen attacks at most 44 other cells on a 12×12 board. The givens and their attacked cells cover at most 3×45=135 of the 144 cells. Some free empty cell is unattacked. Adding a queen there changes energy by −M, which is below 11. The certificate fails.
3. **12<M<24.** The bound above forces the nonnegative integer d_i to be zero for every free queen. Since the givens are mutually nonattacking, all M queens would be mutually nonattacking. More than 12 such queens cannot occupy 12 rows, a contradiction.
4. **M≥24.** Write F=M−k for the number of free queens. Let r_j and c_j count free queens in each row and column. Row and column edges are distinct, and the sum-of-squares inequality gives average degree within the free-queen attack graph at least

   \[
   \frac{\sum_j r_j(r_j-1)+\sum_j c_j(c_j-1)}{F}
   \ge \frac{2F}{12}-2.
   \]

   Attack degrees that also include the givens and diagonals are at least as large. But the certificate bounds every such degree by M/12−1. These bounds contradict each other because

   \[
   \frac{2(M-k)}{12}-2 > \frac{M}{12}-1
   \quad\Longleftrightarrow\quad M>12+2k,
   \]

   which holds for M≥24 and k≤3.

Only M=12 remains. The certificate then requires d_i=0 for every free queen. Together with mutually nonattacking givens, this gives exactly 12 nonattacking queens: a valid complete solution containing the input givens.

Thus every genuinely certified stopped board is valid **under these assumptions**. This does not bound the time required to reach one. In the measured H3f runs, 1,729 chains reached their cap and returned invalid boards with fewer than 12 queens; no wrong stopped board occurred. The audit recomputed the actual returned-board margins independently in float64 and found no stopped-certificate violations. Those finite checks do not replace the graph-equality and input assumptions in the proof.
