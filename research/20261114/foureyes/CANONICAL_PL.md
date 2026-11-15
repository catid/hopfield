# A convex canonical form for complete digit-averaged PL

This argument concerns the exact average over all $9!$ digit relabelings, with an entrywise L1 penalty on the raw unordered-pair matrices and unary vectors. It does not identify the paper's finite SGD trajectory with the new solver.

Write the categorical energy as

$$E(x)=\sum_{i<j}W_{ij}(x_i,x_j)+\sum_i B_i(x_i).$$

For a fixed training set and its free-cell masks, pseudolikelihood (PL) is convex in the energy parameters: each conditional negative log probability is a linear term plus a log-sum-exp. Let $\bar L$ be its exact average over every simultaneous permutation of the nine digit labels. This objective, and the uniform entrywise L1 penalty, are invariant under that group.

Average any parameter vector over the group. Convexity implies that neither the averaged loss nor the L1 penalty increases. The resulting pair matrix has the form

$$W_{ij}=d_{ij}I_9+o_{ij}(\mathbf1\mathbf1^\top-I_9),$$

and each unary vector is constant. Subtracting $o_{ij}\mathbf1\mathbf1^\top$ from a pair matrix changes every complete board's energy by the same constant, leaving every conditional probability unchanged. Subtracting the constant unary likewise leaves PL unchanged. With $J_{ij}=d_{ij}-o_{ij}$, the remaining parameters are

$$W_{ij}=J_{ij}I_9,\qquad B_i=0.$$

The gauge change cannot increase the penalty:

$$9|d_{ij}-o_{ij}|\leq9|d_{ij}|+9|o_{ij}|
\leq9|d_{ij}|+72|o_{ij}|.$$

Thus, for raw penalty coefficient $\lambda>0$, a global minimizer exists in canonical form. Existence follows from continuity and coercivity of the positive L1-regularized objective. The equivalent canonical problem is

$$\min_{J\in\mathbb R^{3240}} L(J)+9\lambda\sum_{i<j}|J_{ij}|.$$

At canonical parameters, relabeling digits leaves equality interactions and conditional probabilities invariant, so unaugmented $L(J)$ already equals the complete digit average. **The coefficient is $9\lambda$, not $18\lambda$:** each unordered pair is stored once.

This requires unrestricted real $J$. Imposing $J\geq0$ during optimization needs a separate argument. It does not prove uniqueness, equivalence for a finite sample of permutations, exact finite-budget convergence, or correctness of any graph extracted from the coefficients. In particular, keeping only $J>0$ and replacing its values by one is a further inference transformation, not part of the equivalence theorem.

## Numerical gates and adaptive design

The [solver](canonical_pl.py) uses FP64 proximal gradient with soft thresholding, a quadratic-majorizer backtracking check, 256 updates and initial step 540. Its two source-only gates use four previously seen noisy N600 banks: [raw coefficient 1/30000](canonical_pl_gate.json) and [raw coefficient 1/6000](canonical_half_gate.json).

Checks compare manual and autograd gradients, finite differences, raw-energy PL, digit-permutation invariance, the origin count formula, and CPU/GPU FP64 coefficients and support. Both gates pass. The public [gate exports](../../../results/research/foureyes/canonical_study/) retain complete receipts and coefficients.

At 1/30000, largest-gap selection retains only 3, 813, 1 and 810 edges across the four fixtures, while positive support retains 919, 931, 929 and 929. At 1/6000, positive support retains 807, 808, 810 and 809. These source observations motivated the positive-support rule and stronger primary penalty **before fresh banks or inference outcomes**. This is adaptive source-only design, not an untouched first proposal.

The earlier raw-SGD weak-penalty gate failed three of four precision checks and remains [published separately](GATE_WEAK_PENALTY.md). It did not trigger a fresh raw-SGD experiment. The new canonical study changes parameterization, optimizer and graph selection jointly. Its [fresh outcome](REPORT-canonical-study.md) falsifies the zero-wrong-stop target despite small numerical stationarity residuals.
