# Exact conditional coverage under independent uniform clue masks

Fix any sequence of valid complete Sudoku boards $x_1,\ldots,x_N$ and fixed clue counts $k_t\in\{0,\ldots,81\}$. These boards need not be independent or uniformly distributed. For each row $t$, independently choose a uniformly random subset $M_t$ of exactly $k_t$ cells. Reveal only the values at those cells.

The count estimator retains an unordered cell pair $e=\{i,j\}$ unless some training row reveals both cells with equal values. It does not receive the Sudoku peer graph. The graph is used below only to describe recovery after the estimator is fixed.

Let $A_{t,e}=1\{x_{t,i}=x_{t,j}\}$, and let $R_e$ indicate that pair $e$ survives. Then

$$
p_t=\Pr(i,j\in M_t)=\frac{\binom{79}{k_t-2}}{\binom{81}{k_t}}
=\frac{k_t(k_t-1)}{81\cdot80},
$$

with $p_t=0$ for $k_t<2$. Independence of masks across training rows gives the exact conditional survival probability

$$
\Pr(R_e=1\mid x_{1:N},k_{1:N})
=\prod_{t:A_{t,e}=1}(1-p_t).
$$

Every true peer pair has $A_{t,e}=0$ in every valid complete board, so all 810 true edges survive with probability one. Write $F_N$ for the number of retained nonpeer pairs, out of the 2,430 possible nonpeers. Linearity of expectation gives

$$
\mu_N=\mathbb E[F_N\mid x_{1:N},k_{1:N}]
=\sum_{e\text{ nonpeer}}\prod_{t:A_{t,e}=1}(1-p_t).
$$

Markov's inequality, or equivalently the union bound over retained nonpeers, gives

$$
\Pr(\text{graph recovery fails}\mid x_{1:N},k_{1:N})
=\Pr(F_N\ge1\mid x_{1:N},k_{1:N})
\le\min(1,\mu_N).
$$

No independence between different cell pairs is assumed. Their survival indicators generally are dependent because a row reveals one shared subset of cells.

For several fixed banks, the probability that any bank fails is at most the sum of their individual $\mu_N$ values, capped at one. This last bound does not require independence between banks and remains valid if their fixed complete boards overlap.

This is a conditional theorem for independently uniform masks, not a theorem about the official puzzle generator's masks. It does not imply a universal sample threshold: if a fixed bank never assigns equal digits to some nonpeer pair, that pair survives every mask realization. It also does not imply that every finite Monte Carlo failure fraction must lie below its probability bound. A deterministic pseudorandom stream is an implementation of the sampling experiment, not a source of mathematical randomness for the proof.

The formula permits a direct control for the clue-only experiment: keep each complete board and its original number of given cells, replace only the cell mask, and compare observed extra-edge counts with the conditional expectation. Any graph-based evaluation remains an audit after support construction, never an input to the estimator.
