# Independent review of fixed-count label-corruption bounds

Fix a clean row, its free-cell mask, and a corruption count $0\le c\le m$, where $m$ is the number of free cells. Choose exactly $c$ free cells uniformly without replacement. Each selected cell independently replaces its original digit by one uniformly chosen from the other eight digits.

For a free cell, $a=c/m$ is its selection probability. For two distinct free cells, $b=c(c-1)/(m(m-1))$ is their joint selection probability. The formulas below are correct:

| Pair status | Initially unequal: equality probability after corruption | Initially equal: equality probability after corruption |
|---|---:|---:|
| Both free | $a/4-9b/64$ | $1-2a+9b/8$ |
| Exactly one free | $a/8$ | $1-a$ |
| Neither free | $0$ | $1$ |

For an initially unequal pair with both cells free, exactly one cell is corrupted with total probability $2(a-b)$; its replacement matches the unchanged other digit with probability $1/8$. If both are corrupted, the two allowed replacement sets have seven common digits, so equality has probability $7/64$. The total is $2(a-b)/8+7b/64=a/4-9b/64$.

For an initially equal pair, equality survives when neither is selected, with probability $1-2a+b$, or both are selected and choose the same one of eight allowed values, with conditional probability $1/8$. This gives $1-2a+9b/8$. Exactly one corrupted cell cannot retain its old digit.

The enumeration script checks every applicable free-status/equality case for $m=0,\ldots,8$ and $c=0,\ldots,m$, using exact rational arithmetic. At $m=0$, only the neither-free case applies. At $m=1$, the both-free formula is inapplicable; do not evaluate its zero denominator. If an implementation caps $c$ by $m$, the formula must use the actual capped count in each row.

For a fixed pair, independent corruption across rows makes its equality count $C$ Poisson-binomial, with row-specific probabilities $p_t$. Define $\bar p=n^{-1}\sum_t p_t$. For every real $\lambda$,

$$
\log\mathbb E e^{\lambda C}
=\sum_t\log(1+p_t(e^\lambda-1))
\le n\log(1+\bar p(e^\lambda-1)),
$$

because the logarithm on the left is concave in $p_t$. Optimizing Chernoff's inequality gives $\exp[-nD(k/n\Vert\bar p)]$ for an upper tail $\Pr(C\ge k)$ when $k/n>\bar p$, or a lower tail $\Pr(C\le k)$ when $k/n<\bar p$. Outside the appropriate tail use the trivial bound one. Handle deterministic $\bar p=0$ or $1$ and out-of-range $k$ directly. This is the binomial-form **Chernoff bound**, not a blanket assertion that an exact binomial CDF bounds every Poisson-binomial tail.

If the estimator retains pairs with $C\le T$ for integer $T$, the failure events are $C_{\rm true}\ge T+1$ and $C_{\rm nonpeer}\le T$. This upper-tail $+1$ is necessary. A union bound sums those probabilities over fixed true edges and nonpeers; dependence between different pairs does not invalidate it.

For a fixed integer band $[L,U]$, the event $C_{\rm true}\le L$ for every true edge and $C_{\rm nonpeer}>U$ for every nonpeer guarantees that **every threshold in the band** recovers the graph. Its complement is bounded by upper tails at $L+1$ and lower tails at $U$. For real-valued endpoints, use $\lfloor L\rfloor+1$ and $\lfloor U\rfloor$ in those failure tails.

This uniform-in-band event permits data-dependent threshold selection that is guaranteed to stay in the fixed band. It does not prove that the unconstrained largest-gap rule stays in that band. In general,

$$
\Pr(\text{selected threshold fails})
\le\Pr(\text{band separation fails})+\Pr(T\notin[L,U]).
$$

Merely observing that a chosen threshold lands in the band does not turn the first unconditional bound into the same conditional bound given that event. A band selected using noisy observed counts needs its own selection adjustment or a simultaneous guarantee. A band defined solely from clean boards, fixed masks and the corruption model is fixed after conditioning on those inputs, even if computed later as a diagnostic.

The clean rows, masks and true graph may be privileged post-fit diagnostic inputs. They must remain outside the estimator. Compute each pair's row-specific probabilities before averaging; replacing varying $m_t$ by a global mean generally changes the formula. The argument assumes independent replacement draws for selected cells and independent row corruptions; it does not cover a shared digit permutation or a corruption budget imposed jointly across all rows.
