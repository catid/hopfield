# Coverage from the pinned Sudoku augmentation group

The following is an exact statement under **independent, uniformly sampled group elements**. The fresh pseudorandom experiment is a separate implementation check; it does not turn a deterministic PRNG into an independent random source.

Fix any valid solved 9×9 Sudoku board. Include that original board and apply the pinned spatial augmentation independently V times. Connect two cells when they have never held the same digit in these V+1 boards. If X_V is the number of retained nonpeer edges, then

\[
\mathbb E[X_V]=810(5/6)^V+1296(8/9)^V,
\qquad
\Pr(X_V>0)\le \min\{1,\mathbb E[X_V]\}.
\]

Every true Sudoku peer edge is retained, so X_V=0 is exactly graph recovery. No optimizer is involved. The probability bound does not assume that different cell-pair coverage events are independent.

## The sampled group

The [pinned `shuffle_sudoku` implementation](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/c01103738605ba39d1430519b1ee0c62f4c707f8/dataset/build_sudoku_dataset.py) draws a uniform permutation of the three bands, an independent uniform permutation of the three rows within each band, and the corresponding four independent column/stack permutations. It independently chooses whether to transpose. Digit relabeling is independent and does not affect equality events.

The row group has size 6⁴, as does the column group. Their product and the transpose form a group of size 2·6⁸ = 3,359,232. The band permutation and its three within-band permutations give a unique row-group element; likewise for columns. Transposition is outside the row/column product group, so the source's sampling procedure is uniform over this spatial group under the ideal independent-uniform-draw assumption.

## Two nonpeer orbits

An unordered pair of nonpeer cells belongs to one of two orbits:

| Orbit | Description | Number of pairs |
|---|---|---:|
| A | Same band or same stack, but neither a row, column nor box peer | 972 |
| B | Different bands and different stacks | 1,458 |

Each cell has 12 nonpeer partners in its band and 12 in its stack, giving 81·24/2 = 972 pairs in A. Each cell has 6·6 = 36 partners in different bands and stacks, giving 81·36/2 = 1,458 pairs in B.

The group acts transitively on each orbit. For a same-band pair, band permutations map its band to any target band, a within-band row permutation maps its two distinct rows to any target distinct rows, and stack plus within-stack column permutations map its two distinct stacks and their columns to any targets. Transposition connects this class to same-stack pairs. For a B pair, the two distinct bands and the two distinct stacks can be mapped independently, with independent within-band and within-stack permutations mapping the remaining coordinates. These constructions cover ordered pairs and therefore unordered pairs as well.

A uniform group element maps any fixed pair uniformly over its orbit: all target pairs have equal-sized stabilizer cosets. This is the only group-probability fact required below.

## Equality counts in every valid solution

For each digit, its three occurrences in a band occupy distinct rows and distinct stacks. They contribute three equal-digit pairs per band. Across three bands and nine digits, this gives 81 same-band equal-digit pairs. The stack argument contributes another 81, disjoint from them because equal digits never share a box. Thus A contains exactly 162 equal-digit pairs.

There are nine occurrences of each of nine digits, giving 9·C(9,2)=324 equal-digit pairs in total. All are nonpeers. The remaining 162 lie in B. Hence a fixed pair has equality probability

\[
q_A=162/972=1/6,\qquad q_B=162/1458=1/9
\]

after a uniform spatial augmentation, regardless of the chosen valid base solution.

The original board has already exposed 162 pairs in each orbit. Initially unexposed counts are therefore 972−162=810 and 1,458−162=1,296. An initially unexposed pair in orbit A remains unexposed after V independent draws with probability (5/6)^V; for B the probability is (8/9)^V. Summing these probabilities proves the expectation. A union bound, equivalently Markov's inequality for the nonnegative integer X_V, proves the failure bound.

## What the bound says

For one fixed valid base solution, V=100 gives an upper failure bound of approximately 0.009949, so ideal recovery probability is at least 99.0051%. With six fixed base solutions, a further union bound gives total failure probability at most six times the expression above; V=116 gives a bound of approximately 0.009062. Independence between the six groups is unnecessary for that union bound, although the within-group draws must meet the theorem's sampling assumption.

At V=63 the expected extra-edge count is approximately 0.784559. At V=99 it is approximately 0.0111934. These are exact-formula predictions, not parameters fitted to the stored-variant observations. The preceding six-group experiment motivated checking this mechanism; the fresh-stream validation protocol freezes its bases, PRNG streams, predictions and tests before the new outcomes.

The implementation validation uses 32 fixed base boards, 256 fresh streams per base, and 127 variants per stream. It checks two representative orbit probabilities for each base and three coverage upper bounds per base. All 160 tests form one Bonferroni family at level 0.05. The published binomial tests and intervals are diagnostics under the ideal draw model. Absence of a statistical contradiction is not a proof of PRNG independence or an external-data generalization claim.

The theorem explains recovery from **task-specific spatial augmentation**. It does not establish learning from one unaugmented example. Digit permutations alone cannot expose a new equality event between fixed cell positions.

[Frozen fresh-stream protocol](protocol-coverage.json) · [Sampling code](coverage_sample.py) · [Earlier stored-prefix audit](REPORT-onebase.md)
