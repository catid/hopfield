# Exact lower bounds for the fixed-pool equality cover

The selector reads all 1,000 complete labeled boards in each pool. These bounds concern how many of those boards must be retained to preserve their full equality coverage. They are not bounds on learning from initially unseen labels.

Every valid Sudoku board contains nine occurrences of each of nine digits. Its equal-digit pair set therefore has exactly $9\binom92=324$ unordered pairs. If a pool observes $U$ distinct equal-digit pairs, any subset preserving that union has at least $\lceil U/324\rceil$ boards. In these pools $U=2430$, giving the exact elementary lower bound eight.

The following stronger certificate is a **post-selection geometry audit**. Neither the greedy selection nor the MILP receives these geometric facts.

Let $B$ be the cell pairs in different row bands and different column stacks. Each cell has 36 such partners, so $|B|=81\cdot36/2=1458$.

Fix one digit in a valid board. It has three occurrences in each of three bands, giving $3\binom32=9$ equal-digit pairs within a band. It similarly has nine pairs within a stack. These sets cannot overlap: equal digits cannot share a box. Of its $\binom92=36$ pairs, exactly $36-9-9=18$ therefore belong to $B$. Across nine digits, every valid board covers exactly 162 pairs in $B$.

All 1,458 pairs in $B$ are observed in each full pool. Thus at least nine boards are needed even to cover $B$. If nine boards did cover it, their nine sets of size 162 would have to be pairwise disjoint, because $9\cdot162=1458$.

The independent audit checks every unordered pair of candidate boards in each pool, using exact Boolean intersections. The minimum intersection sizes in $B$ are respectively **1, 1, 2, 2, 2, 3** for seeds zero through five. Hence no two candidate sets are disjoint, and no nine-board subset can cover $B$. Every exact cover drawn from each fixed pool therefore requires **at least ten boards**.

This is an exact finite-pool certificate, independent of the floating-point MILP solver. It checks 499,500 candidate pairs per pool, or 2,997,000 in total. It does not rule out a smaller construction using other Sudoku boards outside these pools, and it does not close the gap to the 26–27-board feasible covers found here.
