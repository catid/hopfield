# A global binary-objective certificate for the selected maze

The predeclared audit uses exactly the previously selected 90×90 maze with 352,212 shortest answers and the three frozen models. Walls are fixed off, the supplied start and goal are fixed on, and the archived hidden readout is fixed. The claim below concerns **every assignment of the free binary answer cells**. It does not concern continuous probabilities, a different hidden solution, another graph, or a population accuracy rate.

The [pinned answer-energy source](https://github.com/Prefrontal-AI/CRAM/blob/62bbe830c7d6f87f2ba18f30a28039258b986454/cram/cram_maze_hard_graph.py) is quadratic once the readout is fixed. Relative to the board containing only the two endpoints, it is

\[
\Delta E(x)=\sum_i h_i x_i+\sum_{i<j}J_{ij}x_ix_j,\qquad x_i\in\{0,1\}.
\]

For a Maze-Unique pair table, the free–free quadratic coefficient is `T11 − T10 − T01 + T00`. Its unary terms are `T10 − T00` at the first cell and `T01 − T00` at the second. If the other cell is clamped to `c`, the unary increment is the direct table difference `T1c − T0c` or `Tc1 − Tc0`. Out-of-grid cells contribute no pair. The open-cell bias contributes `b0 − b1`.

Each directed nonself gated link contributes `−κ_d a_id` to its free–free pair, or to a free cell's unary coefficient when the other endpoint is clamped on. A link touching a clamped wall contributes zero. The self link contributes `−κ_0 a_i0` to the unary coefficient because `x_i² = x_i` at binary values. Both directions of a nonself link are included. Binary entropy is the same constant at zero and one, including the source's epsilon convention, and cancels in the difference.

All learned answer parameters and archived attention entries are float32 numbers. The certificate interprets those stored values as exact dyadic constants. Construction uses rational arithmetic and retains a common power-of-two denominator, every unary and pair coefficient, and separate bias, pair-table, self-link and clamped-link unary contributions. The independent verifier reconstructs these quantities from the public raw constants using a different, per-node calculation.

For a negative pair coefficient `Jij = −wij`, choose `aij ∈ [0,wij]`. Binary variables satisfy

\[
-w_{ij}x_ix_j\;\ge\;-a_{ij}x_i-(w_{ij}-a_{ij})x_j.
\]

Positive pair terms are nonnegative and may be dropped. Allocate each negative edge's weight to its endpoints and define

\[
s_i=h_i-\sum_{e\ni i}\operatorname{allocation}_{e\to i}.
\]

Then `ΔE(x) ≥ Σ_i s_i x_i`. If every `si > 0`, every nonempty free-cell assignment has greater energy than the endpoint-only board, making that board the **unique global binary minimum under the stated clamps and readout**. This proves more than stability to one-bit flips or comparison with shortest paths.

The primary, equal half-allocation test fails for all three models. Its most negative slacks are approximately −10.7; that failure supplies no global-optimality conclusion. The predeclared secondary LP chooses asymmetric allocations to maximize the smallest slack. Its floating-point allocations are clipped to each exact edge interval, converted to exact rational numbers, and checked independently. The resulting minimum exact slacks are about 4.35, so they provide the certificate.

For completeness, a nonnegative vector `y` with `Σ_i y_i = 1` supplies the allocation-LP dual upper bound

\[
U(y)=\sum_i y_i h_i-\sum_{i<j:J_{ij}<0}w_{ij}\min(y_i,y_j).
\]

The published dual is normalized and checked with exact rational arithmetic. Its upper bound differs from the primal minimum slack by less than 6×10⁻¹¹ in each model. The dual certifies the tightness of this allocation relaxation; the primal inequality alone proves the binary global minimum.

The source correspondence is also checked numerically against the repository implementation in float64: all unary coefficients, every pair coefficient using nine separated grid-color probes, and 256 fixed random binary boards per model. These checks have errors below the independently reconstructed conservative allowance, about 0.000183. The allowance is `4 γ_m (A + n_free/10¹¹) + 10⁻⁸`, with `γ_m = m·2⁻⁵³/(1−m·2⁻⁵³)`, `m = 256HW + 1000`, and `A` an absolute sum bound on all pair, bias and gated-link terms. The operation count exceeds the source's per-board float64 multiply/add count; binary entropy is constant. Exact coefficient and allocation inequalities themselves have no rounding error. This is not a guarantee about every possible float32 summation order.

Run the [standalone verifier](verify_global_bound.py) from a public checkout:

```bash
python research/20261114/work/verify_global_bound.py
```

It reads only `results/research/work/global-bound/` and requires Python plus NumPy. It needs no checkpoints, private artifacts, upstream checkout, Torch, SciPy or solver. The public certificate includes the graph, fixed readouts, answer tensors, exact coefficients, allocation, dual, source/checkpoint hashes, original half-allocation failures and numerical comparison receipts. The source reconstruction script is [exact_binary_bound.py](exact_binary_bound.py); the secondary optimizer is [allocate_binary_bound.py](allocate_binary_bound.py).
