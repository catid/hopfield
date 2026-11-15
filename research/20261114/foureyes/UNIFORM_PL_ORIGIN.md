# What the proximal coefficient means for the digit-averaged objective

An exact source-only diagnostic separates two questions that the finite training results do not answer: what minimizes the **uniformly digit-averaged regularized objective**, and what can be a fixed point of the implemented momentum/proximal update. These are different mathematical objects. This post hoc analysis adds no optimizer fit, target input or inference trajectory.

For 64 frozen H4s noisy training banks, zero parameters are the **unique global minimum** of the fully digit-averaged pseudolikelihood plus the nominal coefficient `λ=0.003/9=1/3000` times the sum of absolute parameter values. The paired clean boards have that same property in 60/64 banks; zero is not a minimum in the other four. In contrast, zero is **not a minimum in any of the 128 clean/noisy banks** when the coefficient is `λ/2`. The latter coefficient is relevant to a constant-step stationary equation of the implemented heavy-ball/proximal update, as derived below. Neither result proves convergence of the actual finite, stochastic, cosine-scheduled run.

Let `F` be the total number of free cells. For pair `(i,j)`, define integer sufficient statistics

\[
A_{ij}=\sum_r(f_{ri}+f_{rj}),\qquad
C_{ij}=\sum_r(f_{ri}+f_{rj})\,\mathbf1[y_{ri}=y_{rj}].
\]

At zero parameters, every conditional digit probability is `1/9`. Averaging over **all 9! simultaneous digit relabelings** gives

\[
g_{ij,aa}=\frac{9C_{ij}-A_{ij}}{81F},\qquad
g_{ij,ab}=-\frac{g_{ij,aa}}8\;(a\ne b),\qquad g_B=0.
\]

Equal source digits map uniformly to the nine diagonal digit pairs; unequal source digits map uniformly to the 72 off-diagonal pairs. Each individual digit remains uniform. These identities give the formula directly, without a Monte Carlo approximation or an assumption that the source cell pairs are independent. An exact rational fixture independently enumerates every S3 permutation and all 54 pair/digit coordinates for a four-cell example.

Write the averaged pseudolikelihood as `L(θ)`, where θ contains every raw pair-table and unary parameter. It is convex because each conditional term is a linear energy plus a log-sum-exp of linear energies. Its supporting hyperplane gives

\[
L(\theta)+\lambda\|\theta\|_1
\ge L(0)+g(0)^\top\theta+\lambda\|\theta\|_1
\ge \log9+(\lambda-\|g(0)\|_\infty)\|\theta\|_1.
\]

A strictly positive final coefficient proves that zero is the unique global minimum over all raw pair/unary parameters. A negative coefficient identifies a signed single-coordinate descent direction, proving zero is not a minimum. For every such case the public certificate supplies an explicit rational step and positive decrease bound using the one-coordinate Hessian bound `A/(4F)`. Equality alone would not prove uniqueness; no audited bank is on that boundary.

For the noisy banks, the maximum absolute zero gradient ranges from **0.000269257 to 0.000299325**, strictly below 1/3000 and above 1/6000. For the clean banks it ranges from **0.000316414 to 0.000344441**. All comparisons and descent certificates use integer counts and exact rational arithmetic, not rounded decimal thresholds.

The source implementation updates

\[
v_{t+1}=m v_t+\nabla L(\theta_t),\qquad
\theta_{t+1}=\operatorname{prox}_{\alpha\lambda\|\cdot\|_1}
(\theta_t-\alpha v_{t+1}),\quad m=\tfrac12.
\]

At a **deterministic constant-step fixed point**, the velocity would be `v=∇L/(1−m)`. The proximal stationary equation is therefore

\[
0\in\nabla L(\theta)+(1-m)\lambda\,\partial\|\theta\|_1.
\]

At zero this gives the threshold `(1−m)λ=λ/2`. This derivation does not assert that the finite stochastic schedule minimizes either objective, nor that adding momentum changes the stated experimental recipe. Consequently, the nominal-objective zero certificate must **not** be used as an explanation that the actual trained models converge to zero. The measured nonzero fits and search failures remain separate evidence.

This distinction is also why exact uniform digit averaging must not be confused with the particular 32 or 256 sampled permutations used during training. The theorem concerns the explicitly defined full group average. It does not establish a guarantee for those finite sampled losses or for the search decoder. The paper reproduction is unchanged.

The [source diagnostic](check_uniform_pl_origin.py) checks all 64 paired banks. A [separate exporter](export_uniform_pl_counts.py) independently reconstructs every weighted equality count by directional free-cell counting before releasing compact integer sufficient statistics. The [public-only verifier](verify_uniform_pl_origin.py) checks all **256 exact certificates**, including every pair-gradient bound and signed descent witness, using only Python and NumPy. A separate agent independently reviewed the gradient, convexity and momentum derivations; it did not validate the empirical bank counts.

[All certificates and provenance](../../../results/research/foureyes/uniform_pl_origin/summary.json) · [per-bank results](../../../results/research/foureyes/uniform_pl_origin/per_bank.csv) · [public sufficient counts](../../../results/research/foureyes/uniform_pl_origin/sufficient-counts.npz) · [independent count-export receipt](../../../results/research/foureyes/uniform_pl_origin/count-export.json)
