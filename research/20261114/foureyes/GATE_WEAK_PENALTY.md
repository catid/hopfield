# A tenfold weaker raw PL penalty failed its numerical gate

**The proposed raw-SGD recipe failed three of four source-only precision checks.** No fresh training-bank experiment or target search was launched. This preserves a failed prospective gate, rather than reporting a fresh accuracy result.

The motivation was the [exact origin analysis](UNIFORM_PL_ORIGIN.md): the nominal regularized, fully digit-averaged objective has zero as its unique optimum in all 64 prior noisy banks. Reducing the raw entrywise penalty from 1/3,000 to **1/30,000** places it below the origin-gradient maximum in those banks. This does not imply that the paper's finite momentum/cosine trajectory minimizes that objective.

The proposed recipe retained 256 updates, initial step size 540, momentum 0.5, the cosine schedule and the fixed largest-gap graph transformation. Before any new scientific fit, four previously used N600 noisy fixtures were reserved for CPU float32, GPU float32 and GPU float64 comparisons, with identical digit-relabeling draws. The gate required identical selected supports and coefficient disagreement below one quarter of the smallest coefficient-to-threshold separation.

| Prior fixture | GPU float32 edges | GPU float64 edges | CPU float32 edges | Gate |
|---|---:|---:|---:|---|
| 1 | 3 | 3 | 3 | Failed coefficient separation |
| 2 | 2 | 10 | 1 | Failed support and coefficient agreement |
| 3 | 1 | 1 | 1 | Passed |
| 4 | 1 | 1 | 1 | Failed coefficient separation |

The largest observed coefficient disagreement was 0.050816. All four cases and their unsuccessful checks are retained. The four-worker gate completed in 10.276 seconds; no fixture was replaced and no threshold or step size was tuned within the study. The inherited protocol fields describe the parent experiment; only the explicitly declared source-only gate was executed.

These observations establish failure of this recipe's declared numerical checks, not a general impossibility result for weaker regularization. A subsequent symmetry-constrained convex formulation and backtracking optimizer constitute a separate intervention and must be tested separately.

[Frozen gate protocol](weak_penalty_gate.json) · [Bounded gate runner](weak_penalty_gate.py) · [Inherited precision comparison](gate_long_matched.py) · [All measurements and execution receipt](../../../results/research/foureyes/weak_penalty_gate/gate.json)
