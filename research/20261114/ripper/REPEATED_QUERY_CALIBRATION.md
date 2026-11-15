# Learning a noise upper bound from repeated noisy queries

Each calibration pair asks for the same clean cell twice, with independent replacement noise at the same fixed rate eta used later in training. A replacement selects uniformly from the other eight digits. The pair disagrees if exactly one label changes, or if both change to different digits. Thus d=2eta(1−eta)+(7/8)eta²=2eta−(9/8)eta².

On 0<=eta<=8/9, d is monotone, with maximum8/9. Its stable inverse is eta=8d/(8+sqrt(64−72d)). For a fixed10,000 independent calibration pairs, a one-sided99.5% Clopper-Pearson upper bound for d yields a noise upper bound with failure probability at most.005. The estimator sees only disagreement indicators, not clean labels. We round eta upward to the fixed1e−6 rational grid. Exact integer binomial-tail comparisons verify the rounded endpoint and reject the previous grid point; an independent80-digit Decimal calculation is also retained. Floating-point root finding only proposes a grid index and cannot determine an unsafe final endpoint.

If the inverted or outward-rounded bound reaches8/9, the model uses a safe no-pruning fallback. It does not substitute a smaller convenient bound. A fully retained graph contains every true edge, though it may be useless for solving.

Calibration and training use globally disjoint source pools, independent noise streams, and fixed sample sizes. Conditional on a calibration transcript whose upper bound covers the true eta, the training masks remain predictable and the [conservative e-process argument](NOISE_UPPER_BOUND_EPROCESS.md) applies. Giving the edge test error budget.005 means crossing threshold3240/.005=648000. The total probability of deleting any true edge is at most.005+.005=.01 for one estimated-bound model. No independence between calibration failure and edge failure is needed for this union bound; independent training noise is needed for the conditional e-process argument.

Exact-known-eta and fixed2% controls use the same.005 edge budget. Their per-model deletion guarantee is.005 under the specified noise law. The experiment separately measures accuracy noninferiority and zero wrong certified stops in the estimated-bound and exact-known-eta arms. These measured conditions are stronger outcomes than deletion protection alone.

The calibration cost is20,000 noisy labels in addition to37,500 training labels per estimated-bound model:57,500 total. Known-parameter controls use37,500 training labels. Each of32 banks has one calibration/noise/tie realization and is evaluated on the same10,000 held-out puzzles. No calibration result changes the fixed sample count, q1, label budget, endpoint, margin, or stream definitions.

The guarantee is limited to the stated fixed independent replacement-noise family, with the same eta in calibration and training and eta in[0,8/9]. It does not cover arbitrary nonstationary or correlated corruption, false extra edges, solve accuracy, or arbitrary stopping certificates.
