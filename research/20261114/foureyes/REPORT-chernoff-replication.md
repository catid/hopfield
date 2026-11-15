# Noise-law thresholds protect true constraints but fail exact recovery at N600

**The frozen ≥99% exact-recovery hypothesis is falsified.** A threshold derived from the correct corruption law recovers **453/512 graphs (88.4766%)** at 600 training boards; its one-sided exact 95% upper bound is **90.7255%**, below 99%. It preserves every true edge in all 512 draws, but retains **60 extra edges**. At 1,000 boards it recovers every graph; that endpoint is descriptive and does not replace the failed primary.

| N | Estimator | Exact / 512 | All true edges retained / 512 | Total missing edges | Total extra edges |
|---|---|---:|---:|---:|---:|
| 600 | Original gap | 494 | 509 | 3 | 21,865 |
| 600 | Gap restricted below median | 503 | 509 | 3 | 6 |
| 600 | Chernoff, assumed 3 changes, actual 9 | 0 | 0 | 43,493 | 0 |
| 600 | Chernoff, correct 9-change law | 453 | 512 | 0 | 60 |
| 1,000 | Original gap | 511 | 512 | 0 | 1 |
| 1,000 | Gap restricted below median | 511 | 512 | 0 | 1 |
| 1,000 | Chernoff, assumed 3 changes, actual 9 | 0 | 0 | 146,309 | 0 |
| 1,000 | Chernoff, correct 9-change law | 512 | 512 | 0 | 0 |

These are threshold rules for one count estimator, not a new optimizer comparison. Earlier [matched optimizer controls](REPORT-raw-transform.md) remain separate. All differences outside the registered correct-law N600 recovery probability are descriptive.

## What is guaranteed, and what is tested?

The [estimator](noise_chernoff.py) receives corrupted digit labels, the original free-cell masks, and a supplied corruption law. For each row, exactly c free cells are selected uniformly without replacement; each chosen digit independently changes to a uniformly selected other digit. The actual experiment uses c=9 and preserves clues.

For a truly never-equal pair, let f_i and f_j indicate free cells, m be the number of free cells, a=c/m, and b=c(c−1)/(m(m−1)). Its row-wise equality probability after corruption is

$$p_{ij}=(f_i+f_j)a/8-f_if_j\,9b/64.$$

For fixed clean rows and masks, independent row corruption gives a Poisson-binomial count. The binomial-form Chernoff bound using its mean bounds the upper tail. For each pair choose the smallest retained integer threshold T for which the bound on **P(C≥T+1)** is at most **0.01/3,240**. The threshold depends only on the masks and supplied law, not on noisy counts or the clean labels.

A union bound then controls the probability of dropping **any** truly never-equal pair by at most 1%, conditional on the stated source/mask/noise assumptions. It does not require pairwise independence. This protects true constraints; it does **not** control extra constraints, exact graph recovery, misspecified corruption, or arbitrary solver stopping certificates. The failed exact-recovery primary therefore does not contradict the mathematical deletion bound.

The incorrect assumed-c=3 control sharply violates the law used in calibration and deletes many true edges. It is outside the guarantee. Original masks here are fixed before corruption; the argument cannot simply condition on masks selected adaptively from earlier corruptions.

An [independent review](../ripper/review_noise_chernoff.py) verifies the formulas, tail indexing, threshold minimality and exact small Poisson-binomial tails in 28,416 checks. The largest apparent tail exceedance is 1.1×10⁻¹⁵ from floating-point rounding. The broader [corruption-law derivation](../ripper/NOISE_PAIR_BOUND_REVIEW.md) distinguishes this Chernoff bound from any claim of exact binomial-CDF domination.

## Prospective design and audit

[Protocol](chernoff_replication.json), SHA-256 `0ddf51ff76fa651ed36a13f6a2fd4056a620797e209683d8f3d7a396f0a065b3`, froze the rule, 512 independent 1,000-row source samples, nested N600 prefix, controls, fresh source/noise RNG streams and single primary before outcomes. Cross-bank and earlier-study row overlaps are allowed under independent sampling from the same eligible official corpus. These are new draws, not globally disjoint inputs or a new distribution.

The first six previously inspected H4n banks were used for exploratory feasibility. All six passed correct-law recovery at N600/N1000; that feasibility result is retained by hash and is not pooled. Its small sample did not predict the prospective N600 failure.

The [independent auditor](audit_chernoff_replication.py) replays sampling, rereads selected source rows from the CSV, validates source solutions/clues, reconstructs every corruption and independently counts raw digit equalities. It rebuilds all **1,024 count arrays / 4,096 thresholded models**. For the new thresholds it explicitly sums row-wise null probabilities, independently of the producer's faster matrix formula, then checks that every threshold satisfies the bound and every preceding threshold fails it. All masks, counts, thresholds and source/output hashes match. The audit took 17.19 seconds. [Independent SciPy verification](../../../results/research/foureyes/chernoff_replication/statistics-verification.json) checks all 16 separate one-sided exact recovery endpoints.

Parsing and validating the corpus dominated input preparation, so the producer also creates a source-hashed cache of its complete parsed arrays. Future cache loads verify both array hashes. A [cache check](../../../results/research/foureyes/chernoff_replication/cache-validation.json) compares cached data with the retained source-bank inputs; the independent scientific audit still rereads the original CSV. These preparation measurements are not a training-speed comparison.

[Public summary and provenance](../../../results/research/foureyes/chernoff_replication/summary.json) · [Per-bank outcomes and support hashes](../../../results/research/foureyes/chernoff_replication/per_bank.csv) · [Recovery table](../../../results/research/foureyes/chernoff_replication/recovery.csv) · [Producer](chernoff_replication.py) · [Auditor](audit_chernoff_replication.py) · [Statistical verifier](verify_chernoff_statistics.py)

No inference was run in this construction study. Notably, exact recovery ranks the original gap above correct-law thresholds at N600, while the gap's few failures introduce far more extra constraints. A separate solver experiment is needed to learn whether that structural ranking predicts average solve accuracy; neither ranking is treated as an inference result here.
