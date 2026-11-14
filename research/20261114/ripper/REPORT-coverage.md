# H1m: fresh-stream validation of the augmentation coverage theorem

The frozen validation ran **1,040,384 fresh spatial variants** across 32 fixed valid base boards and 8,192 PRNG streams. There were **0 statistical contradictions** in the prespecified 160-test Bonferroni family at level 0.05. The minimum adjusted p-value was **0.330554**. This empirical check is distinct from the exact ideal-sampling theorem.

| Spatial variants | Predicted extra edges | Measured mean extra edges | Exact graphs / 8,192 |
|---:|---:|---:|---:|
| 0 | 2106 | 2106 | 0 |
| 1 | 1827 | 1826.95 | 0 |
| 3 | 1378.97 | 1379.13 | 0 |
| 7 | 794.303 | 794.29 | 0 |
| 15 | 274.045 | 274.218 | 0 |
| 31 | 36.4852 | 36.489 | 0 |
| 63 | 0.784559 | 0.786743 | 3765 |
| 99 | 0.0111934 | 0.0107422 | 8105 |
| 127 | 0.000413342 | 0.000854492 | 8185 |

For each base, two representative-pair equality rates were tested against the predicted probabilities 1/6 and 1/9. Three per-base graph-failure rates were tested against the union bounds at 63, 99 and 127 variants. All 64+96 tests share a Bonferroni family; the CSV reports raw and adjusted p-values and simultaneous exact-binomial intervals.

The theorem uses independent uniform group draws. The experiment uses deterministic NumPy PRNG streams and the unchanged pinned source function. No statistical contradiction is evidence of implementation consistency, not proof of ideal independence. The mean-curve standard errors are descriptive; no extra unregistered expectation tests are asserted.

[Theorem and source assumptions](AUGMENTATION_COVERAGE_THEOREM.md) · [Frozen validation protocol](protocol-coverage.json) · [Sampling code](coverage_sample.py) · [Audit](audit_coverage.py) · [Tracked results](../../../results/research/ripper/coverage.json)
