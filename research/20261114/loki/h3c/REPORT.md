# H3c: fixed training prefixes

**Frozen primary hypothesis: supported.** No-decay Adam after 1,000 updates was compared with preconditioned SGD after 4,000 updates, on all 6,513 unique n=10 official test inputs, with 20 chains and three paired search seeds. Both retain the original 4,000-update cosine schedule. The fixed 1,000-update checkpoint was specified before any prefix search outcome; no checkpoint was selected from this curve.

| Method | Updates | N20 accuracy (%) | Coverage (%) | Mean moves / chain | Wrong stops | Capped chains |
|---|---:|---:|---:|---:|---:|---:|
| BP AdamW, decay 0.01 (cached) | 4,000 | 99.605916 | 92.283227 | 5021.70 | 0 | 22,262 |
| BP Adam, no decay | 250 | 99.984646 | 92.629171 | 2757.00 | 0 | 6,513 |
| BP Adam, no decay | 500 | 99.984646 | 92.746345 | 2802.76 | 0 | 6,334 |
| BP Adam, no decay | 1,000 | 99.984646 | 93.112748 | 2989.04 | 0 | 7,180 |
| BP Adam, no decay | 2,000 | 100.000000 | 93.637243 | 3211.37 | 0 | 7,731 |
| BP Adam, no decay | 4,000 | 100.000000 | 93.559127 | 3313.01 | 0 | 7,806 |
| SGD + geometry preconditioner | 250 | 99.595680 | 91.870327 | 2552.14 | 0 | 5,757 |
| SGD + geometry preconditioner | 500 | 99.738984 | 92.381803 | 2628.68 | 0 | 5,494 |
| SGD + geometry preconditioner | 1,000 | 99.877169 | 92.854220 | 2708.93 | 0 | 5,528 |
| SGD + geometry preconditioner | 2,000 | 99.984646 | 93.605624 | 2815.23 | 0 | 5,464 |
| SGD + geometry preconditioner | 4,000 | 99.984646 | 93.585165 | 2860.85 | 0 | 5,690 |

Primary accuracy difference: +0.000000 percentage points; one-sided 95% bootstrap endpoints [0.000000, 0.000000]. The exact upper bound on the fraction of inputs with any paired regression is 0.045986% (0 observed inputs). The noninferiority margin is 0.1 point; search repeats are clustered inside each input.

Median complete in-memory fit: Adam at 1,000 updates 2.031168 seconds; SGD at 4,000 updates 8.076013 seconds. Ratio 0.251506; the frozen target was at most 0.5. All three timings and warmups appear in the JSON. Both methods were timed on Loki GPU0 after one warmup each; other campaign work waited until timing finished.

Complete in-memory fitting: preloaded host arrays and fitting indices, then GPU transfer, task construction, all updates, training-derived unit/tau/delta, and CPU inference-ready model. Excludes dataset disk I/O, Python import, CUDA context, checkpoint disk writes. Eager PyTorch has no torch.compile phase. No intermediate snapshots in timed fits.

The two newly reconstructed 4,000-update trajectories exactly match all cached learned tensors and calibration scalars. Their previous searches are reused without rerunning. Other checkpoints are prefixes of these same trajectories, each with its own training-derived temperature and stopping statistics. Earlier snapshots add no state update and were excluded from timed prefix fits.

This is an adaptive research follow-up on a previously used official test set. It provides evidence conditional on the frozen settings and cohort, not untouched external confirmation or broad optimizer superiority. There is one deterministic fitted trajectory per optimizer; the three seeds vary search only. All secondary checkpoints and failures are reported.

Positive radial rescaling is an analytic inference invariant when energy, temperature unit, stopping threshold and tolerance scale together. A small CPU power-of-two fixture verified identical boards, stopping, moves, FLOPs and scaled energies. Keeping an absolute tolerance fixed changes the relative tolerance; this was not treated as a useful new GPU hypothesis.

[Frozen protocol](protocol.json), [runner](driver.py), [auditor](analyze.py), [tracked summary](../../../../results/research/loki/h3c/summary.json), [per-seed counts](../../../../results/research/loki/h3c/per_seed.csv).
