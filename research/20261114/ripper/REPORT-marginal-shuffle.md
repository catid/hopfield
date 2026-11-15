# H1bs: real curvature exceeds a null with exact empirical cell marginals

The preregistered joint target is **supported: 32/32 banks have strictly greater real curvature concentration than the empirical-marginal shuffle under both mask designs**, exceeding the required 29. All 64 individual comparisons are strict and resolved. Unlike the earlier IID-uniform null, this control preserves every observed cell's exact empirical digit histogram. The seven uniform-mask and six balanced-mask nonexact real graphs still pass, so the contrast does not certify graph correctness.

| Procedure and mask | Real R >shuffle R | Shuffle selected degrees | Shuffle exact reference graphs |
|---|---:|---|---:|
| BP AdamW: outside this source-only study; not run | — | — | — |
| Uniform masks | 32/32 | degree1:19; degree2:12; degree3:1 | 0/32 |
| Balanced masks | 32/32 | degree1:21; degree2:11 | 0/32 |

No shuffled arm selects degree 20. Uniform shuffled outputs include seven integral nonexact and 25 fractional graphs; balanced outputs include nine integral nonexact and 23 fractional graphs. Fixed-degree-20 and top-810 controls recover zero exact reference graphs. Every curvature maximum is unique. Fractional outputs remain valid objective evidence and are resolved failures only for descriptive graph recovery. Degree zero remains outside the selector's candidate set, so its low positive selected degrees do not implement abstention.

| Mask | Archived real R: minimum /median /maximum | Shuffled R: minimum /median /maximum |
|---|---|---|
| Uniform | 0.191026 /0.229846 /0.259527 | 0.070704 /0.092775 /0.133975 |
| Balanced | 0.190281 /0.230395 /0.268114 | 0.079733 /0.096620 /0.129130 |

These ranges are descriptive, without a fitted threshold or confidence calibration. The real banks and their earlier statistics were already examined; only this conditional shuffle intervention was prospectively frozen before its real histograms and new shuffle outcomes were calculated.

All 32 H1bq banks and both archived 72-cell mask arms are retained, including every real graph failure. For each bank, arm and cell, its labels on observed rows are converted to a canonical sorted multiset. A separate frozen arm stream draws an independent permutation for each cell and assigns those labels to ascending observed row positions. Permuting the canonical multiset has the same conditional law as uniformly permuting the archived observed labels directly, including when labels repeat. The implementation retains missing-value sentinels on unobserved rows.

This preserves the exact masks and **all 5,184 per-cell digit histograms** across the 64 arms. It randomizes cross-cell row associations but retains finite-population dependence within each column and does not preserve row-level digit counts. Each arm is shuffled independently; their values may differ on overlapping queried cells. There is no common latent shuffled board across the two arms, and coincident realized values do not justify deduplication.

The statistic is unchanged: with exact doubled LP objectives at degrees 0 through 40, `κ(d)=F(d+1)+F(d−1)−2F(d)` for `d=1,...,39`, and `R=max κ/Σκ`. All-zero curvature gives `R=0/1`; no observed arm has a zero denominator. Producer integer cross-products agree with independent rational comparisons. Strict ties fail. A bank succeeds only when both arms satisfy `R_real>R_shuffle`; any false comparison is a resolved joint failure, and otherwise an uncertified required grid degree yields an unresolved joint outcome. The endpoint uses lower `successes` and upper `successes+unresolved`, with threshold 29. Observed bounds are [32,32].

| Observation accounting | Cell-label instances |
|---|---:|
| New official rows, official queries and clean-oracle cells | **0** |
| Each shuffled arm's observed-entry assignments | 414,720 |
| New assignments across two independent shuffled arms | **829,440** |
| Shared synthetic assignments charged once | **0** |
| Historical real physical query union reused | 460,721 |

The synthetic assignments reuse acquired labels; they are not new official queries. The official noisy-label sequence remains **10,540,119**, and source ledgers remain **2,132,001 Ripper /2,368,426 combined IDs**. No new source selection, acquisition, noise fitting or puzzle input occurs. Historical overlapping real queries are counted once in their 460,721 union; arm-specific shuffled assignments are counted separately.

Before any histogram read, the frozen protocol pins **all 64 raw H1bq observation-file hashes**, along with the public real protocol, masks, models, receipts and summary. During charged execution, the independent raw audit reconstructs each histogram from observed labels and separately reconstructs the original pair counts, matching the original archives and models. All 64 histogram-provenance checks pass. Public files contain the derived histograms, shuffled arrays and seeds. Public replay verifies the conditional shuffle, histogram preservation, counts, costs and proofs; the correspondence between those histograms and original real labels requires the pinned raw archive and is explicitly labeled inherited raw-audit evidence in public verification. Original source/noise origin remains inherited from H1bq's complete audits.

Inference retains the common-degree prior, bounded degree grid, exact half-even Laplace rates, fresh positive-degree solves and analytic zero witnesses. No true degree, edge count, geometry or shuffle indicator enters the estimator. All **2,560 new shuffled LP proofs and 2,560 archived real proofs**, **15,360 altered-proof rejections**, **128 zero witnesses** and **64 queried-label substitution gates** pass. Shuffled intermediate fractional grid counts are 847 uniform and 900 balanced. No primal is snapped. Certificates establish optimality, not uniqueness or graph correctness. The real models are rechecked without refitting.

Execution completes **32 bank jobs plus summary in 34.643012 aggregate CPU-seconds and 4.499413 wall seconds**, within 900 CPU-second /180-second limits. A live sample at 1.15 seconds shows eight jobs complete and eight active workers near 100% CPU each. Average useful CPU consumption is about 7.7 cores. Four Ripper GPUs each show 0% utilization and 2 MiB allocation because the workload consists of CPU histograms, counts and LPs. Final progress has 33 completed commands and no owned active worker.

Synthetic preflight verifies independent histogram and permutation implementations, exact histogram/mask preservation, canonical input-order invariance and exhaustive repeated-label permutation examples. It also checks separate-arm accounting, pair counts, full queried-label substitutions while preserving missing sentinels, exact ratios beyond binary64 precision, zero/tie/unresolved rules and joint endpoint bounds. The lead agent reviews the final code and gate before freeze. Frozen science remains unchanged.

Local, Ripper and public-only checkout verification pass. Deterministic PNG/SVG/provenance rerenders are byte-identical and the figure is visually inspected. The result strengthens the declared association-sensitive signal contrast; it is not a fresh-source confirmation or a calibrated graph-confidence result.

[Frozen protocol](protocol-marginal-shuffle.json) · [Study](marginal_shuffle_study.py) · [Independent audit](audit_marginal_shuffle.py) · [Portable verifier](verify_marginal_shuffle_public.py) · [Results](../../../results/research/ripper/marginal_shuffle/summary.json) · [Figure](../../../assets/figures/marginal_shuffle.png)
