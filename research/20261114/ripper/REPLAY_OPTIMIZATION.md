# Faster independent acquisition replay

A bounded 200-row profile of H1ad's independent replay found repeated NPZ decompression as the main cost: 1.713 of 2.484 profiled seconds were in zlib decompression. Indexing `noise['uniforms'][row]` through a lazy `NpzFile` reloaded the complete array each row. The separate optimized replay loads each noise array once, marks it read-only, and verifies its bytes remain unchanged.

The optimized reference mask also keeps the accumulated pair-weight sum instead of recomputing it from the beginning at each greedy step. It preserves the original left-to-right floating-point operation order. This helper matched all 512 frozen synthetic cases, including ties and fully excluded states, and measured a 1.99× kernel speedup in that comparison.

On the complete old pilot seed 0, the optimized replay matched every scientific field in the preserved original audit: 13,320 masks, 43,156,800 log-LR values, noise draws, calibration endpoints, thresholds, first crossings, supports, and prefix checks. Only implementation hash, timestamp, elapsed time and optimization metadata differ. The original audit remains unchanged. No scientific fit or search was rerun.

Observed complete-replay wall times were 39.762 s for the original and 3.670 s for the optimized replay, a 10.83× ratio. The original ran alongside six CPU audits; the optimized replay ran during four GPU workers. These are observed wall times, not a controlled equal-load throughput measurement. H1ad's 32 fresh original replays had already completed before the optimization was ready, so they remain the study's required evidence. The optimized implementation is available for future independent audits.

- [Read-only optimized replay](audit_scaled_noise_fast_readonly.py)
- [Mask helper](fast_reference_mask.py) and [helper validation](validate_fast_reference_mask.py)
- [Bounded profiler](profile_scaled_replay.py)
- [Whole-audit comparison](validate_replay_optimization.py)
- [Public validation receipt](../../../results/research/ripper/replay_optimization.json)
