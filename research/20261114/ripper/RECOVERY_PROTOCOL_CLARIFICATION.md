# H1w pre-acquisition clarification

The frozen protocol's `seeds` field lists all 64 bank seeds, 0 through 63. Its inherited prose `rng` field mistakenly says “s in 0..31.” The intended and implemented range is 0 through 63, with the same stated formulas `default_rng(20261125+s)` for tie permutations and `default_rng(20261225+s)` for shadow permutations. No policy, rate, target, sample count, source draw, or stopping rule changes.

This clarification was written while input preparation was running, before acquisition or graph outcomes existed. The original protocol remains unchanged. Exact inference deduplication requires the same support, search seed, cohort, and complete settings.
