# H3e: fixed-recipe transfer to 12-Queens

The fixed primary hypothesis was **supported**. Adam at 250 updates versus SGD at 4,000, with certificate-first output selection: accuracy difference +0.073242 percentage points; coverage difference -0.926865 points.

All arms use the same 100,000 unique training inputs and all compatible labels. Evaluation uses 8,192 disjoint new n=12 inputs, 20 chains and three repeated search seeds, with no n=12 tuning. This is a generated-domain study rather than an official benchmark reproduction.

[Full report and protocol](../../../../research/20261114/loki/h3e/REPORT.md), [summary and hashes](summary.json), [per-seed counts](per_seed.csv), [per-clue counts](per_clue_seed.csv).
