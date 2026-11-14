# H3f: training-count and complete-bank controls

Frozen no-optimizer exclusion-energy noninferiority result: **supported**. All14,200 n=12 complete solution patterns occur in training. Evaluation holds out partial inputs, not complete patterns; the compatible-training-bank baseline makes that limitation explicit.

The inferred exclusion support is combined with a declared fixed energy-design prior and supplied offset geometry. It is not learned-weight recovery.

[Full report and protocol](../../../../research/20261114/loki/h3f/REPORT.md), [summary and hashes](summary.json), [per-seed counts](per_seed.csv).

[Conditional certificate soundness proof](../../../../research/20261114/loki/h3f/CERTIFICATE_PROOF.md) explains why this fixed energy cannot certify an invalid board when its inferred graph equals the true attack graph and the input has at most three valid givens. It does not guarantee convergence.
