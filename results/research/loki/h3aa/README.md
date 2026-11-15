# H3aa tracked evidence

The frozen proposal-effect hypothesis is **supported**. With byte-identical permutation starts, permutation proposals reach 87.231445% certified success versus 20.874023% for arbitrary-cell proposals: **+66.357422 pp**, one-sided 95% endpoints **[65.429281, 67.277018]**, zero wrong stops. Changing initialization alone gives a descriptive +0.284831 pp with endpoints spanning zero.

All arms use the same frozen BP AdamW energy and source temperature. All nine searches are fresh, on 4,096 new inputs with 20 chains, 20,000-proposal caps and seeds 0/1/2. All 737,280 saved states and shared initial hashes are audited. Permutation proposals cost more per move and total measured search time; this is a quality result, not a speedup.

[Full report](../../../../research/20261114/loki/h3aa/REPORT.md), [frozen protocol](../../../../research/20261114/loki/h3aa/protocol.json), [summary](summary.json), [per-seed counts](per_seed.csv), [per-input capacities](per-input-capacity.npz), [figure](initialization-triangle.png), [SVG](initialization-triangle.svg), [plot provenance](plot-provenance.json).

The figure reads only tracked summary data and uses deterministic SVG metadata/hash salt. Rebuild with `artifacts/plotting-venv/bin/python research/20261114/loki/h3aa/plot_public.py` from the repository root.
