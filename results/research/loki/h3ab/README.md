# H3ab tracked evidence

The matched-compute and quality hypotheses are **supported**. Source-selected 12,000 permutation proposals achieve 88.769531% certified success versus 21.061198% for 20,000 arbitrary proposals: **+67.708333 pp**, one-sided 95% endpoints **[66.788330, 68.563232]**, zero wrong stops. Fresh median solve times are 21.311479 versus 21.859988 seconds; ratio **0.974908** passes the ±10% gate without retuning.

All weights, source temperature endpoints and saved permutation starts are shared. Cap-specific annealing duration is part of each recipe. Source timing selection uses three streams on both GPUs; its point ratio 0.951364 passes the ±5% gate, while the source ratio uncertainty [0.949337, 0.952639] slightly crosses that boundary. The declared gate uses the point median. No target accuracy enters selection.

All 491,520 scored states are audited. Total GPU-worker cost **694.981729 seconds includes 561.060409 seconds of calibration**, warmups and fresh target workers. Calibration overhead is not omitted. The shared integrity guard hashes historical target artifact bytes; it does not decode or use target arrays for calibration, as documented in the access clarification.

[Full report](../../../../research/20261114/loki/h3ab/REPORT.md), [frozen protocol](../../../../research/20261114/loki/h3ab/protocol.json), [access clarification](../../../../research/20261114/loki/h3ab/CALIBRATION_ACCESS_NOTE.md), [summary](summary.json), [all calibration timings](calibration-selection.json), [per-seed results](per_seed.csv), [exact capacities](per-input-capacity.npz), [figure](matched-compute.png), [SVG](matched-compute.svg), [plot provenance](plot-provenance.json).

Rebuild the figure from tracked summary data with `artifacts/plotting-venv/bin/python research/20261114/loki/h3ab/plot_public.py`. SVG metadata and hash salt are deterministic.
