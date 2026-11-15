# Figures and their data

All README graphs show measured results. They are ordinary scientific plots, with downloadable SVG or PDF versions beside the PNGs.

| Figure | Public data | Renderer |
|---|---|---|
| Exact compilation of learned queen models versus GPU search | [Exact-compilation exports](../results/research/loki/h3ae/) | [plot_public.py](../research/20261114/loki/h3ae/plot_public.py) |
| Oracle diagnosis of confidence-weight budget boundary | [Oracle diagnostic](../results/research/ripper/confidence_oracle.json) | [plot_confidence_oracle.py](../scripts/plot_confidence_oracle.py) |
| Second maze hard-negative round | [Second-round exports](../results/research/work/second-negative-transfer-summary.json) | [plot_second_negative_transfer.py](../research/20261114/work/plot_second_negative_transfer.py) |
| Confidence weights and capped Sudoku returns | [Confidence exports](../results/research/ripper/confidence_weight.json) | [plot_confidence_weighting.py](../scripts/plot_confidence_weighting.py) |
| Confidence weighting at half training budget | [Budget exports](../results/research/ripper/confidence_budget.json) | [plot_confidence_budget.py](../scripts/plot_confidence_budget.py) |
| Queen search across four source-bank fits | [Four-bank exports](../results/research/loki/h3ac/) | [plot_public.py](../research/20261114/loki/h3ac/plot_public.py) |
| Queen search on two additional families | [New-family exports](../results/research/loki/h3ad/) | [plot_public.py](../research/20261114/loki/h3ad/plot_public.py) |
| Maze hard-negative constraint transfer | [Hard-negative exports](../results/research/work/hard-negative-transfer-summary.json) | [plot_hard_negative_transfer.py](../research/20261114/work/plot_hard_negative_transfer.py) |
| Convex digit-invariant PL and zero-stop soundness | [Canonical study](../results/research/foureyes/canonical_study/) | [plot_canonical_pl.py](../scripts/plot_canonical_pl.py) |
| No-optimizer Sudoku and budget curve | [Count and learned exports](../results/research/foureyes/) | [plot_count_results.py](../scripts/plot_count_results.py) |
| Sudoku counts from raw solutions | [Raw-count and PL exports](../results/research/ripper/) | [plot_raw_count_results.py](../scripts/plot_raw_count_results.py) |
| Sudoku matched transformation control | [Transformed-PL exports](../results/research/foureyes/raw_transform/) | [plot_raw_transform_results.py](../scripts/plot_raw_transform_results.py) |
| Sudoku fixed-restart control | [Restart exports](../results/research/foureyes/restart/) | [plot_restart_results.py](../scripts/plot_restart_results.py) |
| Reserved Sudoku test and sparse speed | [Ripper](../results/research/ripper/), [stopping rules](../results/research/foureyes/) | [plot_public_results.py](../scripts/plot_public_results.py) |
| Maze size transfer, branch depth and step size | [Work exports](../results/research/work/) | [public renderer](../research/20261114/work/) |
| N-Queens optimizer and update budget | [Loki exports](../results/research/loki/) | [Loki scripts](../research/20261114/loki/) |
| N-Queens unseen complete patterns | [Pattern holdout exports](../results/research/loki/h3g/) | [plot_public.py](../research/20261114/loki/h3g/plot_public.py) |
| Sudoku clue-mask controls | [Mask exports](../results/research/ripper/) | [plot_clue_mask_results.py](../scripts/plot_clue_mask_results.py) |
| Sudoku online acquisition confirmation | [Acquisition exports](../results/research/ripper/) | [plot_online_confirmation.py](../scripts/plot_online_confirmation.py) |
| Sudoku spike-mixture clue acquisition | [Acquisition exports](../results/research/ripper/spike.json) | [plot_spike_results.py](../scripts/plot_spike_results.py) |
| Sudoku fresh-bank noisy-label recovery | [Fresh-bank exports](../results/research/foureyes/noise_source_replication/) | [plot_noise_source_replication.py](../scripts/plot_noise_source_replication.py) |
| Sudoku noise replication and failure diagnosis | [Noise exports](../results/research/foureyes/noise_replication/) | [plot_noise_replication.py](../scripts/plot_noise_replication.py) |
| Maze selective fallback and complete timing | [Selective exports](../results/research/work/selective-fallback-summary.json) | [plot_selective_fallback.py](../research/20261114/work/plot_selective_fallback.py) |
| Maze objective ranking versus solving accuracy | [Matched training exports](../results/research/work/contrastive-training-summary.json) | [plot_contrastive.py](../research/20261114/work/plot_contrastive.py) |
| Sudoku noisy adaptive querying | [Noisy-query exports](../results/research/ripper/noisy.json) | [plot_noisy_acquisition.py](../scripts/plot_noisy_acquisition.py) |
| Sudoku conservative noise-bound accuracy cost | [Noise-bound exports](../results/research/ripper/noise_bound.json) | [plot_noise_bound_results.py](../scripts/plot_noise_bound_results.py) |
| Sudoku repeated-query noise calibration | [Calibration exports](../results/research/ripper/calibrated.json) | [plot_repeated_query_calibration.py](../scripts/plot_repeated_query_calibration.py) |
| Sudoku calibration versus training at equal query cost | [Equal-budget exports](../results/research/ripper/equal_budget.json) | [plot_equal_query_budget.py](../scripts/plot_equal_query_budget.py) |
| Sudoku calibration transferred to a higher noise rate | [Rate-transfer exports](../results/research/ripper/rate_transfer.json) | [plot_noise_rate_transfer.py](../scripts/plot_noise_rate_transfer.py) |
| Sudoku implied-edge confirmation | [Fresh confirmation](../results/research/foureyes/logical_completion_confirm/) | [plot_logical_completion.py](../scripts/plot_logical_completion.py) |
| Sudoku mixture evidence law | [Mixture exports](../results/research/ripper/mixture_law.json) | [plot_mixture_evidence_law.py](../scripts/plot_mixture_evidence_law.py) |
| Sudoku logical certificates and invalid zero-energy witness | [Certificates](../results/research/foureyes/logical_soundness/) | [plot_sudoku_soundness.py](../scripts/plot_sudoku_soundness.py) |
| Sudoku bounded evidence priorities | [Bounded-selector exports](../results/research/ripper/bounded_selector.json) | [plot_bounded_evidence_selector.py](../scripts/plot_bounded_evidence_selector.py) |
| Sudoku likelihood-ratio priority failure | [LR-selector exports](../results/research/ripper/lr_selector.json) | [plot_lr_evidence_selector.py](../scripts/plot_lr_evidence_selector.py) |
| Sudoku doubled training at 5% noise | [Scaled-noise exports](../results/research/ripper/scaled_noise.json) | [plot_doubled_noisy_training.py](../scripts/plot_doubled_noisy_training.py) |
| Sudoku noisy-label likelihood and count controls | [Noisy-PL exports](../results/research/foureyes/noisy_pl/) | [plot_noisy_pl_results.py](../scripts/plot_noisy_pl_results.py) |
| Sudoku noisy-label training update budget | [Long-training exports](../results/research/foureyes/noisy_pl_long/) | [plot_noisy_pl_long_results.py](../scripts/plot_noisy_pl_long_results.py) |
| Queen-plus-camel soft-weight transfer and wrong stops | [Matched transfer exports](../results/research/loki/h3u/) | [plot_public.py](../research/20261114/loki/h3u/plot_public.py) |
| Queen-plus-camel weights versus search on fixed support | [Fixed-support exports](../results/research/loki/h3v/) | [plot_public.py](../research/20261114/loki/h3v/plot_public.py) |
| Queen-plus-camel permutation initialization and search prior | [Permutation exports](../results/research/loki/h3w/) | [plot_public.py](../research/20261114/loki/h3w/plot_public.py) |
| Queen-plus-camel matched measured GPU time | [Matched-time exports](../results/research/loki/h3ab/) | [plot_public.py](../research/20261114/loki/h3ab/plot_public.py) |
| Queen-plus-camel shared-initialization proposal control | [Triangle exports](../results/research/loki/h3aa/) | [plot_public.py](../research/20261114/loki/h3aa/plot_public.py) |
| Complete-source minimum D4 rule | [Rule-selection exports](../results/research/loki/h3x/) | [plot_public.py](../research/20261114/loki/h3x/plot_public.py) |
| Fixed multi-family rule-transfer stress grid | [Stress-grid exports](../results/research/loki/h3y/) | [plot_public.py](../research/20261114/loki/h3y/plot_public.py) |
| Pooled smaller-source rule information | [Pooled exports](../results/research/loki/h3z/) | [plot_public.py](../research/20261114/loki/h3z/plot_public.py) |
| Sudoku independent recovery confirmation | [Fresh recovery exports](../results/research/ripper/recovery_confirm.json) | [plot_recovery_confirmation.py](../scripts/plot_recovery_confirmation.py) |
| Sudoku sample-count recovery | [Recovery exports](../results/research/ripper/recovery.json) | [plot_recovery_results.py](../scripts/plot_recovery_results.py) |
| Maze binary-preserving continuous relaxation | [Maze exports](../results/research/work/) | [plot_native_relaxation.py](../research/20261114/work/plot_native_relaxation.py) |
| N-Queens source-bank robustness | [H3k exports](../results/research/loki/h3k/) | [plot_public.py](../research/20261114/loki/h3k/plot_public.py) |
| Superqueen rule-transfer boundary | [H3l exports](../results/research/loki/h3l/) | [plot_public.py](../research/20261114/loki/h3l/plot_public.py) |
| Superqueen smaller-source control | [H3m exports](../results/research/loki/h3m/) | [plot_public.py](../research/20261114/loki/h3m/plot_public.py) |
| Superqueen downward transfer | [H3n exports](../results/research/loki/h3n/) | [plot_public.py](../research/20261114/loki/h3n/plot_public.py) |
| Original data-efficiency reproduction | [102 measured rows](../results/reproduction/data_efficiency/rows.csv), [collection](../results/reproduction/data_efficiency/collection.json) | [original renderer](../scripts/plot_data_efficiency.py) |

Regenerate the first two README figures from a public checkout:

```sh
uv run --with matplotlib --with numpy python scripts/plot_public_results.py
uv run --with matplotlib --with numpy python scripts/plot_count_results.py
```

The figure receipt at [assets/figures/provenance.json](../assets/figures/provenance.json) records the renderer and exact input hashes. The original reproduction plotter expects the local `artifacts/` layout; its unchanged final figures and compact inputs are mirrored under `results/reproduction/`.
