# Figures and their data

All README graphs show measured results. They are ordinary scientific plots, with downloadable SVG or PDF versions beside the PNGs.

| Figure | Public data | Renderer |
|---|---|---|
| No-optimizer Sudoku and budget curve | [Count and learned exports](../results/research/foureyes/) | [plot_count_results.py](../scripts/plot_count_results.py) |
| Sudoku counts from raw solutions | [Raw-count and PL exports](../results/research/ripper/) | [plot_raw_count_results.py](../scripts/plot_raw_count_results.py) |
| Sudoku matched transformation control | [Transformed-PL exports](../results/research/foureyes/raw_transform/) | [plot_raw_transform_results.py](../scripts/plot_raw_transform_results.py) |
| Sudoku fixed-restart control | [Restart exports](../results/research/foureyes/restart/) | [plot_restart_results.py](../scripts/plot_restart_results.py) |
| Reserved Sudoku test and sparse speed | [Ripper](../results/research/ripper/), [stopping rules](../results/research/foureyes/) | [plot_public_results.py](../scripts/plot_public_results.py) |
| Maze size transfer, branch depth and step size | [Work exports](../results/research/work/) | [public renderer](../research/20261114/work/) |
| N-Queens optimizer and update budget | [Loki exports](../results/research/loki/) | [Loki scripts](../research/20261114/loki/) |
| N-Queens unseen complete patterns | [Pattern holdout exports](../results/research/loki/h3g/) | [plot_public.py](../research/20261114/loki/h3g/plot_public.py) |
| Original data-efficiency reproduction | [102 measured rows](../results/reproduction/data_efficiency/rows.csv), [collection](../results/reproduction/data_efficiency/collection.json) | [original renderer](../scripts/plot_data_efficiency.py) |

Regenerate the first two README figures from a public checkout:

```sh
uv run --with matplotlib --with numpy python scripts/plot_public_results.py
uv run --with matplotlib --with numpy python scripts/plot_count_results.py
```

The figure receipt at [assets/figures/provenance.json](../assets/figures/provenance.json) records the renderer and exact input hashes. The original reproduction plotter expects the local `artifacts/` layout; its unchanged final figures and compact inputs are mirrored under `results/reproduction/`.
