This guide distinguishes **checking saved evidence** from **running new fits
or inference**. Commands below were checked against the current source; no
new experiments were launched while preparing this guide. Run commands from
the repository root unless a different working directory is shown. The
[confirmed scope](../configs/study_scope.json) includes CRAM and its controls
and ablations; ordinary BP AdamW and the other baseline methods remain
published comparisons, outside local baseline retraining.

**Source, environment, and data.** The reviewed [paper](../references/2611.17309v1.md),
[conversion review](../references/CONVERSION_REVIEW.md), and
[coverage map](PAPER_COVERAGE.md) define the targets. The optimized source is
pinned by [upstream_pin.json](../upstream_pin.json) to
`62bbe830c7d6f87f2ba18f30a28039258b986454`. For a new checkout:

```sh
git clone https://github.com/Prefrontal-AI/CRAM.git upstream
git -C upstream checkout --detach 62bbe830c7d6f87f2ba18f30a28039258b986454
uv sync --project upstream --frozen --python 3.13.13
```

The recorded environment is Python 3.13.13, PyTorch 2.11.0+cu128, CUDA 12.8,
Triton 3.6.0, and NumPy 2.5.4. `upstream/uv.lock` has SHA256
`fdef606f276fefc325c1f7de3110c6032172d2532057348f4af43b3715ad1d09`.
Per-host packages, GPU/driver identities, and source hashes are retained in
`artifacts/deployment/{work,ripper,loki}-runtime.json`; the source/data
manifests are in the same directory. Local RTX PRO 6000 measurements do not
reproduce H200 wall times or establish a hardware-matched baseline speedup.

Saved-result commands require the actual `artifacts/`, datasets, and run
directories named in their receipts. A source-only checkout is insufficient.
For new data preparation, the pinned `upstream/scripts/prepare_data.py`
accepts `sudoku`, `maze-unique`, `maze-hard`, and `multi-solution`; the last
requires `--gram-src` pointing to the pinned GRAM builder. See its module
docstring, `upstream/scripts/sudoku_extreme_splits.json`,
`artifacts/data_provenance.json`, and `artifacts/deployment/data_manifest.json`
for exact dataset revisions, rows, and hashes. These preparation commands
download/write data; they are not saved-result audits.

**Independent implementation and document checks.** The following runs
small CPU correctness tests, rather than the benchmark experiment queues:

```sh
CUDA_VISIBLE_DEVICES= upstream/.venv/bin/python -m pytest -q tests/test_reference.py
CUDA_VISIBLE_DEVICES= upstream/.venv/bin/python -m pytest -q tests/test_data_efficiency_completion.py tests/test_data_efficiency_migration_evidence.py
```

`src/hopfield_cram/` is the independent implementation. Pytest's repository
configuration adds `src` to the import path; outside pytest use `PYTHONPATH=src`
or install the local package with `uv pip install --python <python> --no-deps -e .`.
The separate upstream suite is `upstream/.venv/bin/python -m pytest -q` from
`upstream/`; it includes CUDA checks and requires prepared datasets.

Structural document validation can be repeated with:

```sh
uv run --with beautifulsoup4 --with lxml --with pypandoc_binary python scripts/validate_paper.py
```

It rewrites conversion-validation artifacts under `references/review/`.
`scripts/check_paper_math.js` checks the extracted expressions with KaTeX
(recorded version 0.19.1); run it with Node able to resolve that package.
These checks supplement the retained manual PDF/LaTeX review.

**Saved-result audit and figure regeneration.** These commands read saved
measurements and write reports, derived arrays, or figures. They do not fit
models or launch test searches. Regenerating a report changes its hashes,
so dependent final receipts must be regenerated afterward.

| Purpose | Command from repository root |
|---|---|
| Full Sudoku fitted-threshold or hand-set result audit | `upstream/.venv/bin/python scripts/audit_sudoku_results.py learned` or replace `learned` with `hand` |
| Main models, controls, and saved-board checks | `upstream/.venv/bin/python scripts/audit_completed_controls.py` |
| Difficulty, failed-ablation reconstructions, and long BPTT evidence | `upstream/.venv/bin/python scripts/audit_final_controls.py` |
| Successful Maze-Hard 900/100 development control | `upstream/.venv/bin/python scripts/audit_maze_hard_development.py`; validates saved CSV row alignment, predictions, independent BFS checks, and provenance without inference. Receipt: `artifacts/maze_hard_development/validation.json`. |
| Tables 2/8 comparison and Table 4 halting comparison | `upstream/.venv/bin/python scripts/export_main_comparison.py`; `upstream/.venv/bin/python scripts/export_halting_comparison.py` |
| Table 7 / Appendix D.2 / Figure 4 final report | `artifacts/plotting-venv/bin/python scripts/finalize_objective_report.py` |
| Objective tables and Figure 4 plots | `artifacts/plotting-venv/bin/python scripts/export_objective_studies.py` |
| N-Queens free-pair and preconditioning aggregation | `upstream/.venv/bin/python scripts/finalize_nqueens_controls.py` |
| Figure 3a prerequisite check and export | `upstream/.venv/bin/python scripts/finalize_sudoku_difficulty.py --once` |
| Figure 3b collection and conditional plot | `upstream/.venv/bin/python scripts/collect_data_efficiency.py --plot` |
| Figure 3b plot alone, from retained collection | `artifacts/plotting-venv/bin/python scripts/plot_data_efficiency.py` |
| Independent final Figure 3b counts, hashes, and figure receipt | `upstream/.venv/bin/python scripts/audit_final_data_efficiency.py --visual-reviewed-png SHA256`; supply the exact PNG digest only after visually inspecting it. Requires completed collection, exited collector, and all three resumed-search audit receipts. |
| Final cross-family completion receipt | `python3 scripts/finalize_reproduction.py` |

Figure rendering uses the separate environment at `artifacts/plotting-venv`.
To recreate that environment in a new checkout:

```sh
uv venv --python 3.13.13 artifacts/plotting-venv
uv pip install --python artifacts/plotting-venv/bin/python -r artifacts/figures/plotting-requirements.txt
```

This pins Matplotlib 3.11.2 and its recorded dependencies. The same interpreter
runs `scripts/render_sudoku_compute.py`, `scripts/render_learned_couplings.py`,
`scripts/render_hand_energies.py`, and `scripts/render_maze_figure.py` from
their saved figure inputs. `scripts/reproduce_maze_figure.py` instead executes
inference and belongs with fresh experiments.

Run all desired collectors/exporters before the final figure audit. Render
Figure 3b after the last collection update, so `figure3b.json`'s source SHA256
matches the retained `collection.json`. Its final filenames require 102
validated rows and 17 complete six-seed points; partial runs produce an
explicit preview. The independent final data-efficiency receipt also records
a visual review and must be refreshed if a recorded input changes. Run
`finalize_reproduction.py` last: it requires all completion receipts and
checks their hashes; it does not manufacture missing results. Saved Sudoku
flags permit count/status audits, but do not permit reconstruction of final
Sudoku boards that were not retained.

**Fresh experiments require a separately staged campaign.** The recorded
workers use `/home/catid/hopfield`, hosts `foureyes`, `ripper`, `work`, and
`loki`, physical GPU locks, and task-owned output directories. Some wrappers
also retain historical process IDs or chain into later queues. There is no
single portable rerun command. Keep original evidence intact; stage new run
destinations and a new provenance manifest, verify source/data hashes, and
remap host/GPU assignments before launching. Existing completion markers may
skip work, while other runners overwrite their default result paths.

| Family | Entry point and frozen specification |
|---|---|
| Main Sudoku | `upstream/experiments/sudoku.py`: `diag` fits all declared splits, `search` runs development, `final` runs full test, and `time` measures complete fits. Exact argument arrays are in `configs/reproduction_manifest.json`, jobs `sudoku-fit`, `sudoku-dev-fit-s0` through `s2`, `sudoku-test-fit-s0` through `s2`, and timing jobs. Working directory: `upstream/experiments`. |
| Fresh mazes | `upstream/.venv/bin/python scripts/run_maze_reproduction.py unique --seed 0 --out <new-directory>`; replace `unique` with `hard` for Maze-Hard. Seeds 0–2; `unique-time --device cuda` and `unique-time --device cpu` provide separate timing. `scripts/run_maze_unique_queue.sh` retains the original three-seed sequence. |
| Maze-Hard development control | `upstream/.venv/bin/python scripts/run_maze_hard_development.py --seed S --gpu GPU --out <new-directory>` for seeds 0–2; runs the separate 900-training/100-development protocol and retains predictions for independent validation. This is distinct from the main fit on all 1,000 training mazes. |
| Independent maze checks | `scripts/run_reference_maze_unique.py --seed S` trains and evaluates on CPU. `scripts/validate_reference_maze_hard.py --seed S --out <new-directory>` performs fresh CPU inference using saved optimized weights. Neither is a saved-result-only audit. |
| Hand-set controls | `scripts/run_hand_sudoku.py --seeds 0 1 2` runs development and full test on CUDA; `scripts/run_hand_maze_reproduction.py` runs the CPU Maze-Unique control. |
| Main N-Queens / graph coloring | `bash scripts/run_nqueens_main.sh N GPU` for N=8 or 10; `bash scripts/run_graph_coloring_main_cpu.sh N`; GPU fit timing is separate in `scripts/run_graph_coloring_fit_gpu.sh`. These shell scripts use original output paths. |
| N-Queens controls / Sudoku difficulty | `scripts/run_nqueens_free_reproduction.py --n N --seed S`; `scripts/run_nqueens_preconditioning.py --n N --seed S --precond on\|off`; `scripts/run_sudoku_difficulty.py --seed S --test-size full`. Use seeds 0–2 and the frozen per-family recipes. |
| Table 7 objectives | `python3 scripts/objective_queue.py worker --lane HOST:GPU --config <new-config>`; original `configs/objective_jobs.json` contains 81 managed jobs plus three separately owned long BPTT jobs. |
| Figure 4 curves | `python3 scripts/figure4_queue.py worker --lane HOST:GPU --config <new-config>`; original `configs/figure4_jobs.json` contains 100 curves and probe schedules. `configs/figure4_queues/*.sh` are command inventories, not substitutes for the provenance/locking worker. |
| Appendix D.2 | `python3 scripts/d2_queue.py worker --lane HOST:GPU --config <new-config>`; original `configs/d2_supplemental_jobs.json` contains nine development-only fits. The worker waits for its Table 7 lane and then launches its Figure 4 worker. |
| Failed Maze-Hard ablations | `scripts/run_maze_hard_ablations.py --variant token-source --seed S` (S=0–2), or `--variant edge-gates --seed 0`. These are documented recipe reconstructions and launch CUDA work; `--preflight` only stages provenance. |
| Figure 3b data efficiency | `python3 scripts/data_efficiency_queue.py worker --lane 0` (lanes 0–2); original `configs/data_efficiency_jobs.json` pins 102 jobs and the separate `dataeff_source` tree. The current worker has fixed config/output paths and requires adaptation for a new campaign. |

The main Sudoku commands must explicitly retain `--scale hbar --hi 0.40
--lo-t 0.31 --theta 0.45 --stop fit`. Figure 3a uses the separate zero-energy
stopping protocol; a 10,000-puzzle standalone search is not trajectory-equivalent
to selecting that prefix from the full 412,786-puzzle search. The upstream
`final` command also checks a frozen-settings marker and a matching passing
development result. This local protocol marker is not a request for new user
permission; preserve the development gate before scoring test data.

Objective, D.2, and Figure 4 workers use the isolated `objective_source`
snapshot and configure `CRAM_DATA`, `V1_DATA`, `PYTHONPATH`, thread limits,
and GPU identity from their configs. Their `collect` subcommands copy saved
outputs over SSH without inference; `objective_queue.py summarize` uses
already collected local data. Collection is therefore a network/file mutation,
separate from a local audit. `finish_long_bptt.py` and `main_followups.py` are
experiment runners despite their names. Do not replay `build` actions over
the frozen configurations.

Figure 3b additionally needs the seed-42 reference made by
`reproduction/baselines/build_trm_sudoku.py`, its pinned TRM builder and
builder environment, followed by `scripts/prepare_data_efficiency.py`.
Preparation uses `reproduction/baselines/.venv/bin/python` and the paths in
`reproduction/baselines/env.sh`; that data-building dependency does not bring
baseline training into scope. `--include-baseline-data` is outside the default
CRAM preparation scope. The original unseeded arrays remain unavailable, so
new results must retain the recreation label and must not be presented as
same-training-array comparisons with published baselines.

Optional checkpoint sharding is an execution optimization with separate
same-host, cross-host, and actual-resume parity gates. Historical handoff and
launch receipts are not reusable launch instructions for a different process
or checkpoint. Preserve the original budget, RNG identities, source archives,
and timing distinction described in [the DE report](DATA_EFFICIENCY_REPRODUCTION.md).
For an active authorized campaign, `python3 scripts/status.py --save` and
`python3 scripts/study_progress.py --save` record live resource use and forward
progress across the configured hosts without launching experiments.
