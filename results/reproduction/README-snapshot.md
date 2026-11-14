# CRAM / Hopfield paper implementation and reproduction

Implementation and measured reproduction of **Classical Energies Learn to Reason 1,000× Faster than Looped Models**, arXiv:2611.17309v1, by Benjamin Hoover, Hugh Blayney, and James C. R. Whittington.

## Reviewed paper

Read the complete [Markdown paper](references/2611.17309v1.md) and [conversion review](references/CONVERSION_REVIEW.md). All 24 PDF pages were reviewed against the original LaTeX. The conversion retains 15 numbered equations, 8 tables, 6 figures with original vector assets, 66 references, and Appendices A–F. This work was completed before implementation and reproduction runs.

## Implementations

- `src/hopfield_cram/`: independent, readable PyTorch implementation of the pairwise energy, exact pseudolikelihood, SGD with proximal L1, deterministic relaxation, focused Gibbs search, task parameterizations, and the Maze-Hard hidden fixed point with an implicit adjoint. See [implementation notes](docs/IMPLEMENTATION.md).
- `upstream/`: pinned authors' implementation, including sparse Triton search kernels used for full-scale reproduction. [upstream_pin.json](upstream_pin.json) records its repository and commit. Its published result files are imported evidence, not locally reproduced results.
- `tests/test_reference.py`: independent brute-force and gradient checks, including numerical comparisons with pinned upstream code.

The reference solver prioritizes inspectability and correctness; full benchmark evaluation uses the audited optimized upstream implementation. A reference solver's explicit move budget is not the paper's FLOP budget.

The [rerun guide](docs/RERUN_GUIDE.md) documents environment setup, saved-result audits, figure regeneration, and the separate requirements for launching a fresh experiment campaign.

The optional multi-GPU search dispatcher in `scripts/sharded_sudoku_moves.py` derives its move kernel from that pinned release, under the retained [Apache 2.0 license](references/CRAM-LICENSE.txt). Its bounded validation compares complete search state and pre-refresh values against the original solver before measuring throughput. Execution records distinguish this dispatcher from the original single-GPU implementation.

## Environment and checks

```bash
git clone https://github.com/Prefrontal-AI/CRAM.git upstream
git -C upstream checkout --detach 62bbe830c7d6f87f2ba18f30a28039258b986454
cd upstream
uv sync --frozen --python 3.13.13
cd ..
upstream/.venv/bin/python -m pytest -q tests/test_reference.py
```

The locked reproduction environment uses Python 3.13.13, PyTorch 2.11.0+cu128 and Triton 3.6.0. It is deployed on the authorized hosts foureyes, ripper, work and loki. Local measurements use RTX PRO 6000 Blackwell GPUs; the paper's H200 timing numbers are retained as published comparisons, not hardware-matched measurements.

To use the independent implementation directly, set `PYTHONPATH=src`, or install this package into your chosen environment with `uv pip install --python <python> --no-deps -e .`.

## Example: train the Sudoku energy

```python
import torch
from hopfield_cram import SudokuEnergy, TrainingConfig, fit_pseudolikelihood, fit_statistics

# labels: long tensor (N,81), digits 0..8; free: boolean tensor (N,81).
# Only the training solutions and clue masks belong here.
model = SudokuEnergy().to(labels.device)
losses = fit_pseudolikelihood(
    model, labels, free,
    TrainingConfig(updates=32, learning_rate=540, momentum=.5,
                   l1=3e-3/9, digit_augmentation=True, seed=0),
)
statistics = fit_statistics(model, labels)
```

Training uses only labeled solutions and local conditional probabilities; it does not backpropagate through search. The implementation exposes both free-cell-weighted PL (the released code) and per-board PL (the displayed equation) explicitly.

## Reproduction evidence and status

The confirmed [study scope](configs/study_scope.json) is **CRAM main results, controls and ablations, with clearly labeled published baseline results**. The [main comparison](docs/MAIN_COMPARISON.md) places fresh measurements alongside the published baselines. The [paper coverage map](docs/PAPER_COVERAGE.md) connects every table, figure and supplemental claim to its evidence and declared limits. The broader original source inventory remains in [configs/reproduction_manifest.json](configs/reproduction_manifest.json); its baseline training jobs are outside the active scope. Neither manifest is evidence that a job has completed.

Actual local evidence is under `artifacts/`, with task-specific fit/evaluation outputs under `upstream/experiments/runs/` and the corresponding authorized remote task directories. In particular:

- `artifacts/upstream_test_summary.json`: 91 upstream tests pass; the first attempt had six missing-data failures that passed after data preparation.
- `artifacts/reference-tests.json`: independent implementation test evidence.
- `artifacts/checkpoint_maze_unique.log`, `checkpoint_maze_hard.log`: three-seed released-weight validation.
- `artifacts/sudoku_fresh_vs_released.json`: from-scratch weight comparisons; largest difference 2.3842e-7, same sparse parameter counts.
- `artifacts/sudoku_dev_s*.log`, `sudoku_test_s*.log`: development and full held-out test evaluation of fresh Sudoku fits.
- `artifacts/sudoku_audit/learned-full-test-audit.json`: complete three-seed Sudoku audit; [result/count report](docs/SUDOKU_OPERATION_AUDIT.md) distinguishes fresh CRAM from published baseline comparisons.
- `artifacts/reference_maze_unique/`, `reference_maze_hard/`: full-dataset independent implementation checks.
- `artifacts/sudoku_gpu_timing.log`, `sudoku_cpu_timing.log`: matched-draw complete fit timing, cold and seven warm repeats retained.
- `artifacts/data_provenance.json`, `artifacts/deployment/data_manifest.json`: dataset identities and file hashes.
- `artifacts/utilization.jsonl`: timestamped live GPU/process snapshots.

Run `python3 scripts/status.py --save` for an updated read-only hardware snapshot.
Run `python3 scripts/study_progress.py --save` to include live process counters, queue stages, training updates and search checkpoint timestamps across all four hosts. Minute-by-minute snapshots are retained under `artifacts/study_progress/`.

**The agreed reproduction scope is complete, with the limits below:** main CRAM results and controls, all 84 objective-comparison runs, nine supplemental implicit-gradient runs, 100 training curves, and all 102 data-efficiency runs. Fresh Sudoku solves 99.996689% across three full test runs with zero wrong stops; Maze-Unique solves 100%, Maze-Hard 99.9%, graph-coloring quality matches the paper exactly, and N-Queens differences are recorded. The final data-efficiency curve reaches a six-seed median of 99.415% with 30 unique training puzzles and 99.99% with 100.

The data-efficiency study uses an explicitly seeded recreation because the original unseeded reference was not released; missing failed-ablation recipes are labeled reconstructions. Baselines are published-only, and the paper's H200 “1,000×” training-time claim is not independently established on this Blackwell campaign. See [REPRODUCTION.md](REPRODUCTION.md), [the final Figure 3b](artifacts/data_efficiency/figure3b.png), and [the consolidated audit receipt](artifacts/final_reproduction_audit.json) for measured results and verification boundaries.
