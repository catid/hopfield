# Independent implementation and validation

The readable implementation is `src/hopfield_cram/`. It does not import the authors' implementation. Full dataset runs use the separately pinned optimized release in `upstream/`, commit `62bbe830c7d6f87f2ba18f30a28039258b986454`; its sparse Triton kernels provide the paper's high-throughput search. Passing reference tests is mathematical validation, not a claim to reproduce full test-set accuracy or H200 timings.

## Equations and task parameterizations

`PairwiseEnergy` stores one cost table per unordered pair. `energy()` evaluates Equation (1), with optional negative entropy, and `fields()` evaluates its pair/unary derivative. Reverse fields use the table transpose; self-interaction blocks are zero. `conditional_logits()` and `pseudolikelihood()` implement the exact one-cell conditionals of Equations (5)–(6), without any inference trajectory in the training graph.

| Model | Learned parameters | Structure supplied before training |
|---|---:|---|
| `SudokuEnergy()` | 263,169 | 81 opaque cells, 9 values; free pair/unary tables |
| `MazeEnergy(30, 30)` | 18 | Four shared 2×2 half-window tables and two unary costs |
| `NQueensEnergy(8)` / `(10)` | 113 / 181 | One on–on cost per displacement and one shared unary cost |
| `GraphColoringEnergy(8)` / `(10)` | 7,332 / 18,660 | Every pair table is an affine function of **all** adjacency tokens |
| `MazeHardEnergy(30, 30)` | 29 | Maze-Unique's 18 parameters plus one mass, five attention biases and five couplings |

All core models start at zero. N-Queens divides each offset gradient by its pair count relative to the mean count. Graph coloring's default is the full input-dependent model, with no built-in mapping from an edge token to its corresponding pair. Maze unary costs use the paper's additive convention; when comparing the release, `B = -b`. The optional maze context mask enables unary costs only on open non-endpoint cells, as in the release.

`euler_step()` implements Equation (3); `relax()` performs deterministic Euler relaxation with hard clamps. The continuous-time energy theorem does not guarantee monotonicity for an arbitrary finite step. `focused_gibbs()` implements single-cell heat-bath updates, focused cell selection, optional restarts/temperature schedules, and energy, local-margin, fixed-point, or termwise stopping. Its caller supplies an initial state and clue mask, never test solution labels. Each batch row is an independent chain. It recomputes fields to provide a transparent reference: its move budget is **not** the paper's FLOP budget.

## Local training and reduction convention

`fit_pseudolikelihood()` uses heavy-ball SGD, momentum 0.5 by default, the paper's cosine schedule, and proximal L1 after each update. Sudoku optionally draws a fresh independent digit permutation per board and update. The helper returns observed losses; timing and device/data preparation belong to the caller.

The released code averages over all free cells in a batch. Averaging the paper's displayed per-input Equation (5) instead would give each board equal weight, even when boards have different clue counts. The reference exposes this distinction explicitly: `pseudolikelihood(..., reduction="cells")` matches the release; `reduction="boards"` first normalizes each nonempty board by its own number of free cells. Empty all-clamped training batches raise an error.

```python
from hopfield_cram import SudokuEnergy, TrainingConfig, fit_pseudolikelihood

# labels: (training_boards, 81), values 0..8; free: boolean non-clue mask.
model = SudokuEnergy().to(labels.device)
losses = fit_pseudolikelihood(
    model, labels, free,
    TrainingConfig(updates=32, learning_rate=540,
                   l1=3e-3 / 9, digit_augmentation=True, seed=0),
)
```

Graph context is a `(batch, n_vertices*(n_vertices-1)//2)` tensor of binary upper-triangle adjacency tokens in `model.pairs` order. Ordinary maze states are flattened; `MazeEnergy` takes a boolean unary mask as context. N-Queens uses binary occupancy and receives given queens through the inference clamp mask.

## Maze-Hard fixed point and adjoint

`MazeHardEnergy` implements Equations (11)–(15). The learned `log_excess_mass` represents `log(m-4)`, with the paper's initial excess mass `4*exp(4)`. The hidden state is stored as `log(z)`: ordinary floating-point `z` would underflow along long corridors. A custom autograd function settles the log-domain Jacobi equation in its forward pass and solves the transpose fixed-point Jacobian system in backward. It never differentiates through a stored solver trajectory. Unreachable cells retain `log(z)=-inf`; completely unreachable attention windows use the release's explicit fallback to attention biases alone.

The attention self term is quadratic in continuous path probability, `-kappa_0*alpha_ii*p_i^2`. Its continuous derivative contains `2*p_i`, while exact binary conditionals use `p_i^2=p_i` and contain the self coefficient only once. `fields()` and `conditional_logits()` keep these distinct. Using continuous fields directly as discrete conditional costs would be incorrect for this extension.

`fit_maze_hard()` implements local pseudolikelihood with Adam, default 4,000 updates, batch 64 and learning rate 0.1. Its Torch minibatch RNG differs from the release's NumPy RNG; bit-identical final checkpoint hashes are not expected. `prepare(tokens)` settles the hidden layer once, allowing reuse during answer inference. Token encoding is wall=0, open=1, start=2, goal=3.

```python
from hopfield_cram import MazeHardEnergy, fit_maze_hard, solve_maze_hard

model = MazeHardEnergy(30, 30).to(tokens.device)
losses = fit_maze_hard(model, tokens, labels, seed=0)
result = solve_maze_hard(model, test_tokens)
```

`solve_maze_hard()` uses the released inference settings: initial answer logit 3, step size 0.2, at most 2,048 steps, and a rounded-board single-flip fixed-point check every 16 steps with tolerance `1e-6`. A local fixed point is not a certificate of puzzle correctness. General `relax()` has residual stopping instead and should not be confused with this paper-specific inference protocol.

## Validation evidence

Run from the workspace root:

```sh
PYTHONPATH=src upstream/.venv/bin/python -m pytest -q tests/test_reference.py
```

Observed on CPU: **27 passed in 1.14 s**. The earlier core/latent parity run had **26 passed in 1.08 s**; the final additional test checks the exact Maze-Hard answer-relaxation and stopping schedule. Machine-readable outcomes, runtime versions, source hashes and JUnit cases are in `artifacts/reference-tests.json` and `artifacts/reference-tests.xml`.

Independent oracles cover:

- Literal pair-energy enumeration, autograd derivatives, symmetric dense interactions, and zero self blocks.
- Exact one-cell conditional enumeration and parameter gradients; explicit cell-versus-board reduction behavior.
- Every task's parameter count, maze boundaries/transposes, N-Queens offset preconditioning, and graph cross-token dependence.
- The continuous-time Lyapunov identity at inverse temperature 1.7, clamped Euler updates, and a measured small-step energy decrease.
- Zero-temperature Gibbs energy monotonicity, fixed-point stopping, seeded repeatability, best-board return, all-clamped inputs, and an actual learned two-solution distribution.
- Hidden log-state and implicit gradients against a separate dense linear solve; complete loss-to-mass finite differences; a corridor whose `z` underflows in float64 while `log(z)` remains finite; disconnected attention and a short Adam fit.

Separate parity checks compare with the pinned release: Sudoku's actual 81×9 energy/PL/parameter gradients; maze energy/PL/gradients with the explicit unary sign mapping; graph coloring's full affine model; and Maze-Hard's hidden state, offset-reordered attention, visible energy, exact PL, all parameter gradients, and answer predictions/stopping steps. These comparisons use synthetic tensors and small graphs, not copied published predictions. Upstream-dependent parity tests skip if that optional checkout/package is unavailable; the independent tests remain usable.

This reference implements the principal PL method and latent extension. The twelve-objective training comparison, dataset-scale sweeps, hand-set controls and looped-model baselines are separate experiment protocols in the pinned release; their run status belongs in the reproduction report, not in these unit-test claims.

## Full-dataset independent inference check

The independent pairwise implementation was also trained and tested end to end on Maze-Unique for seeds 0–2. `scripts/run_reference_maze_unique.py` decodes the pinned arrays directly, trains from zero with the exact 20-update NumPy permutation/minibatch schedule, and uses only `hopfield_cram` for energy, pseudolikelihood and focused Gibbs inference. All three CPU runs solved all 1,000 test mazes with zero wrong stops or budget failures. The fit loops took 0.213–0.242 seconds and full searches 46.17–48.06 seconds on one CPU thread. CPU RNG/batching differs from the optimized CUDA search. Maximum coefficient differences from the fresh optimized fits were 1.5074e-4, 3.4571e-5 and 1.0175e-4; checkpoint identity is not claimed. Per-update losses, exact batch indices, models, predictions and hashes are retained under `artifacts/reference_maze_unique/`.

The independent `solve_maze_hard` implementation was also run on all 1,000 test mazes for each of the three fresh optimized-fit checkpoints. On one CPU thread, float32 inference took 29.19, 28.97 and 29.52 seconds. Every prediction, stopping step and stopping flag agreed with the optimized implementation for all three seeds. Each solved 999/1,000; all reproduced the same incorrect fixed point on maze 106. This validates independent inference using transferred trained weights; it is not an independent training reproduction.

Run `scripts/validate_reference_maze_hard.py --seed <0|1|2>` with the project interpreter. The script imports only `hopfield_cram` for inference, explicitly maps the checkpoint's signs and offset order, and never feeds the saved predictions or labels into the solver. Source/data/checkpoint hashes and all output arrays are retained in `artifacts/reference_maze_hard/`. These CPU times describe the readable implementation and are not the paper's GPU timing comparison.
