# Maze reproduction

Fresh fits reproduce the paper's Maze-Unique and Maze-Hard accuracies across training seeds 0, 1, and 2. The hand-set Maze-Unique control also solves every development and test maze across three inference seeds. These are new measurements from the pinned optimized release; the independent readable implementation and its parity tests are documented in [IMPLEMENTATION.md](IMPLEMENTATION.md).

The experiment source is commit `62bbe830c7d6f87f2ba18f30a28039258b986454`. The paper is arXiv:2611.17309v1. Published values below come from its tables and the separately identified imported `upstream/results/op-counts.json`; local results come from `artifacts/fresh_maze/`. The machine-readable comparison is [reproduction_summary.json](../results/reproduction/fresh_maze/reproduction_summary.json).

| Task / model | Paper solved, % | Fresh solved per seed, /1,000 | Paper full-count FLOPs / maze | Fresh full-count FLOPs / maze, mean of seeds |
|---|---:|---|---:|---:|
| Maze-Unique, trained CRAM | 100.00 ± 0.00 | 1,000; 1,000; 1,000 | 836,870.85 | 836,870.85 |
| Maze-Unique, hand-set | 100.00 ± 0.00 | 1,000; 1,000; 1,000 | 837,048.85 | 837,048.85 |
| Maze-Unique, WSP-DeadEnd (rule-programmed, k=0) | — | 1,000; 1,000; 1,000 | — | 1,804,550.40 |
| Maze-Unique, WSP-Wavefront (rule-programmed, k=0) | — | 1,000; 1,000; 1,000 | — | 18,093,737.80 |
| Maze-Unique, WSP-Potential (rule-programmed, k=0) | — | 1,000; 1,000; 1,000 | — | 164,266,824.70 |
| Maze-Hard, CRAM + latent | 99.90 ± 0.00 | 999; 999; 999 | 14,058,028.20 | 14,058,028.20 |
| Maze-Hard, WSP-DeadEnd (rule-programmed, k=0) | — | 0; 0; 0 | — | 184,291.20 |
| Maze-Hard, WSP-Wavefront (rule-programmed, k=0) | — | 1,000; 1,000; 1,000 | — | 17,110,750.60 |
| Maze-Hard, WSP-Potential (rule-programmed, k=0) | — | 1,000; 1,000; 1,000 | — | 50,624,506.10 |

The paper rounds the CRAM operation counts to `8.37e5`, `8.37e5`, and `1.41e7`. Three runs of a shared 1,000-maze test set are three seed evaluations, not 3,000 independent mazes. Maze-Unique and the hand-set control have zero wrong stops and zero budget failures. Maze-Hard CRAM+z has one wrong stop and zero budget failures in each seed. Under zero-training Weight-Space Programming (`WSP-*`, `k=0` training examples, `0` learned parameters, [wsp_maze_complete_summary.json](../results/wsp/maze/wsp_maze_complete_summary.json)), purely local `WSP-DeadEnd` solves `1,000/1,000` on tree-structured Maze-Unique (`1,804,550.4` `FC2` FLOPs; `2,022,599.9` w/ compare) but halts on the cyclic 2-core across all `1,000` Maze-Hard test mazes (`0/1,000` solved, `100%` wrong halts), whereas both geodesic `WSP-Wavefront` and resistive-flow `WSP-Potential` solve **`1,000/1,000` on both Maze-Unique and Maze-Hard with zero wrong stops**—including test index **106** (`117/117` shortest-path cells); see [WSP_BENCHMARK_REPORT.md](WSP_BENCHMARK_REPORT.md).

## Frozen protocols and validation

**Maze-Unique:** 18 free parameters, zero initialization, 20 pseudolikelihood updates, SGD with learning rate 1, momentum 0.5, cosine schedule, and no L1. Each fit uses the official 1,000-maze training data and its seeded minibatch draws. Test inference is the official zero-temperature focused Gibbs search from all open cells on, with inference seed 0. All three trained models solve the full 1,000-maze test set.

The initialization claim in Appendix C.4 also holds for the fresh fits: with every free cell off and the endpoints still clamped on, the empty board has lower energy than the labeled path on all 100 development mazes for each seed. Independent NumPy sums over unordered offset pairs and unary terms confirm the source energy calculation. The solution-minus-empty energy gap ranges from 91.98 to 131.70 across the 300 seed/maze evaluations. This verifies why unrestricted global energy minimization is not equivalent to the specified all-open dead-end-pruning trajectory. Evidence: [maze-empty-board-energy-audit.json](../results/reproduction/figures/maze-empty-board-energy-audit.json).

**Maze-Hard:** the frozen `experiments/recipes/maze-hard-latent.json` recipe has 29 parameters and 4,000 Adam updates. Training uses the official fused Triton hidden-state and adjoint routines. Scoring rebuilds the model with the default PyTorch hidden solver, as the released final scorer does, then performs damped descent from logit 3 with step size 0.2, a fixed-point check every 16 updates, and the 2,048-step budget. A separate output directory per seed avoids the official fit command's shared `models.json` read-modify-write race.

The separate successful-recipe development claim in Appendix F is also reproduced: seeds 0, 1 and 2 each solve **100/100 held-out training mazes**, with zero wrong stops and zero budget failures. Each fresh fit uses the other 900 training mazes, selected by the released `numpy.random.default_rng(0).permutation(1000)` split. All 4,000 updates use the unchanged published recipe; this validation performs no recipe selection and opens only `train.csv`. The same 100 development mazes are evaluated under three fitted models, so 300 successful seed evaluations do not represent 300 distinct mazes.

Full development predictions, inputs, labels, stop flags and steps are retained. An independent CPU audit decodes the original CSV, verifies the split identities and all 4,000-update schedules, and checks exact labels, walls, endpoints, connectivity, path degrees, shortest lengths and BFS tie order. Every seed produces 100 valid shortest paths. The three fits ran concurrently on Foureyes GPUs 0–2; their complete cold fit times were 11.3723, 11.5216 and 11.1061 seconds on RTX PRO 6000 Blackwell Max-Q GPUs. These single observations include setup and the training loop, exclude CUDA context creation and evaluation, and are not a matched H200 timing comparison. The [development validation receipt](../results/reproduction/maze_hard_development/validation.json) hashes 73 evidence files and records all three results. This held-out control is separate from the 1,000-training-maze fits used for Table 8.

**Hand-set Maze-Unique:** the exact official factorized energy is used on CPU, with no training. For each inference seed 0, 1, and 2, the official development gate runs first: all 100 development mazes solved with no wrong stop. Only then is its full 1,000-maze test set scored. There are 16 stored numbers, zero learned parameters, and three effective nonzero energy coefficients (`-1`, `-0.1`, `1.5`), which explains the paper's parameter count of 3. An independent exhaustive check of all 256 binary patterns of the eight neighbors confirms the local field formula and exact dead-end-removal rule; maximum floating-point discrepancy is `3.58e-8`. See [source_consistency.json](../results/reproduction/fresh_maze/hand_maze_unique/source_consistency.json).

Predictions were captured during each official single test pass, then checked independently using NumPy and deque BFS. The audit checks exact labels over every cell, walls, endpoints, path degrees, connectivity, shortest length, and the down/left/right/up shortest-path tie convention. All 1,000 targets in each dataset match independent BFS. Each trained or hand-set Maze-Unique run produces 1,000 valid shortest paths; each Maze-Hard run produces 999. Full inputs, labels, predictions, stop flags, and Maze-Hard descent steps are retained in per-seed `test_predictions.npz` files. No hyperparameters or stopping rules were changed in response to test results.

The one Maze-Hard failure is test index **106** in every seed. It stops after 160 descent steps with all 117 target cells plus six extra open cells at zero-based coordinates `(29, 0)` through `(29, 5)`. Endpoints and walls are respected, but the extra cells create invalid path degrees. This demonstrates that the local fixed-point stop is not a certificate of puzzle correctness. See [hard_failure_audit.json](../results/reproduction/fresh_maze/hard_failure_audit.json).

## Operation-count conventions

The raw released evaluators retain older component counts. Those are **not** the FLOPs shown in the paper's final tables:

| Task / model | Legacy evaluator FLOPs / maze, seed mean | Paper convention FLOPs / maze, seed mean |
|---|---:|---:|
| Maze-Unique, trained CRAM | 560,511.72 | 836,870.85 |
| Maze-Unique, hand-set | 560,511.72 | 837,048.85 |
| Maze-Hard, CRAM + latent | 6,012,871.80 | 14,058,028.20 |

The paper convention, called FC2 in the release, charges 2 per multiply-add and 1 for each other arithmetic or transcendental evaluation. Comparisons, selections, masks, indexing, movement, casts, and random-number generation are excluded from that total; the counter separately records comparison-inclusive totals.

We freshly ran the official `FullCounter` on training mazes using each new model. The measured cost is an exact affine function of the relevant move counts on all sampled training mazes: six per trained/hand-set Maze-Unique model and eight per Maze-Hard model. We applied these measured coefficients to the already saved test steps, without rescoring the test set. Maze-Hard hidden-step counts are uniquely decoded from the saved legacy count; decoded descent steps must exactly equal those captured directly during the original scoring pass. The three Maze-Hard full-count means are `14,060,729.7`, `14,073,696.9`, and `14,039,658.0`. All aggregate operation counts exactly match the imported full-precision records.

See [fc2_summary.json](../results/reproduction/fresh_maze/fc2_summary.json), each per-seed `fc2_flops.json`, and the hand-set [fc2_flops.json](../results/reproduction/fresh_maze/hand_maze_unique/fc2_flops.json). Counts are logical operations; they do not measure wall time or GPU utilization.

## Local timing and numerical agreement

Work used RTX PRO 6000 Blackwell Workstation GPUs; Loki used RTX PRO 6000 Blackwell Max-Q Workstation GPUs. The remote runtime was Python 3.13.13, PyTorch 2.11.0+cu128, CUDA 12.8, and Triton 3.6.0. Dataset hashes and runtime inventories are in `artifacts/deployment/`; every maze run also records its own source, data, model, wrapper, driver, and device provenance.

Maze-Unique complete-fit timing follows the official scope: data already in host memory through the inference-ready model, including conversion, transfer, initialization, updates, compilation, and bookkeeping; file reading, process imports, evaluation, and saving are excluded. Cold means the first fit in a fresh process, with CUDA context creation separately reported. Warm means the median of seven further complete fits, each reinitialized with the same seed. The CPU run uses one thread and no visible GPU.

| Seed | GPU cold, s | GPU warm median, s | CPU cold, s | CPU warm median, s |
|---|---:|---:|---:|---:|
| 0 | 1.29570 | 0.10758 | 0.04325 | 0.03915 |
| 1 | 0.81125 | 0.10739 | 0.04288 | 0.03863 |
| 2 | 0.70096 | 0.10734 | 0.04279 | 0.03790 |

Mean warm complete fit is **0.10743 s on Loki GPU** and **0.03856 s on one Loki CPU thread**. The paper reports 0.13 s on an H200; the hardware differs, so these are local timings rather than an H200 speed reproduction. For this tiny 20-update model, the CPU is faster under the same fit boundary. Every timing repeat passed the released model-equivalence tolerance, although not bitwise equality. Simultaneous independent work on other devices and CPU preparation can affect these wall times.

Maze-Hard complete cold fit times were 28.3214 s (work GPU0, seed0), 19.3773 s (work GPU1, seed1), and 26.4949 s (Loki GPU0, seed2); training-loop components were 26.8386, 19.0431, and 25.0266 s. These are single cold observations with potentially different compilation-cache states, not repeated steady-state benchmarks. Instrumented inference wall times include prediction transfers and are not used for speed claims.

New model weights differ slightly from the released H200-trained checkpoints. Maximum absolute parameter discrepancies are `1.8451e-4`, `6.3181e-5`, `1.0082e-4` for Maze-Unique and `1.1444e-4`, `8.2016e-5`, `1.7738e-4` for Maze-Hard. Accuracy and aggregate operation counts still match. Per-parameter differences and checkpoint hashes are retained in `weight_comparison.json`; this is not a claim of bit-identical training.

The initial CPU timing wrapper failed because both it and the upstream timing command set interop threads. The wrapper now leaves that setting to the upstream CPU command. Only the failed CPU timing was resumed; completed test scoring was not repeated. Raw failure logs remain available. The upstream timing prose also hardcodes “H200”; local timing summaries preserve that original text under `upstream_definition` and explicitly correct the active hardware label without changing measurements.

## Independent implementation mapping

Full dataset scoring above uses the official optimized implementation directly; it requires no model-format conversion. Separate source tests compare the readable implementation against the release on synthetic tensors and small graphs:

- Maze-Unique: released `T = MazeEnergy.tables`, released `b = -MazeEnergy.unary_cost`. Energy, pseudolikelihood, and parameter gradients match with the explicit unary sign change.
- Maze-Hard: released `T = maze.tables`, `b = -maze.unary_cost`, and `mu = log_excess_mass`; `beta` and `kappa` are reordered to the release's attention-offset order. Tests cover the hidden state, attention, visible energy, exact pseudolikelihood, gradients, and answer predictions/stopping steps.

These tests are in [test_reference.py](../tests/test_reference.py), with recorded evidence in `artifacts/reference-tests.json`. They establish implementation consistency; they do not substitute for the full dataset measurements above. Fresh run metadata confirms 18 stored/learned parameters for Maze-Unique and 29 for Maze-Hard.

## Commands and artifacts

### Figure 6: fresh Maze-Hard trajectory

The official figure exporter was replayed on CPU for **test maze 0**, using the fresh seed-0 checkpoint, with all input adapters and outputs redirected under `artifacts/figures/`. This single-maze figure replay exactly matches the prediction captured during the original full-test pass and the target. It takes 120 hidden-state steps and 144 damped descent steps. The checkpoint SHA256 is `84d694828c90cbf55bcf2eb84136324e2a5d482e4847e50d7b9f67b39c8aa230`.

The continuous answer energy decreases from `−2395.937988` to `−6634.140137`, with no increases before the stopping check. The rounded board remains unchanged for 48 additional steps. Float32 evaluation shows three post-stop energy increases, each at most `0.00048828125`; the final energy equals the energy at the stop. The largest answer change falls from `1.3583e-4` on the first extra step to `3.2596e-9` on the last. The observed trajectory supports energy decrease and stability to the stated precision, rather than bitwise monotonicity after convergence.

Standalone plots are [energy (PNG)](../results/reproduction/figures/maze-hard-energy-fresh.png) and [maze panels (PNG)](../results/reproduction/figures/maze-hard-panels-fresh.png), with matching `.pdf` and `.svg` files for export. The [raw figure JSON](../results/reproduction/figures/maze-hard-figure-fresh.json) preserves every energy, hidden state, pointer, and snapshot; the [audit JSON](../results/reproduction/figures/maze-hard-figure-audit.json) records source and checkpoint hashes. Reproduce with `CUDA_VISIBLE_DEVICES='' upstream/.venv/bin/python scripts/reproduce_maze_figure.py`, then `artifacts/plotting-venv/bin/python scripts/render_maze_figure.py`. The plotting environment is isolated from the pinned experiment environment.

The maze fit wrapper accepts a fresh output directory and refuses to overwrite a finished run:

```sh
CRAM_DATA=$PWD/upstream/data CUDA_VISIBLE_DEVICES=0 upstream/.venv/bin/python \
  scripts/run_maze_reproduction.py unique --seed 0 --out artifacts/new_run/maze_unique/seed0
CRAM_DATA=$PWD/upstream/data CUDA_VISIBLE_DEVICES=0 upstream/.venv/bin/python \
  scripts/run_maze_reproduction.py hard --seed 0 --out artifacts/new_run/maze_hard/seed0
```

Repeat with training seeds 1 and 2. For complete Maze-Unique fit timing, use `unique-time` with the corresponding fitted output directory and `--device cuda` or `--device cpu`. The hand-set wrapper is `scripts/run_hand_maze_reproduction.py`; its development/test outputs are isolated under `artifacts/fresh_maze/hand_maze_unique`. Operation reconstruction is `scripts/count_fresh_maze_flops.py`. Both counter scripts use training inputs for the fresh operator measurements.

For the successful Maze-Hard 900/100 development control, use a new output directory for each seed:

```sh
upstream/.venv/bin/python scripts/run_maze_hard_development.py \
  --seed 0 --gpu 0 --out artifacts/new_run/maze_hard_development/seed0
```

Repeat for seeds 1 and 2 on available GPUs. Revalidate the saved original three-seed control without training or inference with `upstream/.venv/bin/python scripts/audit_maze_hard_development.py`.

A separate released-checkpoint Sudoku development audit is recorded at [released_sudoku_dev/summary.json](../results/reproduction/released_sudoku_dev/summary.json): all three checkpoints solve 10,000/10,000 development puzzles with no wrong stop or budget failure. Their mean FLOPs differ from the H200 records by +4.76%, −3.77%, and −3.04%, so the exact-record checker reports a difference for each seed. The cause of those trajectory/count differences is not established by this audit. These development results are separate from fresh Sudoku training and full-test reproduction.
