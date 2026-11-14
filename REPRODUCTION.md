# Reproduction status

Complete as of 2026-11-14 UTC within the confirmed scope and declared limits: the reviewed paper conversion, independent implementation, seven main CRAM model configurations, full test evaluations, controls, objective ablations, and all experimental figures. Figure 3b's final search finished and all 102 measured runs validated. The confirmed scope is **CRAM main results, controls and ablations, using clearly labeled published baseline results**. Published results are never presented here as new measurements. [Study scope](configs/study_scope.json) records this choice.

## Completed evidence

| Experiment | Protocol | Locally observed |
|---|---|---|
| BP AdamW baseline control | Published comparison only, outside local training scope | No local completed baseline result; task-owned baseline training canceled after scope clarification |
| Independent implementation | 27 CPU tests; brute-force, finite-difference and pinned-upstream gradient/solver comparisons | 27 passed |
| Independent Maze-Unique training and inference | Seeds 0–2; 20 updates on 1,000 training mazes; full 1,000 test mazes, CPU | 1,000/1,000 each, zero wrong stops and budget failures |
| Independent Maze-Hard inference | All 1,000 test mazes, each fresh checkpoint, seeds 0–2, float32 CPU | Every prediction and stopping step matches optimized inference; 999/1,000 each |
| Authors' implementation | Pinned source; 91 CPU/CUDA tests | 91 passed; six initial missing-data failures passed after preparation |
| Sudoku fresh training | Seeds 0–2; 1,000 official training puzzles each; 32 PL/SGD updates | Correct sparse counts; max weight difference from released checkpoints 2.3842e-7 |
| Sudoku fresh development | 10,000 held-out development puzzles, separately for each seed | 10,000/10,000 each; no wrong stops |
| Sudoku fresh full test | 412,786 puzzles per seed, seeds 0–2, four chains and frozen search budget | 412,776 / 412,771 / 412,770 solved; 99.996689% ± 0.000779 percentage points; zero wrong stops, 41 total budget failures |
| Sudoku released development | Same 10,000 development puzzles, seeds 0–2 | 10,000/10,000 each; operation counts vary from H200 records |
| Maze-Unique fresh training/test | Seeds 0–2; 1,000 training and 1,000 test mazes | 1,000/1,000 each; no wrong stops; 836,870.85 FLOPs/maze under paper counting |
| Maze-Unique hand-set control | 3 inference seeds; 1,000 test mazes each | 1,000/1,000 each; 837,048.85 FLOPs/maze |
| Maze-Hard fresh training/test | Seeds 0–2; 4,000 updates; 1,000 training and 1,000 test mazes | 999/1,000 each; same incorrect maze 106; 14,058,028.2 mean FLOPs/maze |
| Maze-Hard frozen-recipe development check | Seeds 0–2; 4,000 updates on 900 training mazes, remaining 100 for development | 100/100 each; zero wrong stops and budget failures; all 300 saved predictions pass independent shortest-path checks |
| Graph coloring fresh models and hand-set controls | 8/10 vertices; deterministic CPU fits; 3 inference seeds; all 255/192 test graphs | Every paper quality cell reproduced; zero conflicts for lowest-energy selection from 20 samples |
| N-Queens fresh models | Sizes 8/10, seeds 0–2, complete 1,240/18,466 test rows; lowest-energy N20 | N20 accuracy 100% / 99.994585%; coverage 98.022561% / 93.568272%, close to paper values |
| Maze-Unique training objectives | All 12 objectives × seeds 0–2; 900 training, 100 development, 1,000 test mazes | All 36 runs solve 1,000/1,000; frozen published choices, no repeated local tuning |
| Sudoku difficulty curves (Figure 3a) | Three full 412,786-puzzle zero-energy searches after separate development gates; figure selects first 10,000 test puzzles | Full searches solve 412,776 / 412,771 / 412,770, with zero wrong stops. The figure subset solves 9,999 / 10,000 / 9,999; these subset percentages are not full-test accuracies. |
| Hand-set Sudoku | Seeds 0–2; complete 412,786-puzzle test set per seed | All 1,238,358 solved; zero wrong stops and budget failures; mean paper-convention FLOPs 108,460,051.10 |
| N-Queens free-pair and preconditioning controls | Sizes 8/10; seeds 0–2; complete 1,240/18,466 test rows | Free-pair N20 accuracy 100% / 98.221957%. At size 10, removing preconditioning lowers matched N20 accuracy from 99.994585% to 97.532402% and produces incorrect stops. |
| Table 7 objective comparison | All 84 fits and evaluations, including three full 3,840,000-update BPTT schedules | All required inputs validated. Sudoku full BPTT achieves 48.567333% ± 4.120049 points, versus published 48.82% ± 3.73. |
| Figure 4 training curves | Five objectives × 20 seeds; all declared probes, without schedule truncation | 100/100 curves; 9,740 probes; all 12 dense-clock validations and 228 overlapping checkpoint comparisons pass. |
| Figure 3b data efficiency | 17 training sizes × six seeds; seed-42 reference recreation; 32 updates; 10,000 test puzzles each | All 102 runs validated. Six-seed medians: k=1 and 3, 0%; k=10, 1.19%; k=30, 99.415%; k=100, 99.99%; k=300 through 10,000, 100%. |
| Appendix D.2 implicit differentiation | Three frozen learning rates × three seeds, 60,000 updates; development only | All nine complete; maximum 3/512 solved (0.5859375%), consistent with the published at-most-0.6% result. |
| Appendix F failed Maze-Hard ablations | Four controlled recipe reconstructions; original historical recipes/seeds unavailable | Token source solves 0/300; edge gates solve 0/100. Counts match, but these remain reconstructions. |

The maze predictions were also checked against an independent BFS/path checker. The Maze-Hard failure contains six extra cells; it is a real incorrect solution, not a label-parser disagreement. Source models, predictions, input hashes, settings and metrics are retained.

The full Sudoku audit verifies all puzzle identities, development gates, model hashes and per-puzzle status flags. Its paper-convention mean is **2,029,803,113.68 FLOPs per puzzle** (sample SD 43,310,405.89 across seeds), versus the published 2,015,770,227.88, a 0.6962% difference. The raw search proxy uses a narrower count and is retained separately. Evidence and no-rerun anytime curves are in `artifacts/sudoku_audit/learned-full-test-audit.json` and `learned-anytime-curves.json`.

Detailed reports: [main comparison with published baselines](docs/MAIN_COMPARISON.md), [paper coverage](docs/PAPER_COVERAGE.md), [implementation](docs/IMPLEMENTATION.md), [Sudoku results and operation counts](docs/SUDOKU_OPERATION_AUDIT.md), [stopping rules](docs/HALTING_REPRODUCTION.md), [mazes](docs/MAZE_REPRODUCTION.md), [N-Queens/graph coloring](docs/MULTI_SOLUTION_REPRODUCTION.md), and [learned couplings](docs/LEARNED_COUPLING_AUDIT.md). Fresh compute, learned/hand-set weights and Maze-Hard figure artifacts are in `artifacts/figures/`.

The [final objective report](docs/OBJECTIVE_REPRODUCTION.md) compares all objective cells with published results, the [Figure 4 audit](docs/FIGURE4_AUDIT.md) checks the complete probe schedules, and the [final controls audit](results/reproduction/final_controls_audit.json) checks difficulty searches, failed-ablation reconstructions, and long BPTT schedules. Standard one-step BPTT reaches 99.576% ± 0.343% on its 50,000-puzzle test, but its three-seed development mean peaks at 98.828%. Its local Table 7 time-to-99% is therefore absent. The separate twenty-seed Figure 4 experiment crosses 99% at 1.879 seconds. Test accuracy, development crossing, and different seed cohorts remain separate.

## Timing measurements

These are local Blackwell/Threadripper measurements. H200 paper times are not hardware-matched controls. Startup, warm repeats, training-loop-only time and complete inference-ready fit time remain separate in the raw records.

| Complete fit | Local warm median | Protocol |
|---|---:|---|
| BP AdamW baseline control | Published only | Outside local training scope; no matched local speed ratio |
| Sudoku, GPU, seed 0 | 35.40 ms | 1 cold + 7 warm fits; includes inference preparation; timed model equals the locally saved model |
| Sudoku, one CPU thread, seed 0 | 967.02 ms | Same GPU-generated digit permutations replayed; same sparse support; max tensor difference 2.3842e-7 |
| Maze-Unique, GPU | about 107.4 ms | 7 warm fits per seed; exact values in maze report |
| Maze-Unique, one CPU thread | about 38.6 ms | 7 warm fits per seed; exact values in maze report |

The Figure 2a comparison uses fresh CRAM inference counts and **published baseline sweeps** on the same first 10,000 held-out puzzles. Its tightest matched-accuracy ratio over the seven main baselines is **275.425× fewer FLOPs versus EBR**, close to but different from the paper's 282×. Applying the same matching code to the published CRAM curve gives 282.154×, recovering the paper's rounded value. The separately footnoted lattice/handwritten-search variant is retained in the detailed JSON and gives 237.9×. These are comparisons with imported baseline evidence. The “1,000× faster training” wall-time claim is not independently remeasured against local baselines, which are outside the confirmed scope.

## Completed data-efficiency study

Figure 3b comprises 17 training sizes × six seeds, each with 99 stored variants per original puzzle, 32 frozen fit updates, and a 10,000-puzzle search. At 06:38 UTC, all 102 measured runs validated, with no missing, invalid or degenerate rows. The [data-efficiency report](docs/DATA_EFFICIENCY_REPRODUCTION.md) and [collection.json](results/reproduction/data_efficiency/collection.json) retain every configuration, result and source identity. The final figure is available as [PNG](results/reproduction/data_efficiency/figure3b.png) and [PDF](results/reproduction/data_efficiency/figure3b.pdf), with an [independent final audit](results/reproduction/data_efficiency/final-audit.json).

All six one-puzzle models measured 0% accuracy after the full search budget; no result was imputed. The last three searches continued from archived checkpoints with all state fields checked before resuming, and their independent audits verify preservation of previously stopped puzzles and the complete original budget. The final curve uses the explicit seed-42 reference recreation, so its training rows are not identical to the paper's unavailable unseeded arrays. Published CRAM k=1 test records remain missing and are not filled with local measurements. Partial baseline fits from before scope clarification are retained as canceled evidence, not baseline results.

## Measured runtime improvements

Measured optimization work includes a 1.27–1.60× improvement from dense N-Queens mapping at size 8 and a 5.74–17.59× improvement in exact coverage postprocessing, both with equivalence checks. Packing two deferred-probe processes onto one GPU was 7.5% slower without MPS. A separately controlled private-MPS comparison was 1.799× faster with identical solved counts; eligible deferred probes used two-worker private MPS, while timed fits retained exclusive GPU use. Completed probe shards and private-daemon shutdown were independently checked. Replacing fixed scheduler sleeps with exit-aware waits reduced a measured 0.25-second child's dispatch completion from 15.00 to 0.26 seconds without changing the experiment.

For the three longest data-efficiency searches, a single-host dispatcher shards the original move kernel across GPUs while preserving global random keys, refresh/check logic and compaction. Bounded benchmarks measured 3.89066× on four Foureyes GPUs, 3.88046× on four Ripper GPUs, and 1.99083× on two Work GPUs. All tested state fields matched the original solver byte-for-byte; every production handoff separately checked all 24 fields before continuation. These are move-kernel throughput ratios, not end-to-end study speedups. Archived checkpoints, source snapshots, benchmark proofs and import receipts are retained in `artifacts/data_efficiency/migrations/20261114T0510Z/`. Resumed `search_seconds` preserves the original checkpoint clock plus continuation search time; it excludes staging and discarded work after the last checkpoint.

Live GPU/process snapshots: `python3 scripts/status.py --save`. Actual logs are under `artifacts/` and corresponding task-owned remote directories. The objective queues keep per-job commands, environment, hashes, progress and restart state.

## Exact-reproduction limits found in the release

- The original augmented TRM reference arrays used for the shared data-efficiency study are absent and were originally generated with unseeded randomness. A replacement using the pinned builder and explicit seed 42 and all six nested study datasets have been prepared and validated. Resulting runs are labeled as a protocol recreation rather than an exact match of the paper's training rows.
- FRM's adapter requires an absent campaign folder and source commit `6149a01`. The current public repository has a different single-commit history (`eb13b40`) and package API (`frm` instead of `flow_map_reasoning`); the documented commit cannot be fetched. The alternate repository named in model metadata is also inaccessible. Evidence is in `artifacts/frm_source_audit.json`; a newer release is not silently substituted.
- Baseline retraining is outside the confirmed scope. The source audit above explains why importing a newer FRM result would also be an invalid substitute for the published comparison.
- The difficulty exporter names an older zero-energy stopping variant, whereas the main table uses a fitted threshold. These protocols must not be silently interchanged.
- The released checkpoint evaluator uses some diagnostic metrics that differ from the paper tables: N-Queens/graph coloring majority selection differs from the paper's lowest-energy selection of 20 samples. The reproduction records both explicitly.
- Original H200 timing and exact GPU random trajectories cannot be claimed as locally reproduced on Blackwell hardware.
- The failed Maze-Hard recipes and original seed identities were not released. Controlled reconstructions match the stated counts but cannot establish exact historical reproduction. The unspecified plain-CRAM failure on Maze-Hard and beta=0.5 divergence remain published claims; a separately declared gradient diagnostic does not replace them.
- Saved Sudoku final records contain per-puzzle correctness/status flags rather than final boards. Dataset identity, settings, counts and flags are audited; independent board-by-board reconstruction from those final records is unavailable.

The reviewed [paper](references/2611.17309v1.md), [conversion audit](references/CONVERSION_REVIEW.md), [upstream pin](upstream_pin.json), [experiment inventory](configs/reproduction_manifest.json), source snapshots and data manifests preserve the provenance needed to distinguish these cases. The [consolidated completion receipt](results/reproduction/final_reproduction_audit.json) checks the cross-family completion gates and retained evidence hashes. The [rerun guide](docs/RERUN_GUIDE.md) separates saved-result verification from launching a fresh campaign.
