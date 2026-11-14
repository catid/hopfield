# Difficulty and Maze-Hard controls

Figure 3a, all four Maze-Hard failed-ablation reconstructions, and the three full long-BPTT controls are complete. The independent [final control audit](../results/reproduction/final_controls_audit.json) recomputes the saved results and verifies their provenance. It took 17.04 seconds on CPU and did not rerun training or inference.

## Figure 3a: full searches and plotted subset

Each seed ran the complete 412,786-puzzle Sudoku test search with zero-energy stopping, `E <= 1e-3`. Table 2 instead uses the fitted threshold, `E <= tau + 1e-3`; its fresh thresholds are −4.930538e−6, −3.105553e−6 and −1.650831e−4. These remain separate experiments and records.

The full zero-stop searches use `full-all` rows 10000–422785, inclusive:

| Seed | Solved / 412,786 | Solved, % | Wrong stops | Budget failures | Mean paper-convention FLOPs |
|---|---:|---:|---:|---:|---:|
| 0 | 412,776 | 99.997577437 | 0 | 10 | 1,979,835,271.492478 |
| 1 | 412,771 | 99.996366156 | 0 | 15 | 2,052,997,532.262601 |
| 2 | 412,770 | 99.996123900 | 0 | 16 | 2,056,576,537.270716 |

Across seeds, full-test accuracy is **99.996689164% ± 0.000778745 percentage points** (sample SD), with 41 total budget failures and zero wrong stops. Mean FLOPs across the three seed means are **2,029,803,113.675265**. These are 1,238,358 seed-puzzle evaluations of the same 412,786 distinct puzzles.

Figure 3a selects only `full-all` rows 10000–19999 from those completed full searches:

| Seed | Solved / 10,000 | Solved, % | Wrong stops | Budget failures | Mean paper-convention FLOPs |
|---|---:|---:|---:|---:|---:|
| 0 | 9,999 | 99.99 | 0 | 1 | 2,219,208,881.3534 |
| 1 | 10,000 | 100.00 | 0 | 0 | 2,174,774,616.4944 |
| 2 | 9,999 | 99.99 | 0 | 1 | 2,032,761,665.6508 |

The plotted subset's mean accuracy is **99.993333333% ± 0.005773503 percentage points**; its mean FLOPs are **2,142,248,387.832867**. The 99.99% / 100% / 99.99% values describe this 10,000-puzzle subset, not the full test set. The hardest/easiest difficulty-bin median compute ratio is **19.481069×**. Bin counts are 1,564 / 690 / 593 / 757 / 1,428 / 2,274 / 2,118 / 576 for ratings 0, 1, 2–3, 4–7, 8–15, 16–31, 32–63 and 64+.

The audit checks all full-test identities and arrays, recomputes summary counts and FLOPs, validates the development gates and model/source hashes, and checks every Figure 3a bin and all 121 compute-budget points per seed. The zero-stop and separately executed fitted-threshold runs happened to produce identical saved correctness and FLOP arrays on this hardware; their elapsed times differ. This observed equality does not make the stopping protocols interchangeable.

All searches retain four chains, the original budget, 4,096-move checking interval, temperature schedule and seeds. A standalone 10,000-puzzle search would change the random stream because compaction changes active-chain positions. The full-run-first procedure preserves the released scheduling. Each seed's 10,000-puzzle development gate solved all 10,000 with no wrong stop. Full-test records retain status flags and operation counts, but no final Sudoku boards, so the audit does not independently establish board validity.

Evidence: [Figure 3a JSON](../results/reproduction/difficulty/figure3a.json), [PNG](../results/reproduction/difficulty/figure3a.png), [PDF](../results/reproduction/difficulty/figure3a.pdf), and [finalization hashes](../results/reproduction/difficulty/finalization.json). Published CRAM/FRM/EBR curves remain labeled references. The original search command was `scripts/run_sudoku_difficulty.py --seed SEED --test-size full`; the CPU export is `scripts/export_sudoku_difficulty.py --plot`.

## Maze-Hard failed-ablation reconstructions

Appendix F reports 0/300 development solutions for a learned input per token and 0/100 for independent pair gates. The release provides both model options but only the successful recipe. It omits the historical failed recipes, logs, checkpoints and seed identities. Exact reproduction of those historical runs is unavailable.

The completed reconstructions change only `source='token'` or `coupling='gate'` in the successful recipe. Every run performs 4,000 updates on the same 900 training mazes and evaluates the same 100 development mazes. The split is reproduced from `numpy.random.default_rng(0).permutation(1000)`. No test data or outcome-based tuning is used. The selected reconstruction seeds are explicit assumptions.

| Reconstruction | Seed | Solved / 100 | Wrong stops | Budget failures | Complete fit, seconds |
|---|---:|---:|---:|---:|---:|
| Token source | 0 | 0 | 60 | 40 | 30.449925 |
| Token source | 1 | 0 | 46 | 54 | 31.863918 |
| Token source | 2 | 0 | 45 | 55 | 28.179250 |
| Independent edge gates | 0 | 0 | 65 | 35 | 12.096221 |

Token source totals **0/300**, with 151 wrong stops and 149 budget failures. Edge gates gives **0/100**, with 65 wrong stops and 35 budget failures. Both match the published solved counts while remaining controlled reconstructions. Token source has 32 effective trainable parameters (30 nonzero); edge gates has 27 (all nonzero). These local Blackwell cold-fit times do not establish an H200 speed comparison.

The token-source adjoint uses the released PyTorch fallback; the edge-gate model uses fused latent kernels. All four runs completed after their assigned long-BPTT fit **and** final test, as verified by queue dependency hashes. The audit confirms exact single-field recipe changes, split identities, complete update schedules, model/source/result hashes and per-maze status counts. Development records retain correctness flags and FLOPs but no final boards.

Evidence: [ablation summary](../results/reproduction/maze_hard_ablations/summary.json), raw records under `artifacts/maze_hard_ablations/{token-source,edge-gates}/`, and [final audit](../results/reproduction/final_controls_audit.json). Commands: `scripts/run_maze_hard_ablations.py --variant token-source --seed SEED` and `--variant edge-gates --seed 0`.

The separate statement that plain CRAM solves none of the Maze-Hard **test** mazes lacks a frozen fit/search recipe. Neither a Maze-Unique transfer evaluation nor newly selected Maze-Hard settings would reproduce that unspecified experiment. It remains a published limitation claim.

## Full long-BPTT control

All three full-BPTT runs completed 3,840,000 updates with learning rate 2, unroll length 10 and L1 strength 0.1. Their development-selected checkpoints were evaluated on 50,000 held-out test puzzles, rows 10000–59999. This control is distinct from Figure 4's 60,000-update, learning-rate-60 runs.

| Seed | Completed updates | Selected update | Solved / 50,000 | Solved, % | Wrong stops | Budget failures |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3,840,000 | 3,840,000 | 26,370 | 52.740 | 23,581 | 245 |
| 1 | 3,840,000 | 2,965,821 | 22,251 | 44.502 | 27,716 | 138 |
| 2 | 3,840,000 | 3,840,000 | 24,230 | 48.460 | 25,724 | 186 |

Mean accuracy is **48.567333% ± 4.120049 percentage points**. The audit reran the CPU collector's validation, rehashed and parsed all 11,520,000 update records, checked selection from the development curves, and validated checkpoint/source/final-test hashes and all 50,000 saved FLOP counts per seed. Some correct returned boards did not halt (196 / 105 / 140), so solved counts are not simply halted minus wrong stops. Per-puzzle correctness flags and final boards were not retained by this evaluator; the independent audit is limited to the recorded aggregate accounting and FLOP arrays.

Evidence: [validated collector output](../results/reproduction/long_bptt/collected/summary.json). Regenerate all three control audits without inference: `python3 scripts/audit_final_controls.py`.
