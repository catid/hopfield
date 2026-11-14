# Independent Sudoku result and operation-count audit

All three fresh learned runs and all three hand-set controls passed the full-record integrity audit. Learned CRAM correctly stops on 1,238,317 of 1,238,358 seed/puzzle evaluations, with 41 exhausted budgets and zero wrong stops. The hand-set control solves all 1,238,358. There are **412,786 distinct test puzzles**, each evaluated under three seeds. This audit recomputes the evaluator’s saved correctness flags and operation counts; it does not independently reconstruct every final Sudoku board.

| Method / provenance | Mean solved, % ± sample SD | Wrong stops / budget failures, pooled | Mean paper-count FLOPs/puzzle |
|---|---:|---:|---:|
| BP AdamW control: not measured; outside CRAM-only scope | — | — | — |
| Fresh trained CRAM | 99.996689 ± 0.000779 | 0 / 41 | 2,029,803,113.68 |
| Fresh hand-set CRAM | 100.000000 ± 0.000000 | 0 / 0 | 108,460,051.10 |
| Published trained CRAM reference | 99.995882 (rounded 99.996 in paper) | 0 / 51 | 2,015,770,227.88 |
| Published hand-set reference | 100.000000 | 0 / 0 | 108,229,308.46 |

Published rows come from the pinned release and are reference values. New learned mean operation count is 0.6962% higher than the reference; the new hand-set mean is 0.2132% higher. The 10-fewer-failures observation is a measured trajectory difference, not evidence of a statistically established accuracy improvement.

## Per-seed measurements

| Seed | Learned solved / 412,786 | Learned paper-count FLOPs/puzzle | Hand-set solved / 412,786 | Hand-set paper-count FLOPs/puzzle |
|---|---:|---:|---:|---:|
| 0 | 412,776 | 1,979,835,271.49 | 412,786 | 108,938,685.25 |
| 1 | 412,771 | 2,052,997,532.26 | 412,786 | 107,794,868.34 |
| 2 | 412,770 | 2,056,576,537.27 | 412,786 | 108,646,599.71 |

Every run has the unique, ordered IDs `10000..422785`. Each test signature matches its passed development gate, apart from split/offset/size fields. The learned model hash equals the corresponding fresh fitted checkpoint; temperatures stay `[0.40, 0.31, 0.45]` relative to the learned scale, with the training-fitted stopping threshold. Summary totals and distribution statistics were recomputed from per-puzzle arrays. The audit verifies the original `4e11` proxy budget and the endpoint rounded upward to a complete 4,096-move check.

## Two operation conventions

The raw evaluator’s `summary.mean_flop` is an older in-kernel proxy. It is not the final paper convention. Let `steps = raw_flops / (C × fl_move)`, where `C = 4`. The independent reconstruction is:

```text
paper_flops = steps × C × counted_move
              + counted_start
              + ceil(steps / 4096) × counted_check
```

Freshly counted learned components match the published components exactly: setup `8,546,006`, check `4,266,118`, and per-chain move `919` at neighbor count D=30 or `947` at D=31. Seed 0 has D=30 and seeds 1/2 have D=31. The independent reconstruction equals `cram.opcount.cram0_sudoku` for every puzzle. Fresh hand-set components are setup `142,811`, check `2`, and move `124`; these are measured on the unchanged CPU transcription and bookkeeping routines.

| Quantity | Learned seed mean | Hand-set seed mean |
|---|---:|---:|
| Raw proxy FLOPs/puzzle | 1,283,524,312.54 | 108,317,132.45 |
| Paper convention FLOPs/puzzle | 2,029,803,113.68 | 108,460,051.10 |

The paper’s arithmetic-only FullCounter charges 2 operations per multiply-add and 1 per elementary arithmetic or transcendental evaluation. Comparisons, selections, masks, indexing, data movement, casts, and random-number generation are excluded. Counts stop at the first successful move; actual kernels execute to the end of a check chunk. These logical operation totals are not elapsed-time or GPU-utilization measurements. The temperature schedule resets every 524,288 moves; it does not randomly restart the boards.

## Figure 2a and matched-accuracy comparison

The [fresh compute figure](../results/reproduction/figures/sudoku-compute-fresh.png), also available as PDF and SVG, uses only the first 10,000 held-out test puzzles (`10000..19999`), matching the published baseline sweep. CRAM’s curve is reconstructed from saved first-success counts, with no new model inference. Error bars span the minimum and maximum seed accuracy. Below the maximum cap, a run that has not stopped is conservatively counted as unsolved even if a returned intermediate board might be correct. At the full cap, the actual saved result is used.

Every baseline curve is imported from `upstream/results/compute-sweep/sudoku-extreme.json`, with no local baseline training or inference. Connected baseline points show the nondominated frontier of their fixed sweeps. The table uses each baseline’s highest mean accuracy and the cheapest point on the frozen CRAM cap grid that reaches at least that accuracy. These comparisons depend on grid resolution and the stated logical-count convention.

| Baseline / provenance | Published best accuracy, % | Baseline FLOPs / fresh CRAM FLOPs at matched accuracy |
|---|---:|---:|
| BP AdamW control: not measured; outside scope | — | — |
| EBR (imported published sweep) | 99.8633 | 275.4× |
| EqR (imported published sweep) | 99.4900 | 127,575.8× |
| FRM (imported published sweep) | 99.4200 | 4,730.8× |
| GRAM (imported published sweep) | 94.7367 | 12,916.5× |
| HRM (imported published sweep) | 57.0967 | 960.5× |
| TRM (imported published sweep) | 89.7233 | 722.1× |
| TRMP (imported published sweep) | 99.8700 | 2,408.0× |

The tightest comparison is EBR: published mean solve rate **99.8633%** at `5.41475e11` FLOPs/puzzle; fresh CRAM reaches **99.8867%** at `1.96596e9`, a **275.4×** ratio. The paper reports 282×. Thus the measured saved-curve comparison is close, but does not reproduce the 282× number exactly. The separately footnoted TRMP-fast lattice/handwritten-search variant is retained in the JSON at 237.9×; it is outside the seven-main-baseline figure and minimum-ratio claim.

As a check on the matching rule, applying it unchanged to the released CRAM curve gives **282.1537×**, reproducing the paper's rounded 282×. The released and fresh matching points use the same cap, `177827941003.89227`; released CRAM has 99.9100% accuracy and mean `1,919,076,937.59` FLOPs there, versus fresh 99.8867% and `1,965,959,514.62`. The difference is therefore in the measured trajectories and counts at that cap. The comparison JSON retains both points, the validation result, and the released curve's SHA256.

## Artifacts and commands

The complete audits are [learned-full-test-audit.json](../results/reproduction/sudoku_audit/learned-full-test-audit.json) and [hand-full-test-audit.json](../results/reproduction/sudoku_audit/hand-full-test-audit.json). Component measurements are [sudoku_operation_counts.json](../results/reproduction/sudoku_operation_counts.json) and [cram-hand-sudoku.json](../results/reproduction/sudoku_audit/cram-hand-sudoku.json). The [anytime curves](../results/reproduction/sudoku_audit/learned-anytime-curves.json) keep full-set and first-10k subsets separate; [sudoku-compute-comparison.json](../results/reproduction/figures/sudoku-compute-comparison.json) retains exact matching points and source hashes.

```sh
CUDA_VISIBLE_DEVICES='' upstream/.venv/bin/python scripts/audit_sudoku_results.py learned
CUDA_VISIBLE_DEVICES='' upstream/.venv/bin/python scripts/audit_sudoku_results.py hand
artifacts/plotting-venv/bin/python scripts/render_sudoku_compute.py
```

All exports are redirected to task-owned artifacts. Tracked upstream source and published results remain unchanged.
