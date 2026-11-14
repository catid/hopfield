# Multi-solution reproduction

Offset-shared N-Queens and graph-coloring training, released-checkpoint checks, and hand-set controls are complete for sizes 8 and 10 and inference seeds 0, 1, and 2. Every graph-coloring Table 3 quality cell matches the published per-seed values exactly. N-Queens reproduces the reported performance closely, with small differences retained rather than reported as exact matches. This report covers fresh CRAM and hand-set controls, with clearly labeled published neural baselines for comparison. Baseline retraining is outside the confirmed scope.

The source is pinned to `62bbe830c7d6f87f2ba18f30a28039258b986454`. All figures below are fresh measurements unless explicitly labeled “paper” or “released checkpoint.” Standard deviations are sample standard deviations across the three seeds. Rows labeled “published” are imported from `upstream/results/multi-solution-table.json`, with no claim of local baseline reproduction.

The released N-Queens free-pair architecture controls and matched preconditioning ablations are also complete: validation-selected MPL at size 8 and CD at size 10 for free pairs. Its exact configurations are imported from the pinned published validation record; no test tuning is performed.

## N-Queens results

For CRAM and EBR, N1 is the first chain and N20 chooses the chain with the lowest energy. The published TRM, HRM, and GRAM rows instead use majority vote at N20, as the paper does. Coverage uses all valid completions among 20 chains, capped at `min(number of completions, 20)`, averaging duplicate rows within each input before pooling over distinct inputs.

| Size | Model / provenance | N1 accuracy, % | N20 accuracy, % | Cov@20, % |
|---|---|---:|---:|---:|
| 8 / 10 | BP AdamW control: not measured; outside CRAM-only scope | — | — | — |
| 8 | TRM (published; not locally reproduced) | 67.392473 ± 6.835520 | 67.392473 ± 6.835520 | 51.424731 ± 3.062747 |
| 8 | HRM (published; not locally reproduced) | 84.408602 ± 2.179410 | 84.408602 ± 2.179410 | 58.790323 ± 0.564516 |
| 8 | GRAM (published; not locally reproduced) | 99.435484 ± 0.213367 | 100.000000 ± 0.000000 | 97.920859 ± 0.210190 |
| 8 | EBR (published; not locally reproduced) | 100.000000 ± 0.000000 | 99.946237 ± 0.093121 | 98.091398 ± 0.202953 |
| 10 | TRM (published; not locally reproduced) | 17.50† | 17.50† | 3.292043 ± 3.649970 |
| 10 | HRM (published; not locally reproduced) | 37.80† | 37.80† | 10.664732 ± 3.990865 |
| 10 | GRAM (published; not locally reproduced) | 88.911152 ± 3.056615 | 99.981949 ± 0.008272 | 92.878901 ± 0.319990 |
| 10 | EBR (published; not locally reproduced) | 99.310444 ± 0.275616 | 99.989169 ± 0.005415 | 92.480378 ± 0.658062 |
| 8 | Fresh learned, offset-shared | 100.000000 ± 0.000000 | 100.000000 ± 0.000000 | 98.022561 ± 0.135313 |
| 8 | Paper learned reference | 99.946237 ± 0.046561 | 100.000000 ± 0.000000 | 97.932625 ± 0.061284 |
| 8 | Released checkpoint, locally evaluated | 99.973118 ± 0.046561 | 100.000000 ± 0.000000 | 98.029655 ± 0.146299 |
| 8 | Fresh hand-set evaluation | 100.000000 ± 0.000000 | 100.000000 ± 0.000000 | 98.421936 ± 0.013257 |
| 8 | Paper hand-set reference | 100.000000 ± 0.000000 | 100.000000 ± 0.000000 | 98.392953 ± 0.023346 |
| 10 | Fresh learned, offset-shared | 99.451244 ± 0.081471 | 99.994585 ± 0.000000 | 93.568272 ± 0.041092 |
| 10 | Paper learned reference | 99.445828 ± 0.052036 | 99.994585 ± 0.000000 | 93.552018 ± 0.064604 |
| 10 | Released checkpoint, locally evaluated | 99.384454 ± 0.054244 | 99.994585 ± 0.000000 | 93.552680 ± 0.056905 |
| 10 | Fresh hand-set evaluation | 99.990974 ± 0.006253 | 100.000000 ± 0.000000 | 95.526056 ± 0.025042 |
| 10 | Paper hand-set reference | 99.992780 ± 0.003127 | 100.000000 ± 0.000000 | 95.542050 ± 0.026639 |

† For size-10 TRM and HRM accuracy, Table 3 substitutes the original authors’ 17.50% and 37.80% because the CRAM paper’s baseline reproductions scored lower (6.90% and 22.72%). Their displayed coverage still comes from the CRAM paper’s reproductions. These mixed-provenance reference cells are preserved and none is a local measurement here.

There are 1,240 test rows / 772 distinct inputs at size 8 and 18,466 rows / 6,513 distinct inputs at size 10. These rows include multiple labels for the same partial puzzle; they are not all independent puzzles. All locally evaluated offset-shared learned/released and hand-set chains have zero wrong stopping certificates. Some chains hit the move cap; their saved final boards still participate in N1/N20 and coverage exactly as in the official exporter. Per-seed counts and alternative per-row coverage are retained in [nqueens-paper-results.json](../results/reproduction/multi_solution/nqueens-paper-results.json).

The released `checkpoints/evaluate.py` diagnostic compares majority-vote N20 and GRAM row-mean coverage, which differ from Table 3’s lowest-energy N20 and EBR-style per-puzzle coverage. Its N-Queens exact-record checks reported small mismatches against the original H200 records, including with unchanged released weights. Those failures and logs are preserved. Fresh fits passed source correctness checks and used the unchanged frozen recipe; test outcomes did not change any hyperparameter. Hardware, CUDA numerics, or random trajectories could explain the differences, but this run does not isolate their cause.

## Graph-coloring results

The fresh CRAM and hand-set values below exactly equal the paper export, including every per-seed quality cell. “Conflicts” is total graph-edge conflicts across the test graphs, not a percentage. N20 selects the lowest-energy chain for CRAM and EBR, and majority vote for the imported TRM, HRM, and GRAM rows. Table 3 uses the source's strict `coverage_20` on the already canonical saved predictions. The separately relabeled `coverage_canonicalized_20` is a diagnostic and is not the Table 3 metric.

| Vertices | Model | N1 conflicts, seeds 0 / 1 / 2 | N20 conflicts, seeds 0 / 1 / 2 | Cov@20, % |
|---|---|---|---|---:|
| 8 / 10 | BP AdamW control: not measured; outside CRAM-only scope | — | — | — |
| 8 | TRM (published; not locally reproduced) | 134 / 117 / 103 | 134 / 117 / 103 | 21.820175 ± 0.759671 |
| 8 | HRM (published; not locally reproduced) | 152 / 134 / 133 | 152 / 134 / 133 | 20.577485 ± 0.352473 |
| 8 | GRAM (published; not locally reproduced) | 1 / 3 / 6 | 2 / 4 / 4 | 87.902047 ± 0.562676 |
| 8 | EBR (published; not locally reproduced) | 0 / 0 / 0 | 0 / 8 / 0 | 88.888889 ± 0.063306 |
| 10 | TRM (published; not locally reproduced) | 151 / 191 / 165 | 151 / 191 / 165 | 6.861096 ± 0.508905 |
| 10 | HRM (published; not locally reproduced) | 202 / 179 / 167 | 202 / 179 / 167 | 6.225463 ± 0.890326 |
| 10 | GRAM (published; not locally reproduced) | 2 / 148 / 8 | 3 / 149 / 6 | 37.838848 ± 24.665036 |
| 10 | EBR (published; not locally reproduced) | 0 / 0 / 1 | 0 / 0 / 2 | 54.832679 ± 0.212335 |
| 8 | Fresh learned, free pair tables | 0 / 0 / 0 | 0 / 0 / 0 | 88.852339 ± 0.456506 |
| 10 | Fresh learned, free pair tables | 1 / 0 / 0 | 0 / 0 / 0 | 52.570574 ± 0.617787 |
| 8 | Hand-set | 0 / 0 / 0 | 0 / 0 / 0 | 90.643275 ± 0.633060 |
| 10 | Hand-set | 0 / 0 / 0 | 0 / 0 / 0 | 55.150495 ± 0.360578 |

The test sets contain 255 graphs at size 8 and 192 at size 10. There are zero wrong stops in all learned and hand-set evaluations. Fresh CPU-fitted weights differ from released weights by at most `7.153e-6`, while all quality metrics still match. Released-checkpoint diagnostics also match exactly at both sizes and all seeds. See [graph-coloring-paper-results.json](../results/reproduction/multi_solution/graph-coloring-paper-results.json) and [graph-coloring-fresh-validation.json](../results/reproduction/multi_solution/graph-coloring-fresh-validation.json).

## Published Table 3 parameters and fit times

These are the printed cells of [Table 3](../references/2611.17309v1.md#S4.T3), retained as **published-only** comparisons. Parameters are listed in task order: N-Queens 8, N-Queens 10, graph coloring 8, graph coloring 10. Each task trains its own model. Baseline fit times are single H200 runs; CRAM fit times are the seed-0 median of seven warm H200 fits. Local Blackwell timings appear separately below and do not establish a matched hardware speedup.

| Method (published only) | Parameters, by task | N-Queens 8 fit | N-Queens 10 fit | Coloring 8 fit | Coloring 10 fit |
|---|---|---:|---:|---:|---:|
| BP AdamW control | Outside local training scope | — | — | — | — |
| HRM | 27.3M each | 59 min | 4.0 h | 41 min | 1.7 h |
| TRM | 6.8M each | 1.3 h | 5.0 h | 54 min | 2.1 h |
| GRAM | 10.9M each | 1.3 h | 5.2 h | 53 min | 2.2 h |
| EBR | 7.1M each | 3.2 h | 7.6 h | 4.5 h | 7.6 h |
| CRAM | 113 / 181 / 869 / 475 | 2.9 s | 6.8 s | 1.3 s | 0.3 s |
| Hand-set CRAM | 2 / 2 / 1 / 1 distinct values | none | none | none | none |

## Frozen protocols

**N-Queens:** offset-shared Ising energy with 113 / 181 learned parameters, zero initialization, pseudolikelihood, 4,000 full-batch SGD updates, momentum 0.5, cosine learning rate 30, per-offset gradient preconditioning, and no L1. Training has 7,224 / 108,234 rows. The frozen `nqueens-offset-n10.json` recipe applies unchanged at both sizes. Twenty-chain inference uses local-margin stopping, temperature `1.0 × u`, and a 20,000-move cap. In the source, `u` is the training-row-weighted mean absolute two-state field difference over solution cells; `delta` is the minimum alternative-minus-own field over training solution cells. The paper leaves these scalar definitions implicit.

Appendix C.5 explicitly acknowledges that the paper chose to report the transferred size-10 recipe at size 8 after seeing its original test results. This reproduction fixes that published choice before local evaluation and also reports the original size-8 validation-selected MPL/table control separately; it does not make a new choice using local test scores.

**Graph coloring:** unrestricted affine 3×3 pair tables reading all adjacency tokens; 7,332 / 18,660 full coefficients, of which 869 / 475 are nonzero. The official scored model is fitted on one CPU thread from zero using pseudolikelihood, SGD momentum 0.5, cosine learning rate 10, proximal L1 `1e-4` / `1e-3`, and 1,024 / 256 updates over 7,002 / 13,465 training rows. CPU search uses 20 chains, term-max stopping, temperature `2 × hbar` / `3 × hbar`, threshold `0.45`, and a 10,000-move cap. Separately fitted GPU models serve only the GPU timing measurement.

Hand-set energies follow the official validation-only temperature selection before test evaluation. Each final test run was executed once. All reported metrics were recomputed from its saved chain predictions, without new inference. N-Queens fresh fits are identical across training seeds in canonical tensor bytes because fitting is deterministic full-batch training; inference seeds differ. Serialized checkpoint hashes can differ despite identical tensors because the released save helper embeds UUID-named temporary archives. Both file hashes and canonical tensor hashes are retained.

## Measured timing and optimization

The main offset-shared N-Queens fits and the GPU timings in this section used ripper RTX PRO 6000 Blackwell Max-Q Workstation GPUs. Graph-coloring scored fits used one foureyes AMD Ryzen Threadripper PRO 9985WX thread. The additional free-pair and preconditioning runs below used foureyes GPUs, with per-run host/device provenance retained separately. Runtime: Python 3.13.13, PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0. Cold is a fresh-process complete fit; warm is the median of seven reinitialized complete fits. The paper uses seed 0’s dense-map warm median. File reads and evaluation are outside that fit timing boundary; compilation and preparation inside the fit are included.

The timing tables compare implementations of each fixed pseudolikelihood fit. BP AdamW timing and accuracy controls are not available from this subtask; they cannot be inferred from these implementation comparisons.

| N-Queens size | Seed | Index cold / warm, s | Dense cold / warm, s | Index warm / dense warm |
|---|---:|---:|---:|---:|
| 8 | 0 | 8.5666 / 1.6010 | 3.2753 / 1.2602 | 1.270× |
| 8 | 1 | 2.7024 / 2.0431 | 2.1762 / 1.2772 | 1.600× |
| 8 | 2 | 2.9227 / 1.6234 | 1.9389 / 1.2727 | 1.276× |
| 10 | 0 | 10.1870 / 9.3891 | 10.0829 / 9.4247 | 0.996× |
| 10 | 1 | 10.7617 / 9.6536 | 10.2679 / 9.4378 | 1.023× |
| 10 | 2 | 10.5391 / 9.6389 | 10.1189 / 9.4226 | 1.023× |

The dense mapping improves size-8 warm fit time by 1.27–1.60× on the same GPU. At size 10 it gives no material improvement (0.996–1.023×). Dense/index coefficient discrepancies remain below `9.537e-7`; all official comparison checks pass. The scored model comes from the indexed fit, and dense timing does not rescore the test set. The paper’s H200 dense warm times are 2.89625 s / 6.75241 s, so cross-hardware ratios cannot be attributed purely to software changes.

| Graph-coloring size | CPU cold / warm, s | Local GPU cold / warm, s | Paper H200 GPU warm, s |
|---|---:|---:|---:|
| 8 | 33.51168 / 33.40507 | 1.62653 / 0.52691 | 1.27019 |
| 10 | 36.70621 / 37.99935 | 0.81460 / 0.14985 | 0.34430 |

Every one of the seven graph-coloring warm fits reproduces its own cold model exactly. Independent jobs ran on other GPUs/CPU cores, so the timings are local measurements under recorded workload conditions, not isolated H200 replications.

For exact N-Queens coverage postprocessing, enumerating all 92 / 724 complete solutions once and filtering them against each partial input reduces the all-input counting time by 5.74× / 17.59×, including initial enumeration. Every count agrees with stored counts, input multiplicities, and the source’s full independent enumeration over all 772 / 6,513 distinct test inputs. This optimization changes only metric computation, not inference. Evidence: [nq-metric-count-validation.json](../results/reproduction/multi_solution/nq-metric-count-validation.json).

## Reproducible evidence

The complete protocol is [protocol.json](../results/reproduction/multi_solution/protocol.json). Fitting and scoring commands are retained in [run_nqueens_main.sh](../scripts/run_nqueens_main.sh), [run_graph_coloring_main_cpu.sh](../scripts/run_graph_coloring_main_cpu.sh), and [run_graph_coloring_fit_gpu.sh](../scripts/run_graph_coloring_fit_gpu.sh). Before N-Queens execution, install the frozen n10 recipe in both `runs/NQ/nq11/n10/selected-pl-ising.json` and `runs/NQ/nq11/n10/selected-headline.json`; size-10 final mode requires the former, which the reproduction prose does not explicitly mention. These are copies of the unchanged published recipe, not retuning.

Raw N-Queens outputs are under `upstream/experiments/runs/NQ/nq11/n{8,10}/` and `upstream/experiments/runs/NQ/alt/n{8,10}/hand/`; released-checkpoint chain records are under `artifacts/multi_solution/released/`. Raw graph-coloring outputs are under `upstream/experiments/runs/CRAM-GC/`. Logs and validations are under `artifacts/multi_solution/`. Regenerate quality summaries without rerunning inference:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  upstream/.venv/bin/python scripts/prepare_nqueens_metric_counts.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  upstream/.venv/bin/python scripts/summarize_nqueens_results.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  upstream/.venv/bin/python scripts/summarize_graph_coloring_results.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  upstream/.venv/bin/python scripts/validate_nqueens_fit_artifacts.py
```

The deployment manifest covers all four authorized hosts. The 352-file tracked-source canonical digest is `8e07410763075a1207a8a4c9c69c473cff12ebd324a382ccd59e5d90d4bc7450`; the 125-file, 3,206,443,413-byte data digest is `f2140800598a03c4021c741f2e8c1c6906cd2c7253086e4e745440e7d1627d79`. Per-file hashes are in `artifacts/deployment/source_manifest.json` and `data_manifest.json`. The pinned environment lock SHA256 is `fdef606f276fefc325c1f7de3110c6032172d2532057348f4af43b3715ad1d09`. Tracked upstream source and published result records remain unchanged.

<!-- additional-cram-controls -->
## Additional CRAM architecture and preconditioning controls

All six free-pair runs, six no-preconditioning ablations, and three size-8 locally selected controls are complete. The size-10 preconditioned control reuses the already completed main run. Final test sets were scored once per run; subsequent metrics use saved chain outputs.

### Free pair tables

The release includes a free-pair N-Queens architecture control in its comparison export; it is not an additional visible row of the paper's rendered Table 3. The frozen validation-selected recipes are MPL at size 8 (learning rate 10, masking 0.1, no L1, 10,000 updates) and CD at size 10 (learning rate 100, 100 negative-sample moves, no L1, 2,000 updates). Both use momentum 0.5, cosine decay, local-margin stopping, temperature 3 times the fitted unit, 20 chains, and a 20,000-move cap. Source fit and search routines are unchanged, with TF32 disabled.

| Size / model | N1 accuracy, % | Lowest-energy N20 accuracy, % | Per-puzzle Cov@20, % |
|---|---:|---:|---:|
| BP AdamW control: not measured; outside CRAM-only scope | — | — | — |
| 8, fresh MPL free pair tables | 99.758065 ± 0.080645 | 100.000000 ± 0.000000 | 94.221753 ± 0.146903 |
| 8, imported released reference | 99.784946 ± 0.046561 | 100.000000 ± 0.000000 | 94.074106 ± 0.172283 |
| 10, fresh CD free pair tables | 91.454565 ± 1.905583 | 98.221957 ± 0.122814 | 87.003914 ± 0.152138 |
| 10, imported released reference | 91.952778 ± 0.631789 | 98.223763 ± 0.192531 | 87.116948 ± 0.246084 |

The local-margin stopping rule is not a general correctness certificate for these weaker free-pair fits:

| Size | Wrong stopped chains, seeds 0 / 1 / 2 | Capped chains, seeds 0 / 1 / 2 |
|---|---|---|
| 8 | 4 / 8 / 8 | 61 / 64 / 67 |
| 10 | 23295 / 24358 / 35529 | 3639 / 3920 / 3365 |

Cold fits were timed for every seed; only seed 0 ran the seven reinitialized warm fits used for the publication's timing boundary. This skips fourteen unnecessary timing-only fits per architecture while preserving all three quality runs. Warm repeats are checked against the scored cold model; per-tensor differences are retained in `fit.json`. H200 references and current Blackwell times are hardware-specific measurements, not isolated software speedups.

| Size | Local cold fit, seeds 0 / 1 / 2, s | Local seed-0 warm median, s | Published H200 seed-0 warm, s |
|---|---:|---:|---:|
| 8 | 5.0043 / 4.7885 / 4.0659 | 3.3534 | 8.7832 |
| 10 | 140.8112 / 132.1078 / 137.8586 | 146.8137 | 115.0617 |

### Matched preconditioning ablation

The published size-8 ablation uses the locally selected MPL/table energy, with masking 0.3 and learning rate 30; this differs from the transferred size-10 PL/Ising recipe used in the main size-8 table. We therefore reran its own preconditioned MPL/table control before comparing the ablation. Size 10 uses the same PL/Ising recipe as the main result. Each pair differs only by the per-offset gradient preconditioner; there are 4,000 updates, no L1, and no test tuning.

| Size / matched recipe | Preconditioning | N1 accuracy, % | Lowest-energy N20, % | Per-puzzle Cov@20, % |
|---|---|---:|---:|---:|
| BP AdamW control: not measured; outside CRAM-only scope | — | — | — | — |
| 8 / MPL table | on | 100.000000 ± 0.000000 | 100.000000 ± 0.000000 | 98.216056 ± 0.094252 |
| 8 / MPL table | off | 100.000000 ± 0.000000 | 100.000000 ± 0.000000 | 98.266545 ± 0.039864 |
| 10 / PL ising | on | 99.451244 ± 0.081471 | 99.994585 ± 0.000000 | 93.568272 ± 0.041092 |
| 10 / PL ising | off | 73.796888 ± 0.313218 | 97.532402 ± 0.027789 | 89.072378 ± 0.065196 |

The imported ablation records use diagnostic majority-vote N20 and GRAM row-mean coverage. Those are also retained and compared only against the matching diagnostic values in [nqueens-preconditioning-results.json](../results/reproduction/multi_solution/nqueens-preconditioning-results.json); the table above uses the paper's headline metric definitions recomputed from the saved chains.

Size-10 no-preconditioning produces 96564 / 96585 / 96747 wrong stopped chains across seeds 0 / 1 / 2, out of 369,320 chains per seed. The matched preconditioned control has zero wrong stops. Both size-8 controls also have zero wrong stops and retain 100% N1/N20 accuracy; the large quality loss is specific to size 10 in this experiment.

For seed 0 at size 10, the longest-diagonal interaction contrast is 2.528850 with preconditioning and 0.140220 without it; the shortest-diagonal values are 2.369900 and 5.031328. The respective offsets occur in 2 versus 162 directed cell pairs. This directly measures the paper's explanation that rare long-diagonal interactions barely train without normalization by offset frequency.

The [fresh diagonal-cost plot](../results/reproduction/figures/nqueens-preconditioning-fresh.png), also retained as PDF and SVG, shows the mean and sample SD over all three seeds for each matched control.

Commands and evidence: `scripts/run_nqueens_free_reproduction.py --n N --seed S`; `scripts/run_nqueens_preconditioning.py --n N --seed S --precond on|off`; raw logs in `artifacts/main_followups/`; per-run source/recipe/model hashes in `artifacts/multi_solution/*-provenance.json`. The complete machine-readable free-pair fit bundle is [free-pair-fit-results.json](../results/reproduction/multi_solution/free-pair-fit-results.json). Final aggregation and this report section are reproduced by `scripts/finalize_nqueens_controls.py`.
