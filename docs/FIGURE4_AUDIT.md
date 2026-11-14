# Figure 4 protocol and completed results

All 100 Sudoku curve runs in `configs/figure4_jobs.json` completed their full schedules. The final exporter declared this objective family complete on 2026-11-14 at 00:39 UTC, with no validation errors or truncated schedules. The independent final audit recomputed the curve summaries and checked all 9,740 probes, all 20 seeds per objective, and all 12 dense-clock matches covering 228 shared checkpoints. It also rechecked the collected output hashes. The PNG was inspected visually: all five curves, percentile bands, axes, legend and four 99% crossing markers are legible. This completes Figure 4 and its objective-family dependencies, not the broader paper reproduction.

The final results and published comparisons are in `artifacts/objective_studies/FINAL_REPORT.md`; machine-readable values are in `figure4-comparison.csv` beside it. Exported figures are `artifacts/figure4/figures/figure4.{png,pdf,svg}`, with every measured curve retained in `artifacts/figure4/curves.json`. Final validation evidence is `artifacts/objective_studies/final-validation.json`.

| Objective | Local 20-seed mean T99 (s) | Published H200 T99 (s) | Local final development mean (%) |
|---|---:|---:|---:|
| BP AdamW | Outside this protocol; not run | Not part of Figure 4 | — |
| Pseudolikelihood | 0.086218 | 0.048 | 99.687500 |
| Minimum probability flow | 0.130071 | 0.21 | 99.287109 |
| One-step BPTT | 1.879032 | 4.4 | 99.052734 |
| Contrastive divergence | 16.512295 | 33 | 99.726563 |
| Full BPTT, 60,000 updates | Never | Never | 26.748047 |

The reported ordering is reproduced. Full BPTT ends near the published 27% development mean. Published values are quoted references, not measurements from these GPUs; the local clocks combine documented Blackwell hosts and the declared reconstructed interpolation/readiness policy below. The 20-seed Figure 4 one-step-BPTT crossing differs from its three-seed Table 7 result: the latter's development mean peaks at 98.828125%, despite its 99.576% mean test accuracy, so Table 7 has no local T99.

The original four lanes (`loki:0`, `loki:1`, `work:0`, `work:1`) waited behind Table 7 and the nine D.2 implicit-long fits. On 2026-11-13, the freed `ripper:0` and `ripper:1` devices began deferred Figure 4 work; the initial six-lane allocation had 17, 17, 17, 18, 15 and 16 jobs. Subsequent guarded migrations used released ripper and foureyes capacity. That initial layout and its per-lane command lists are historical, while the current config and migration receipts record the final ownership. Workers acquired each lane's physical-GPU lock and rechecked compute owners.

The source is CRAM commit `62bbe830c7d6f87f2ba18f30a28039258b986454`. Five missing `plan-dense-sudoku-*.txt` inputs are materialized in the clean source from the frozen manifest. They record published choices, not locally repeated hyperparameter searches. The worker verifies source, all 20 training splits, the development split, and the three fresh main-Sudoku checkpoints used to compute the common search-temperature ratio.

| Objective | Learning rate | Method parameter | L1 multiplier | Updates | Probes per full run |
|---|---:|---:|---:|---:|---:|
| Pseudolikelihood | 540 | none | 1 | 32 | 32 |
| Minimum probability flow | 60 | none | 1 | 2,000 | 104 |
| Contrastive divergence | 6 | 324 Gibbs steps | 1 | 2,000 | 104 |
| One-step BPTT | 60 | 10 relaxation steps | 0.01 | 2,000 | 104 |
| Full BPTT | 60 | 10 relaxation steps | 0.1 | 60,000 | 143 |

The paper's BPTT control uses SGD; no BP AdamW experiment is represented in this table. Every objective uses seeds 0–19 and its corresponding 1,000-puzzle `train-s<seed>` split. All probes score the fixed first 512 development puzzles, at upstream search budget `4e10` per puzzle. No Figure 4 command scores test data.

There are 12 dense runs: seeds 0–2 for the four methods other than full BPTT. They probe in the fit process after every update through update 64, then eight points per doubling, and at the final update. Their fit-time cap is the source default 3,600 seconds. The other 88 runs save those same checkpoints and defer probes; their fit-time cap is 310 seconds. Full BPTT uses `training_curves.py train --long` for all 20 seeds. The figure's 60,000-update cosine schedule is different from both the 2,000-update standard Table 7 run and the 480,000/3,840,000-update Table 7 schedules at learning rate 2. A cap may truncate a schedule on this hardware; the runner records the actual last update and never labels a capped run as schedule-complete.

The full schedules contain 9,740 development probes, or 4,986,880 puzzle evaluations before accounting for restarts. This is the dominant expected runtime, rather than the fits alone. Pending probes are resumable through `probes.json`; training uses the source's native checkpoints. A completed deferred run is accepted only when the probe keys exactly equal the saved checkpoint updates and every probe has 512 examples. Dense logs receive the same coverage check. Completed output and checkpoint hashes are retained in the execution record.

## Reuse and time axes

Sparse Table 7 runs do not contain the dense checkpoint grid, so they cannot replace the 12 dense fits. Their fit clocks are reusable: dense seeds 0–2 are assigned to the same physical GPU as their corresponding Table 7 final run. At export, probe accuracy comes from the dense run while each update's time comes from its validated final-run log. The final and dense recipes must agree, and overlapping model tensors should be checked before mapping clocks. Source probes save and restore the CPU and CUDA RNG states, which is why denser probing is intended to preserve the trajectory.

Deferred runs use their own fit clocks. Preserve both raw times and the paper's adjustment: one-time startup is update 1's fit time minus the median duration of updates 2–21. The figure additionally includes inference-ready statistic fitting time (`ready_s`) measured in standard final-run test preparation for seeds 0–2. The released export retains those three overheads separately; it does not implement the final figure interpolation/quantile calculation. An exporter must state how it combines these overheads across 20 seeds instead of inventing an unstated plotting convention. Report mean curves and 25th–75th percentile bands only after all 20 seeds are measured, with missing or capped runs visible.

These machines are Blackwell GPUs, not the paper's H200. Preserve host and GPU identifiers when plotting timing; avoid treating a mix of hardware or cold-fit clocks as a reproduced H200 speed ratio. No unverified retiming output or imported curve file is eligible for local evidence.

## Validation completed during preparation

- All 100 queued probe schedules were compared with the pinned upstream `probe_points` implementation and matched exactly.
- `training_curves.py splits` completed successfully on CPU for all 20 splits: each contains 1,000 distinct puzzle identities with no overlap with the development or test puzzle identities. The log is `artifacts/figure4_splits.log`.
- The manifest contains exactly the five objectives times 20 seeds, with 12 dense and 88 deferred runs. The expansion preserves every dense run's original physical GPU.

The queued runner has been syntax-checked. The four original idle waiters were stopped only after checking that they had no children and that all 100 curve output directories were absent. Their replacement PIDs are loki GPU0 1099342, loki GPU1 1099343, work GPU0 1109997 and work GPU1 1109998. New ripper workers are GPU0 PID858327 and GPU1 PID858328. They started only after the baseline owner explicitly confirmed GPU release and a fresh NVML owner check agreed. The ripper workers use the common `artifacts/gpu0.lock` and `gpu1.lock`; original lanes retain their objective-worker locks. No Table 7 job assignment changed. The before/after configuration hashes and migration map are retained under `artifacts/figure4/migrations/`, with remote stop evidence collected alongside lane artifacts.

Collect current status with `python3 scripts/figure4_queue.py collect`; the local summary is `artifacts/figure4/summary.json`. At expansion, two PL curve jobs began on ripper while the other four lane workers remained `queued_after_table7` without owning GPU locks.

## Measured probe-concurrency experiment

The 512-puzzle deferred probes, rather than subsecond PL fits, dominate runtime. A bounded experiment on ripper GPU0 compared the same PL seed9 checkpoints 16 and 32 in one sequential process versus two simultaneous processes. It ran only after the existing probe child completed, under the same physical-GPU lock, with the model, source, data, search settings and seeds pinned. It changed no MPS, clock, power or other-lane setting.

Sequential execution took 56.953 seconds; two processes took 61.200 seconds, a speed ratio of 0.931. Both modes produced exactly 508/512 and 510/512 solved puzzles. Two resident CUDA contexts therefore did not improve end-to-end runtime in this test. Evidence is `artifacts/figure4/ripper/probe_benchmark/20261113T200335Z/`, including commands, checkpoint hashes, GPU-owner samples and every probe log.

A subsequent isolated CUDA MPS comparison used the same two checkpoints and GPU UUID. Sequential execution took 56.962 seconds; two MPS clients took 31.661 seconds, a 1.799-fold throughput improvement. Solved counts remained exactly 508/512 and 510/512. This checks aggregate results at two checkpoints, not bitwise trajectories or every method. Evidence is `artifacts/figure4/ripper/probe_benchmark/20261113T201309Z-mps/`. The private daemon was restricted to GPU0, with private pipe and log directories, and was stopped before the next fit. No power, clock, compute-mode or global daemon setting changed.

Production deferred probing on `ripper:0` and `ripper:1` now uses `scripts/run_mps_probes.py`. Two clients receive disjoint checkpoint groups through the unchanged upstream CLI; validated atomic shards merge into the canonical `probes.json`. Completed probes are not rescored on resume. Source, model and configuration identities are checked, the parent retains the physical-GPU lock, and its private daemon exits before subsequent training. Other MPS daemons are permitted only with explicitly disjoint full GPU UUIDs and existing private task-pipe directories owned by the current user with mode 0700; ambiguous or shared ownership causes sequential fallback. Shutdown addresses only the exact private pipe. The reviewer checked a completed 104-update production run for exact shard/canonical agreement and daemon cleanup. The measured 1.799-fold improvement is from ripper GPU0, not a separate measurement on every enabled device.

## Automatic collection and plotting

`scripts/watch_objective_studies.py` collects Table 7, D.2 and Figure 4 in parallel every 60 seconds. Its durable singleton process runs under the isolated `artifacts/plotting-venv` environment; the pinned experiment environment is unchanged. Current status and errors are in `artifacts/objective_studies/supervisor.json`. It writes `REPORT.md`, `table7.csv` and summary JSON there, plus validated individual curves in `artifacts/figure4/curves.json`. After collection, the guarded dispatcher may refill an already-released authorized lane with up to four untouched deferred jobs. It validates the full prior-work dependency receipts, absent owner processes, an empty target GPU and the physical lock before launching.

Dynamic transfers preserve every job field except its lane. Dense jobs remain on their original physical GPUs. Any existing run directory or job artifact on any involved host excludes a transfer. All host dispatch locks are held while configuration copies are staged and committed, and workers reload ownership under the same lock before creating provenance. Existing active workers must advertise `dynamic_dispatch_v1`; the two ripper workers acquired this marker only after their current probes completed and were validated. The four original D.2 wrappers load the guarded worker at handoff. New future lanes are admitted only after their explicitly assigned main/data-efficiency or long-BPTT/Maze-Hard work completes. The `mps_future_lanes` allowlist enables the same isolated probe mechanism on those future Max-Q allocations. Receipts and rollback evidence live under `artifacts/figure4/migrations/`.

The exporter requires exact dense-clock validation and all three standard readiness measurements for a method. It retains the raw fit clock, subtracts the stated startup estimate, and adds the median of those three readiness times. Across 20 seeds it uses linear interpolation in elapsed seconds, zero before the first recorded point, and the last recorded value afterwards. It reports means with 25th–75th percentile bands. These interpolation and readiness-pooling choices are explicitly reconstructed because the upstream final plotting script is absent. PNG, PDF and SVG figures are produced only after all five methods have 20 validated curves. Synthetic checks verify the renderer, exact known crossing, and missing-seed rejection; they are not experimental evidence.

The release also contains 100 supplementary ALT-free/Adam Maze curve artifacts. They are legacy optional exports, not a reported arXiv-v1 figure or numerical claim; they are excluded from required reproduction. The paper's Maze objective result is covered by Table 7 with the published SGD/free-kernel protocol.
