**Figure 3b is complete: all 102 runs and 17 six-seed points are measured and
validated** in the 2026-11-14 06:38:02 UTC
[collection](../results/reproduction/data_efficiency/collection.json). There are no missing,
invalid, failed, or degenerate rows. All six k=1 runs score 0%, giving a
measured median and seed range of 0%. The published CRAM k=1 records remain
unavailable; local completion does not fill that reference gap.

The first sampled point with median accuracy at least 98% is
**k=30: 99.415%, with a 96.32–99.81% seed range**. This agrees with the paper's
sampled CRAM crossing at 30 puzzles. The published CRAM median there is
98.865%; published EBR first crosses 98% at k=700 (99.085%), and FRM at
k=1,000 (98.585%). Published TRM does not reach 98% through k=100,000.
Those are contextual published comparisons: the original training-reference
arrays are unavailable, so local CRAM and the published baselines did not
train on the same sampled arrays. They do not establish a locally reproduced
advantage over newly trained baselines.

| Method | Evidence in this Figure 3b study |
|---|---|
| Ordinary backpropagation with AdamW (BP AdamW) | Outside the local baseline-training scope; not run as a separate control |
| CRAM | Fresh seed-42 reference recreation; 102/102 validated runs, plus a separate published curve |
| TRM | Published six-seed measurements only; no local training or evaluation |
| EBR | Published six-seed measurements only; no local training or evaluation |
| FRM | Published six-seed measurements only; no local training or evaluation |
| Weight-Space Programming (`WSP-*`) | Local zero-training (`k=0`) rule-programmed reference across seeds 0–5 on the same 10,000 test puzzles (supplied Sudoku rules; `0` learned parameters) |

All accuracies below are percentages on 10,000 held-out puzzles per seed.
Complete local points are the median and minimum–maximum across seeds 0–5;
published columns are six-seed medians. A dash denotes an unavailable point,
not zero. The collector's 17 completed medians were independently recomputed
from integer solved counts. Missing baseline points at k=12–27 are not
interpolated.

| Unique puzzles k | Local CRAM median | Local CRAM range | Published CRAM | Published TRM | Published EBR | Published FRM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 0.00–0.00 | — | 2.945 | 0.000 | 0.000 |
| 3 | 0.000 | 0.00–0.00 | 0.000 | 7.750 | 0.000 | 0.000 |
| 10 | 1.190 | 0.98–1.60 | 0.905 | 11.890 | 0.000 | 0.030 |
| 12 | 1.650 | 1.05–1.80 | 1.545 | — | — | — |
| 14 | 2.755 | 1.14–4.13 | 2.895 | — | — | — |
| 17 | 11.020 | 3.80–17.78 | 8.840 | — | — | — |
| 19 | 21.810 | 10.42–26.65 | 18.890 | — | — | — |
| 21 | 35.195 | 20.72–63.90 | 37.165 | — | — | — |
| 23 | 63.260 | 41.36–84.06 | 64.545 | — | — | — |
| 25 | 84.370 | 69.29–93.75 | 75.890 | — | — | — |
| 27 | 94.110 | 86.22–98.20 | 93.865 | — | — | — |
| 30 | 99.415 | 96.32–99.81 | 98.865 | 16.645 | 3.665 | 3.075 |
| 100 | 99.990 | 99.98–100.00 | 100.000 | 23.605 | 62.100 | 12.700 |
| 300 | 100.000 | 100.00–100.00 | 100.000 | 55.605 | 57.850 | 76.465 |
| 1,000 | 100.000 | 99.99–100.00 | 100.000 | 85.855 | 99.795 | 98.585 |
| 3,000 | 100.000 | 99.99–100.00 | 100.000 | 87.140 | 99.900 | 98.895 |
| 10,000 | 100.000 | 100.00–100.00 | 100.000 | 87.600 | 99.895 | 98.910 |

### Zero-training Weight-Space Programming (`k=0`, rule-programmed) reference

Separately from the data-learned `k = 1..10,000` curves above (which receive no Sudoku geometry or constraint rules), Weight-Space Programming compiles explicit Sudoku peer/unit rules directly into pure tensor weights at **`k = 0` training examples** (`0` learned parameters, `wsp.free_model.static_problems == []`). Evaluated across seeds `0–5` on the identical `10,000` held-out puzzles (`full-all` rows `10000..19999`), deterministic `WSP-Transformer` solves **`11.690%`**, deterministic `WSP-Det` solves **`15.370%`**, and associative `WSP-Assoc` solves **`100.000%` (median and range `100.00–100.00%`, `0` wrong stops, `24.05M` mean `FC2` FLOPs)**. The unchanged frozen Figure 3b outputs remain untouched; the extended comparison is rendered separately in [wsp-figure3b-data-efficiency.png](../assets/figures/wsp-figure3b-data-efficiency.png) ([PDF](../assets/figures/wsp-figure3b-data-efficiency.pdf), [JSON](../results/wsp/figures/figure3b-wsp.json); see [WSP_BENCHMARK_REPORT.md](WSP_BENCHMARK_REPORT.md)).

At k=1,000 all six fitted model files have identical SHA256
`e5550d0cba005c7b21ff50a55cf1d1755816d23e4013edbf5461d2a487330aeb`,
consistent with Appendix C.9's deterministic fit on the entire shared draw.
Search seeds 0–4 solve 100% and seed 5 solves 99.99%. The seed ranges at
smaller k describe variation across selected training subsets as well as
search seeds; they are not confidence intervals.

The Figure 3b reproduction consists of 102 frozen CRAM runs: 17 training-set
sizes × six seeds. The sizes are 1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27,
30, 100, 300, 1,000, 3,000, and 10,000 unique puzzles. Each puzzle contributes
its original row and its first 99 stored variants. Every fit uses 32 full-batch
SGD updates with momentum 0.5, cosine decay from learning rate 540, and
proximal L1 coefficient `3e-3/9`. Per-update digit relabeling is disabled because
the stored examples already supply the augmentation.

The original unseeded training-reference arrays were not released. This study
therefore uses the explicitly seeded reference-42 recreation, prepared with
the pinned original builder. The six subset seeds select from that reference.
This preserves the declared protocol but does not recreate the paper's exact
sampled training sets. Published baseline curves are reference measurements
from different sampled training sets; no baseline is trained or scored here.
For k<1,000, seed-s subsets are nested prefixes of a seed-s permutation of
the recreated 1,000-puzzle draw. The six k=1,000 jobs use that entire draw;
the larger draws add official training-split puzzles with the original builder.

The queue is frozen in `configs/data_efficiency_jobs.json`, with source hashes
and the reference-provenance hash. Its SHA256 is
`bded95b30345e299fefbaaf5361a5d4b3ba9fd6dbd35a40da7909501d3b17a35`.
All 46 Python source files are pinned to commit
`62bbe830c7d6f87f2ba18f30a28039258b986454`; per-file hashes and the reference
manifest hash are recorded in the configuration and collection. Its isolated Python source is in
`dataeff_source`; outputs are in
`dataeff_source/experiments/runs/CRAM-E/DE2-v99-recreated42/kK-sSEED`.
Each run saves the cold fit, seven warm fits, the model, selected training
rows, and a single search of full-all test rows 10000–19999. Inference uses
the threshold fitted on that run's training solutions with tolerance 1e-3,
four chains, temperature 0.40→0.31 times the mean positive weight over 524,288
moves, focus 0.45 times that weight, checks every 4,096 moves, and the 4e11
legacy-operation budget per puzzle. Full test-data SHA256 is
`2fdc470ee042f11ffa61c89f606f39004d8914fca3312c88602d9f68212dec6c`.
This is distinct from Figure 3a's zero-energy stopping protocol.

The CPU collector runs with:

```sh
upstream/.venv/bin/python -u scripts/collect_data_efficiency.py --watch --plot
```

It retains every expected row in `artifacts/data_efficiency/collection.json`
and `rows.csv`, including pending, running, failed, invalid, and degenerate
runs. A row is measured-complete only after both phases succeeded and all
recorded checks pass:

- Source, frozen configuration, reference, subset manifests, all input/label/
  group-index arrays, test dataset, and model/output hashes agree.
- Training consists of exactly the original plus the first 99 variants of
  every group. Every selected training row is compared against the stored
  source arrays; no sampled spot check replaces this comparison.
- The fit has the exact 32-update configuration, seven warm repeats, and
  bitwise agreement between warm and cold weights. Model shapes, finiteness,
  parameter counts, positive weights, and model hashes are checked.
- The search uses the expected model, fitted threshold, frozen settings,
  seed, and exactly 10,000 ordered test IDs. Solved flags, wrong stops,
  exhausted budgets, and legacy-operation summaries are independently
  recomputed from all per-puzzle records.

A fit with no positive weight cannot define the temperature unit. The
released evaluator skips such a search. The collector records it as
`degenerate` with no test accuracy; it is never converted into a measured
0% result. Complete execution with degenerate rows is labeled separately
from a complete measured figure.

The collector validates large files once and reuses a hash only while device,
inode, size, nanosecond modification time, and change time agree. Completed
row validation is reused only while its source, configuration, training data,
test data, metadata, selected-row archive, and output digests agree. The first
audit read 15.09 GB in 6.99 seconds; a subsequent unchanged audit took 0.054
seconds. These are collector costs, separate from fit and inference timing.

`scripts/plot_data_efficiency.py` plots a fresh point only when all six seeds
have measured results. It uses their median and shows the seed range. Missing
points remain gaps. During execution, outputs are named
`figure3b-preview.{png,pdf,json}` and visibly labeled incomplete. The final
`figure3b` filenames and `figure_complete: true` require all 102 measured
runs. Published TRM, EBR, FRM, and CRAM curves are explicitly labeled.
The published CRAM export lacks the six k=1 test records; those are preserved
as missing reference data rather than inferred to be zero.
The final [PNG](../results/reproduction/data_efficiency/figure3b.png),
[PDF](../results/reproduction/data_efficiency/figure3b.pdf), and
[metadata](../results/reproduction/data_efficiency/figure3b.json) cover all 17 local points.
Visual inspection confirms a final protocol-recreation title, the measured
k=1 point, readable axes and seed-range shading, and separately labeled
published curves. The footer states that published baselines were not rerun
and that their sampled training reference differs from the local recreation.

Published references are retained verbatim in
`upstream/results/sudoku-data-eff-baselines-v99.json` (SHA256
`99779224f0cafebb547e9427b6c347545ef1fb293fd460582ad0554d6c324a5a`)
and `upstream/results/sudoku-data-eff-shared-v99.json` (SHA256
`8589e620eb8acd8c86b55fea9bcbe2512117c4d91ba974d86b499ea9c6b04747`).
The [Figure 3 caption and main-text comparison](../references/review/01_main.md)
define six-seed medians and the 30-versus-700 claim. [Appendix C.9](../references/review/03_appendices_abc.md)
specifies identical sampled training puzzles for the paper's methods, 65,100
TRM updates at batch 768, 60,000 EBR updates at batch 768 with repeated groups
when k<768, and two 100,000-update FRM stages at batch 128 with checkpoint
selection on 256 development puzzles. These baseline training budgets have
not been locally executed in this data-efficiency study.

CPU validator checks use synthetic fixtures in temporary directories only.
They confirm that corrupted wrong-stop summaries and altered selected
training rows are rejected, and that a degenerate model remains unscored.
The checks are recorded in `collector-validation-checks.json`; they are not
training or accuracy results. Live collection status and audit overhead are
appended to `collection-monitor.jsonl`.

The CPU preflight at 20:42 UTC on 2026-11-13 validated all 102 job/data pairs
(97 unique datasets), their array hashes, group layouts, manifests, all 46
frozen source files, and the held-out test-data hash. About 581 GiB remained
free. A conservative output estimate is 2.77 GiB: uncompressed selected
training arrays, models, scaled JSON results with 2× headroom, and two complete
search checkpoints for each of three lanes. This is an output-space estimate,
not measured final storage.

Before execution, the first six queued jobs used 1,000 training puzzles each.
Published H200 records
imply roughly 3.05–3.17 seconds for a cold fit plus seven warm fits, excluding
process startup, file I/O, saving, and CUDA context. Published search operation
counts combined with existing local 10,000-puzzle development throughput give
a **118–185 second workload projection per search**. This is not a local
data-efficiency timing result: recreated training rows, connectivity, long
search tails, and compilation may change elapsed time. Across the 96 runs
with published search counts, the same projection totals 4.64–6.59 GPU-hours;
the six k=1 search costs remain unknown. It is not a completion-time promise.

The pending queue now uses `proc.wait(timeout=15)` to wake immediately when a
child exits while retaining periodic progress checks. A real CPU subprocess
lasting 0.25 seconds took 15.00 seconds through the previous polling loop and
0.26 seconds through the revised loop. Nonzero exit codes and repeated
progress callbacks were also verified. Fit/search code, job order, frozen
protocols, GPU locks, failure records, and scoring remain unchanged. No
additional GPU lanes or MPS searches were launched by this audit.

Evidence: [runtime-preflight-audit.json](../results/reproduction/data_efficiency/runtime-preflight-audit.json)
and [queue-wait-validation.json](../results/reproduction/data_efficiency/queue-wait-validation.json).
Regenerate the CPU audit with `upstream/.venv/bin/python scripts/audit_data_efficiency_runtime.py`.

At 04:41 UTC on 2026-11-14, the three original workers were still searching
k=1 seeds 3–5 while all other studies had released nine GPUs. The 12 untouched
k=3,000/10,000 jobs were therefore dispatched across those nine devices:
foureyes GPU 3, ripper GPUs 0–3, and work/loki GPUs 0–1. The frozen job
configuration and all fit/search code remain unchanged. Each imported record
retains its original job identity plus its actual host, GPU, execution command,
and delegation provenance. The seven warm fits still run exclusively on each
job's GPU. These fit times span multiple hosts and GPU power limits; they must
not be pooled into a claim of matched-hardware training throughput.

The old schedulers had already loaded static queues. Only their three exact
parent PIDs were paused, after checking command lines, parentage, and process
start times; their active search children were not signaled and continued
advancing. The independent recovery watcher waited for every delegated worker
and child to exit, verified physical locks, imported saved outputs, and ran the
full collector before resuming the original schedulers at 04:56:33 UTC.
All 12 delegated runs are now measured and validated; the delegation manifest
has status `resumed`. Their existing validated-completion checks skip the
imported jobs. Durable host launch
receipts and a shared transaction lock protect interrupted SSH launches and
prevent launch/resume races. Failed work is handed back only after its
processes are absent and any partial outputs validate.

The [delegation manifest](../results/reproduction/data_efficiency/extra/delegation.json),
[live progress snapshot](../results/reproduction/data_efficiency/extra/live-progress.json),
and [helper validation](../results/reproduction/data_efficiency/extra/helper-validation.json)
retain the exact assignments, original process identities, recovery state,
unchanged configuration hash, and measured evidence that all 12 GPUs were
productively searching. The controller is `scripts/data_efficiency_extra.py`.

The remaining k=1 searches motivated bounded checkpoint-continuation and
move-kernel sharding benchmarks. The [benchmark bundle](../results/reproduction/checkpoint_resume_benchmark/20261114T044046Z/REPORT.md)
keeps copied input checkpoints, eight input-file hashes, all 46 source hashes,
commands, hardware, versions, timing samples, and exact-state comparisons.
Its input-manifest SHA256 is
`0a0ba02e0f74f9e1819de64bf3248dd2b8651a89881893d0b66765106ce86b0d`.
These measurements cover isolated bounded continuations; they are not final
search times or evidence of extra solved puzzles.

| Execution comparison | Training seed | Work per timed repetition | Measured throughput gain | Correctness evidence |
|---|---:|---|---:|---|
| Native 300 W Max-Q → 450 W Workstation | 4 | 16 checks, two repetitions | 1.2459× | All 24 state fields excluding elapsed and static model/signature equal |
| Native 300 W Max-Q → 450 W Workstation | 5 | 16 checks, two repetitions | 1.2596× | All 24 state fields excluding elapsed and static model/signature equal |
| One → four Max-Q GPUs on foureyes | 3 | 16 checks, three alternating repetitions | 3.8907× | 23 exact-state comparisons passed |
| One → two Workstation GPUs on work | 5 | 32 checks, three alternating repetitions | 1.9908× | 23 exact-state comparisons passed |
| One → four Max-Q GPUs on ripper | 4 | 32 checks, three alternating repetitions | 3.8805× | 23 exact-state comparisons passed |

Native seed-4 checks took 1.342809 versus 1.077795 seconds; native seed-5
checks took 1.340551 versus 1.064307 seconds. Those native gains were below
the 1.3× hardware-migration threshold, so a hardware-only move was not
recommended. The sharded measurements include state copies and exclude
compilation. Work's three native 32-check samples were 34.6221, 35.2480, and
35.3697 seconds, versus 17.6505, 17.7052, and 17.9135 seconds sharded.
Ripper's were 42.3701, 42.8959, and 42.9467 seconds native, versus 11.0543,
11.0253, and 11.0551 seconds sharded. Every timed repetition restores the
same captured initial state; the gains use median elapsed times.
The additional same-host seed-3 proof uses the production capture at move
7,229,440: native 16-check samples were 16.0340, 16.2516, and 16.2694 seconds,
versus 4.1661, 4.1771, and 4.2092 seconds on four GPUs. Its exact-state
comparisons and **3.89066×** gain are retained in the migration's
`seed3/proof.json`. These are within-host, bounded sharding comparisons;
they do not establish a matched-hardware end-to-end speedup for the entire
data-efficiency study.

The sharding change dispatches the unchanged move operations across devices
while preserving global RNG chain identities. Original checks, refreshes,
compaction, temperature, stopping, and budgets remain centralized. An AST
audit restricts the kernel edit to the global chain offset. Validation compares
pre-refresh state, eight checks from the captured original checkpoint, eight
checks from a forced-compaction fixture, repeated timed outputs, and final
native/sharded state. Separate native cross-host proofs cover captured seed 4
from foureyes to ripper and seed 5 from foureyes to work. These bounded exact
traces support the execution change but do not prove every future trajectory.

A production handoff is recorded in
`artifacts/data_efficiency/migrations/20261114T0510Z`. Seed 3 completed on
foureyes GPUs 0–3, seed 4 on ripper GPUs 0–3, and seed 5 on work GPUs 0–1;
all three imported results are validated.
All three actual-resume checks passed exact equality of 24 state fields.
The [progress audit](../results/reproduction/data_efficiency/migrations/20261114T0510Z/progress-audit.jsonl)
retains those receipts and measured process/GPU progress. The handoff is
separate from final measured completion. Final
acceptance requires immutable checkpoint and execution-source archives,
a stopped original writer, successful actual-resume parity, matching same-host
sharding proof, the appropriate cross-host proof, successful completion, and
collector validation of all original input/model/configuration hashes and
every per-puzzle result. The collector checks migration provenance even when
a completed row's other validation is cached. All three continuations passed
these gates before their results entered the completed collection.

The imported seed-3, seed-4, and seed-5 rows were independently checked against their
saved output hashes, migration receipt/handoff/proof hashes, and all 10,000
per-puzzle solved/stopped flags. All completed at 52,461,568 moves, the first
4,096-move check boundary at or beyond their 52,457,941-move cap. They solve
zero puzzles; seed 3 has 3,854 wrong stops and 6,146 exhausted budgets, while
seed 4 has 1,676 wrong stops and 8,324 exhausted budgets, and seed 5 has 569
wrong stops and 9,431 exhausted budgets. These are measured
zero-accuracy outcomes, with one recorded checkpoint resume each.

The [final independent audit](../results/reproduction/data_efficiency/final-audit.json)
recomputes all 1,020,000 saved test evaluations and all 17 medians/ranges,
checks output, dataset, source, and migration hashes, and records the final
visual review. It verified 897 retained input hashes in 7.76 seconds on CPU,
reusing each file digest within that audit to avoid duplicate reads. Each run
uses the same 10,000 distinct held-out puzzle IDs;
the total evaluation count is not a count of distinct puzzles. The collector
exited after writing the final collection and figure. Final collection SHA256
`07fce7fbc473ca9cf3328527962fce20d3eeeb855fa4a6f9b3a607f2285276db`
matches the plot metadata's source hash. PNG SHA256 is
`e80cb055f624d80bd7d2de0a54d1f465bad68b9f55ef9755184c56a96b498dcc`.
The saved Sudoku records retain correctness/status flags rather than final
boards: this audit independently recomputes their counts and consistency,
but cannot independently recheck final boards against Sudoku rules.

The reported `search_seconds` follows the upstream checkpoint timer: it adds
elapsed search time retained in the saved checkpoint to the continuation's
search time. It excludes staging and work discarded between the last saved
checkpoint and retirement of the original process, so it is not end-to-end
wall time. The handoff journal records checkpoint ages at capture of
**515.1, 26.8, and 171.7 seconds for seeds 3, 4, and 5**, respectively.
Those ages are wall-clock proxies for discarded in-memory progress, not
exact discarded move counts or corrections that can simply be added to the
search timer. Migration startup, validation, copying, and coordination also
belong to end-to-end cost. No such aggregate speedup is inferred here.
