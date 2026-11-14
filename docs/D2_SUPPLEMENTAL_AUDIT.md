# Appendix D.2 supplemental claims

All nine supplemental implicit fits and all twelve Table 7 long/intermediate fits completed their declared schedules. The objective-family exporter is complete as of 2026-11-14 at 00:39 UTC. The independent final audit rechecked their collected output hashes and recomputed means, sample standard deviations and development maxima without new test inference. Evidence and complete standard/long comparison tables are in `artifacts/objective_studies/FINAL_REPORT.md`, `table7-comparison.csv`, `d2-implicit-results.csv` and `final-validation.json`.

These nine development-only checks are distinct from Table 7's 72 standard runs and twelve long/intermediate fits. The latter comprise five-step BPTT at 60,000 updates, EqProp at 240,000, and full BPTT at 480,000 and 3,840,000, each with seeds 0–2. Every test row below evaluates 50,000 Sudoku-Extreme puzzles per seed, with checkpoints selected on development data. Published values are quoted references.

| Method and schedule | Local test solved (%), mean ± sample SD | Published test solved (%) |
|---|---:|---:|
| BP AdamW | Outside this objective protocol; not run | Not part of this study |
| Five-step BPTT, 60,000 updates, LR 6 | 99.90733 ± 0.11202 | 99.89 ± 0.07 |
| EqProp, 240,000 updates, LR 6 | 16.55867 ± 3.75927 | 16.05 ± 8.75 |
| Full BPTT, 480,000 updates, LR 2 | 46.76533 ± 3.61701 | 46.24 ± 2.86 |
| Full BPTT, 3,840,000 updates, LR 2 | 48.56733 ± 4.12005 | 48.82 ± 3.73 |

The full 3.84-million-update seeds scored 52.740%, 44.502% and 48.460%; each complete fit clock was 2.463–2.508 hours on local Blackwell GPUs. Published times of 4.8–5.0 hours were measured on H200, so this is not a controlled hardware speed comparison. None of the full-BPTT or EqProp three-run development means reached 99%. Five-step BPTT reached it at update 46,341, corresponding to 139.6597 adjusted local seconds and 1.9321549e15 training FLOPs, versus published 174 seconds and 1.37e15 FLOPs. Different crossing updates account for the arithmetic difference; the trajectory difference's cause is not isolated.

## Implicit differentiation at 60,000 updates

This claim was absent from the nine managed long jobs. The released `slurm/training_methods/long_d.sbatch` explicitly expands methods `bptt`, `bptt5`, `eqp`, `imp` over learning rates 6, 20 and 60 and seeds 0–2, keeping each method's selected knob and L1 multiplier. The implicit setting is knob 10, L1 multiplier 0.01. Thus nine additional development-only fits were declared in `configs/d2_supplemental_jobs.json`, with exact argv and source/data provenance. They make no test-set evaluation.

Four persistent `scripts/d2_queue.py` workers were launched on the existing work/loki lanes on 2026-11-13. Each waited for its Table 7 queue, acquired the same physical-GPU lock, ran its assigned implicit fits, then handed its lane to Figure 4. All four handoffs completed. Only the four idle Figure 4 waiters were replaced; active Table 7 and ripper jobs were untouched. The insertion checked that all nine destination directories were absent and retained old/new PID and configuration-hash evidence. Current status is `artifacts/d2/summary.json`; refresh with `python3 scripts/d2_queue.py collect`.

Each fit has the source default 3,600-second fit cap and 60,000-update cosine schedule. Its existing probe uses 512 development puzzles. Retain every learning rate and seed, actual update count, stop reason and probe curve; report the maximum observed probe solve rate. The paper's "at most 0.6%" corresponds to at most 3/512 puzzles when expressed at this precision. A capped or nonfinite run is reported explicitly and is not evidence that a full 60,000-update schedule completed. The release contains no raw implicit-long result record that fixes a narrower subset of these nine source-script runs.

The largest local development solve rate across all nine fits and all recorded probes was **3/512 = 0.5859375%**, consistent with the published “at most 0.6%.” All nine completed 60,000 updates with `stop=schedule`; none was capped or nonfinite. Their raw fit clocks were 109.676–119.076 seconds. No test data was evaluated.

| Learning rate | Seed 0 maximum solved / 512 | Seed 1 maximum solved / 512 | Seed 2 maximum solved / 512 |
|---:|---:|---:|---:|
| 6 | 1 | 1 | 3 |
| 20 | 2 | 1 | 2 |
| 60 | 1 | 0 | 2 |

## EqProp nudging and gradient agreement

The release hard-codes beta=0.01 in `training_methods.py` and comments that beta=0.5 diverged. The paper reports cosine similarity 0.99999 with full BPTT. It does not provide the diagnostic model, checkpoint, batch, seed, relaxation setting or divergence trajectory; no corresponding diagnostic script or result artifact is released. The code comment names implicit differentiation, while the paper names full BPTT, so the local check compares against both.

`scripts/check_objective_gradients.py` completed a new CPU sanity check in 0.424 seconds. It uses a deterministic 32-puzzle training subset from seed 0, no augmentation, 30 free and nudged steps, and zero weights or Gaussian weights with standard deviation 0.001 and seed 20261113. Full BPTT is checked at both 10 and 30 steps; the implicit adjoint uses 30 iterations. This is a declared local diagnostic, not a reconstruction of the unspecified published measurement.

| Initialization | Beta | Cosine with full BPTT, 10 steps | Cosine with implicit, 30 steps | Maximum absolute negative-phase logit |
|---|---:|---:|---:|---:|
| Zero | 0.01 | 0.9999996712 | 0.9999996712 | 0.0983 |
| Small Gaussian | 0.01 | 0.9999996681 | 0.9999996691 | 0.1171 |
| Zero | 0.5 | 0.9924470036 | 0.9924470036 | 500.0 |
| Small Gaussian | 0.5 | 0.9924399898 | 0.9924399248 | 500.0 |

Every gradient in this bounded check was finite. The large negative-phase logits at beta=0.5 are an observed local effect; a single gradient does not establish or refute training divergence. Exact results, relative gradient errors, source/data hashes and runtime are in `artifacts/d2_objective_gradient_check.json`. Reproducing the original divergence claim requires its missing training recipe or a separately declared controlled ablation.

Local checkpoint-selection differences are recorded as observed differences. Their cause has not been isolated; they are not definitively attributed to hardware, library versions, RNG differences or floating-point behavior.
