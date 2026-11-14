# Sudoku: stopping rules matter

An exact Sudoku rule check added to the learned stopping condition increased low-data solve accuracy from **20.44% to 60.87%** on a reserved development confirmation cohort, without retraining. The paired gain was **40.43 percentage points**, with a two-way bootstrap 95% interval **[35.54, 44.92]**.

The experiment uses the same frozen k=17 and k=21 models, six training seeds, 2,048 official development puzzles at rows [512,2560), four chains and 262,144 moves per chain. It reuses the original-arm controls from the digit-symmetry study. Every returned board was checked independently; all 49,152 original/intervention boards passed the audit, and the intervention made zero wrong stops. A higher move cap is not being introduced, but preventing early incorrect stops increases realized search work. The gate supplies extra Sudoku rules, and its arithmetic is absent from the legacy FLOP proxy. There is no net-compute or cross-host timing claim.

The gate accepts a learned-energy stop only when no equal digits occur across the 810 unordered row, column or box peer pairs. Clues stay clamped. It never reads target solution labels. Preflight checks compare the gate-disabled kernel against all upstream state fields, verify unchanged moves when stopping is disabled, and recompute conflict counts independently.

The first 512 development puzzles were discovery data; that screen gained 40.07 points. The confirmation expanded to six models per training size and used disjoint puzzles. Confidence intervals resample paired training-seed clusters and puzzle indices, retaining shared k groups. Six seeds are a small sample; repeated evaluations are not independent new puzzles.

## Reserved test follow-up

The unchanged gate subsequently reached **61.30%** on the same reserved 10,000-puzzle test slice used by the projection study, versus original CRAM **21.1975%**. Its gain was **40.1025 points**, approximate paired 95% interval **[35.4633, 44.4818]**. All 240,000 original/intervention boards were independently audited, including paired puzzle identities, clues, budgets and zero intervention wrong stops. See the [public test audit](../../../results/research/foureyes/audit-heldout.json). The official test corpus had already been evaluated in the original reproduction; this is a reserved follow-up slice, not a newly collected external dataset.

## Can training-only calibration replace the rule check?

On another 2,048 development puzzles at rows [2560,4608), the minimum training-solution energy threshold achieved 55.29% versus 61.04% with the rule gate. Its difference of −5.75 points, with one-sided 95% bounds [−6.86, −4.58], is **inconclusive** for the frozen five-point noninferiority margin. It never halted any puzzle: all results were best boards returned at the cap. Its conditional wrong-halt rate is therefore **undefined**, not zero.

The training-energy fifth-percentile threshold achieved 55.31% but made 5.81% wrong halts among 155 halted puzzles. The one-sided lower 95% bound was 1.22%, above the frozen 1% requirement, so that hypothesis was **falsified**. Threshold calibration did not establish a practical substitute for task-rule checking.

## Evidence

- Public [summary](../../../results/research/foureyes/summary.json), [per-seed data](../../../results/research/foureyes/per_seed.csv), and [confirmation audit](../../../results/research/foureyes/audit-confirmation.json).
- Frozen [discovery](protocol.json), [confirmation](confirmation.json), [calibration](calibration.json), and [reserved test follow-up](heldout.json) protocols.
- [Kernel](gate_kernel.py), [runner](confirm.py), [independent confirmation auditor](analyze_confirmation.py), [calibration auditor](audit_calibration.py), and [public exporter](export_public.py).

Raw boards, checkpoints and per-puzzle arrays remain in the local artifact archive; the public exports retain their hashes. Regeneration of the independent audits requires those arrays. These studies extend the paper and are separate from its original reproduction.
