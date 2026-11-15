# Exact graph recovery does not settle average solve quality

**The frozen ≥1-point solve-accuracy improvement hypothesis is inconclusive.** On all 512 frozen N600 models from the noise-law study, Chernoff thresholds solve **79.7082%** versus **78.9147%** for the original gap rule. The paired difference is **+0.7935 points**, with a two-way 95% bootstrap interval **[−0.0477, +1.7558]**. Neither the +1-point target nor even a strictly positive difference is established by that interval. All three arms have zero wrong certified stops.

| Threshold rule | Exact training graphs / 512 | Mean returned-board accuracy | Wrong certified stops |
|---|---:|---:|---:|
| Original largest gap | 494 | 78.9147% | 0 |
| Gap restricted below median | 503 | 80.3280% | 0 |
| Correct-law Chernoff | 453 | 79.7082% | 0 |

This is a matched comparison of threshold rules for one count energy; no optimizer is run. [Earlier PL/count controls](REPORT-raw-transform.md) cover optimization. Restricted-gap comparisons are descriptive and were not substituted for the frozen primary.

## Why test the structural ranking?

The preceding [construction study](REPORT-chernoff-replication.md) recovered more exact graphs with the original gap than with Chernoff thresholds. But the gap's failed models retained 21,865 extra edges in total; Chernoff retained only 60 and lost no true edge. Thus an exact-graph pass/fail score suppresses the severity of a model's errors. Here the observed mean solve ranking reverses the original-gap/Chernoff exact-recovery ranking, but its uncertainty prevents a confident superiority claim.

The construction outcomes were known before this solver hypothesis was chosen. [Protocol](noise_solver.json), SHA-256 `caf960c493b3eb05c1579ac257fa530af5ab548a125d97f9ba2d4fe51d201577`, froze the comparison before any inference on full-all rows **[220000,230000)**. These are reserved research inputs from the official corpus already used in the original reproduction, not a new external distribution.

All 512 source-bank models receive the same 10,000 puzzles, four chains, 262,144 moves per chain, unit-one penalties, zero biases, tau=0, tol=0.001, the fixed temperature schedule and **one common search seed, zero**. No search hyperparameter is changed. The 1,536 logical bank/arm comparisons collapse to **70 unique supports with identical inference settings**. Each support is searched once and mapped back to every matching logical bank. Equal source-bank weights are retained; mappings are not independent inference replications.

The frozen uncertainty calculation resamples 512 source-bank indices and 10,000 puzzle indices jointly in 5,000 paired two-way bootstrap draws. It includes the repeated identical models with their declared bank weights. It does not quantify variability across search RNG seeds, because only one common seed was run.

## Certificates and independent audit

For unit-one, nonnegative same-digit penalties with zero biases, total energy is a nonnegative integer. A certified stop under tau=0 and tol=0.001 therefore has energy zero. If the support contains every true Sudoku peer edge and givens are preserved, every row, column and box is valid. Extra edges can prevent some valid answers from receiving a certificate, but cannot create a wrong zero-energy stop. This argument does not cover capped best-board returns or supports missing true edges. The gap variants' observed zero wrong stops here do not imply a general certificate guarantee.

The [independent auditor](audit_noise_solver.py) verifies every support/threshold identity against the prior audited construction, all deduplication mappings and frozen settings. It independently checks **700,000 unique saved returned boards**, givens, true validity, exact-target equality, selected output versus saved state/capture, move caps and certified energies. It also recomputes every saved chain's integer fields and energies. All checks pass. The experiment uses 10,000 unique test inputs; the 15,360,000 logical mapped evaluations are not new independent puzzles or newly executed searches.

All 70 searches completed without errors in **131.53 seconds of four-GPU queue wall time**, at most 8.77 allocated GPU-minutes under the 30-minute envelope. Live counters showed four busy workers while eligible jobs remained; the last two jobs finished without redundant refills. The independent output/statistical audit took 9.25 seconds on Foureyes.

[Public summary and provenance](../../../results/research/foureyes/noise_solver/summary.json) · [Every logical bank](../../../results/research/foureyes/noise_solver/per_bank.csv) · [Model preparation and deduplication](prepare_noise_solver.py) · [Frozen runner](noise_solver.py) · [Independent auditor](audit_noise_solver.py) · [Exact-multiplicity bootstrap implementation](../ripper/bootstrap_two_way.py)

The [197,819-byte outcome archive](../../../results/research/foureyes/noise_solver/audited-outcome-bits.npz) stores only the 70 executed models' validity/stopping bits and puzzle IDs. [The public-only verifier](verify_noise_solver_bootstrap.py) reconstructs all logical bank mappings, reproduces the 5,000-draw interval, and compares 16 draws against independent direct-index resampling. It requires neither checkpoints nor raw board archives.
