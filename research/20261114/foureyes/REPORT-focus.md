# Occasional unfocused moves miss the improvement target

Replacing one in sixteen focused 4,096-move blocks with uniform free-cell selection did **not** meet the frozen +0.1 percentage-point accuracy-gain target. At 4,194,304 moves per chain, the mixed schedule solved 99.5283% versus 99.5883% for focused search: a difference of **−0.0600 points**, paired two-sided 95% bootstrap interval [−0.1533, +0.0283]. The hypothesis is falsified under the prespecified decision rule.

| Cell-selection rule | 262,144 moves/chain | 1,048,576 moves/chain | 4,194,304 moves/chain |
|---|---:|---:|---:|
| BP AdamW | Outside this inference-only comparison | No matched local result | — |
| Always focused on conflicted cells | 80.4433% | 97.4400% | 99.5883% |
| Fifteen focused blocks, one uniform block | 79.3700% | 97.0900% | 99.5283% |
| Always uniform over free cells | 7.6283% | 42.6000% | 71.7183% |

The uniform-only arm and earlier checkpoints are descriptive. Every arm had zero wrong stops. The primary comparison contains 122 input–seed improvements and 158 regressions. This rejects the tested effect size for this fixed schedule; it does not rule out other exploration mechanisms.

All arms use the same fixed count energy, 10,000 official puzzles at `full-all[90000,100000)`, six search seeds, four chains and the prior thermal schedule. Mixed exploration uses the pinned uniform kernel for zero-indexed blocks 15, 31, 47, and so on. The normal 4,096-move refresh reconstructs focus flags before the next block. No new move kernel or energy is introduced.

Each of 18 jobs passed 33 native-general/captured-general/captured-sparse checks. The native reference used a separately specified mode schedule through two uniform blocks and back to focused mode. The independent audit checked **540,000 saved boards**, exact integer fields and energy, reconstructed focus flags, block counts, official input identities and all source/output hashes. Earlier stopped boards and their move counts were preserved.

Uncertainty uses 5,000 paired bootstrap resamples over seed labels and shared puzzle indices. Equal move limits are not identical computation: uniform selection skips focus bookkeeping. Inherited FLOP estimates assume focused mode and must not be used as a mixed-schedule compute comparison. Wall time is descriptive. This follow-up cohort belongs to the corpus evaluated during the original reproduction.

[Frozen protocol](focus.json) · [runner](focus.py) · [mode scheduler](focus_schedule.py) · [independent auditor](audit_focus.py) · [public summary](../../../results/research/foureyes/focus/summary.json) · [per-seed rows](../../../results/research/foureyes/focus/per_seed.csv)
