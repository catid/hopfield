# More chains do not meet the fixed-budget accuracy-gain target

At the same maximum envelope of **16,777,216 chain moves per puzzle**, sixteen chains at 1,048,576 moves each solved 99.5917% of evaluations, versus 99.5883% for four chains at 4,194,304 moves each. The gain was **+0.0033 percentage points**, with paired two-sided 95% bootstrap interval [−0.0817, +0.0834]. The frozen +0.1-point improvement hypothesis is **falsified** under its prespecified decision rule. This does not rule out other allocation strategies or smaller effects.

| Chain allocation | Maximum 1,048,576 chain moves | Maximum 4,194,304 chain moves | Maximum 16,777,216 chain moves |
|---|---:|---:|---:|
| BP AdamW | Outside this inference-only comparison | No matched local result | — |
| One chain | 78.2967% | 97.3817% | 99.5917% |
| Four chains | 79.9617% | 97.2767% | 99.5883% |
| Sixteen chains | 79.1583% | 97.4483% | 99.5917% |

Only the final sixteen-versus-four comparison is primary. Earlier points and the single-chain arm are descriptive. At the primary endpoint, 130 input–seed cases improved and 128 regressed. Every arm had zero wrong stops. Similar aggregate accuracy does not imply identical trajectories or solved-input sets.

The cohort contains 10,000 official puzzles at `full-all[80000,90000)`, with six search seeds and the same previously frozen unit-weight count energy. There are no restarts. Each chain uses the same temperature range, focus rule and 524,288-move annealing period; the number of annealing cycles consequently differs. Input identities and seed labels are paired, but random streams are not identical across chain counts.

These are equal **maximum** chain-move budgets, not equal actual work or wall time. Early stopping, compaction, parallel scheduling and check granularity affect computation. Runtime counters are retained descriptively. No speedup claim is inferred from these quality experiments. The new research slice remains part of the corpus used in the original reproduction.

All 18 jobs passed eight native-general/captured-general/captured-sparse state checks at their assigned chain count. The independent audit validated **540,000 saved boards**, official input identities, all source/output hashes, exact integer fields and energies, and preservation of earlier stopped boards. Uncertainty uses the frozen 5,000 paired bootstrap draws over six seeds and 10,000 shared puzzle indices. Joint all-six success statistics are descriptive.

[Frozen protocol](chain_budget.json) · [runner](chain_budget.py) · [auditor](audit_chain_budget.py) · [public summary](../../../results/research/foureyes/chain_budget/summary.json) · [per-seed results](../../../results/research/foureyes/chain_budget/per_seed.csv)
