# H3b: weight decay and finite search budget

**Preregistered result: supported.** Removing weight decay at fixed learning rate 0.3 was compared with the original AdamW control and preconditioned SGD on all 6,513 unique official n=10 test inputs. Each arm used 20 chains under three search seeds; all fits used the same 97,554 fitting rows and 4,000 updates. No settings were selected on test results.

| Method | N20 accuracy (%) | Coverage (%) | Wrong stopped chains | Capped chains | Mean moves / chain |
|---|---:|---:|---:|---:|---:|
| BP AdamW, decay 0.01 | 99.605916 | 92.283227 | 0 | 22,262 | 5021.70 |
| BP Adam, no decay | 100.000000 | 93.559127 | 0 | 7,806 | 3313.01 |
| SGD + geometry preconditioner | 99.984646 | 93.585165 | 0 | 5,690 | 2860.85 |

No-decay Adam versus preconditioned SGD: accuracy difference +0.015354 percentage points, coverage difference -0.026039 points. One-sided 95% bootstrap lower/upper bounds are [0.000000, 0.046062] for accuracy and [-0.206310, 0.142826] for coverage. These are the two one-sided endpoints, equivalent to a 90% two-sided interval.

The exact regression safeguard observed 0 unique inputs with any paired accuracy regression; its one-sided 95% upper probability is 0.045986%. The preregistered noninferiority margins are 0.1 point for accuracy and 2 points for coverage. Search repeats stay grouped within each input for uncertainty.

## Same-trajectory budget comparison

The following uses the first 1,024 canonical unique test inputs. The 20,000-move output is captured from the exact trajectory continued to 80,000 moves. Previously stopped boards, energies, moves and FLOPs remain unchanged. This is a fixed subset; it is separate from the differently batched primary full-test run.

| Method | Budget | N20 accuracy (%) | Coverage (%) | Mean moves / chain | Capped chains |
|---|---:|---:|---:|---:|---:|
| BP AdamW, decay 0.01 | 20,000 | 99.348958 | 89.040698 | 5058.50 | 3,671 |
| BP AdamW, decay 0.01 | 80,000 | 99.479167 | 89.147287 | 6031.99 | 210 |
| BP Adam, no decay | 20,000 | 100.000000 | 90.930233 | 3308.05 | 1,289 |
| BP Adam, no decay | 80,000 | 100.000000 | 90.930233 | 3561.00 | 31 |
| SGD + geometry preconditioner | 20,000 | 100.000000 | 90.775194 | 2912.55 | 1,005 |
| SGD + geometry preconditioner | 80,000 | 100.000000 | 90.775194 | 3093.02 | 16 |

The CPU parity check caught upstream RNG coupling across blocks before production: extending an earlier block shifts the RNG stream of later blocks. The snapshot solver therefore rejects multi-block cohorts. The declared secondary cohort fits one block; a full 1,024-input, 20-chain production comparison also exactly matches the original 20,000-move search. Both proof and archived prelaunch revision are retained.

Post-inference diagnostics in the JSON recompute float64 single-cell certificate gaps on saved returned boards, separated into correct/wrong and stopped/capped groups. These diagnostics never steer the solver. For capped chains, the returned board is the best-energy visited board, not necessarily the last state.

No-decay Adam uses the same tested AdamW update routine with weight decay set to zero; direct updates were verified bitwise equal to `torch.optim.Adam`. This intervention changes only weight decay. The two cached control fits are hash-identical to H3.

[Frozen protocol](protocol.json), [runner](driver.py), [snapshot solver](capture_solve.py), [auditor](analyze.py), [tracked results and hashes](../../../../results/research/loki/summary.json), [per-seed counts](../../../../results/research/loki/per_seed.csv).
