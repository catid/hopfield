# H1: digit-permutation projection, first screen

Frozen hypothesis supported in this development screen: projected minus original accuracy = **45.2799 percentage points**, approximate paired two-way bootstrap 95% interval **[39.0617, 50.5208]**. The preregistered threshold was 5 points. There are only three independent training-seed clusters; independent confirmation is required.

All 36 searches completed under a 262,144-move-per-chain cap, with four chains and 512 development puzzles each. All 18,432 saved returned boards were independently checked for row, column, box and clue validity, as well as exact labels.

| k | Arm | Solved, seeds 0 / 1 / 2 (of512 each) | Mean accuracy (%) | Wrong stops (total) |
|---|---|---|---:|---:|
| — | BP AdamW: outside post-training intervention protocol | Not run | — | — |
| 17 | original | 70 / 36 / 34 | 9.1146 | 1383 |
| 17 | projected | 341 / 332 / 336 | 65.6901 | 0 |
| 17 | projected_original_hbar | 341 / 332 / 336 | 65.6901 | 0 |
| 21 | original | 207 / 121 / 145 | 30.7943 | 921 |
| 21 | projected | 336 / 323 / 336 | 64.7786 | 0 |
| 21 | projected_original_hbar | 336 / 323 / 336 | 64.7786 | 0 |
| 30 | original | 337 / 331 / 332 | 65.1042 | 20 |
| 30 | projected | 344 / 342 / 340 | 66.7969 | 0 |
| 30 | projected_original_hbar | 344 / 342 / 340 | 66.7969 | 0 |
| 1000 | original | 334 / 334 / 339 | 65.5599 | 0 |
| 1000 | projected | 341 / 343 / 350 | 67.3177 | 0 |
| 1000 | projected_original_hbar | 341 / 343 / 350 | 67.3177 | 0 |

The projection uses only global digit-label symmetry: no Sudoku spatial peer graph is supplied. Training-only threshold and focus statistics are recomputed. The additional original-hbar arm gives identical saved boards, stops and move counts on these runs. Training hbar changes only by floating-point averaging roundoff (maximum relative change1.48e-7), so this control does not isolate a material thermal-scale change here.

Search wall times were 0.75–1.68 seconds per arm under concurrent independent jobs; these short observations are not isolated latency benchmarks. Reduced wall time of the original low-data models often reflects incorrect early stops and is not useful algorithmic speed.

[Protocol](protocol.json) · [Runner](run.py) · [Audit](../../../results/research/ripper/audit-discovery.json)

Capture parity: initial proof mislabeled the number of STATE tensors; the supplementary proof verifies all 25 state_dict fields, including generator and scalar loop state, over four checks. The active worker source was not modified.

The k=1000 archived fitted tensors are identical across seed identifiers; that secondary cell varies only the inference seed. The primary k17/21 cells use distinct training subsets/checkpoints.

Follow-up [confirmation](REPORT-confirmation.md) uses2,048disjoint development rows andsixtrainingseeds; [low-data extension](REPORT-lowdata.md) evaluatesk1/3/10. These reports distinguish the new measurements from the initialscreen.
