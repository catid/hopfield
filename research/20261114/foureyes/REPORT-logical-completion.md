# Adding implied constraints changes search without changing its zero set

**The frozen one-point improvement hypothesis is inconclusive.** On 13 fixed, sound BP AdamW supports missing true Sudoku edges, adding their logically implied edges raises short-cap accuracy from **78.548462% to 79.270000%**. The gain is **0.721538 percentage points**, with a paired input-cluster 95% interval **[0.209981, 1.230019]**. Both arms have zero wrong stops. The interval excludes zero but straddles the prespecified +1-point target.

| Historical source models | Fixed models | Original graph accuracy | Completed graph accuracy | Difference, pp | Paired 95% interval, pp |
|---|---:|---:|---:|---:|---|
| BP AdamW + transform, primary | 13 | 78.548462% | 79.270000% | +0.721538 | [0.209981, 1.230019] |
| Paper SGD, 32 updates + transform, descriptive | 8 | 81.168750% | 81.640000% | +0.471250 | [−0.242625, 1.160125] |
| Paper SGD, 256 updates + transform, descriptive | 11 | 81.015455% | 81.640000% | +0.624545 | [−0.095455, 1.307318] |

These optimizer subsets were selected separately and are not an optimizer comparison. Models and search seed are fixed; intervals quantify variation in the shared fresh inputs only, averaging all models inside each input cluster. They do not include new training-bank or search-seed uncertainty.

## Intervention and exact invariance

The preceding [logical certificate study](LOGICAL_SOUNDNESS.md) proves that all 32 selected models are zero-stop sound despite missing true edges. For each, compare its unit-weight support $F$ with $F'=F\cup G$, where $G$ is the supplied 810-edge Sudoku peer graph. Every added edge is already logically implied at zero energy. Since both energies are nonnegative,

$$E_F(y)=0\quad\Longleftrightarrow\quad E_{F'}(y)=0.$$

Their nonzero energies, fields and search trajectories can differ. Extra false exclusions in the original model remain; completion does not make every support equal to the true graph. Sudoku geometry is explicitly supplied to this diagnostic intervention. This is an adaptive follow-up on previously examined models, with the selection and fresh comparison frozen before new inference.

All arms use four chains, 262,144 moves per chain, search seed 0, unit weights, zero bias and stopping threshold, focus threshold 0.45, temperatures 0.4→0.31, annealing period 524,288, and checks every 4,096 moves. The inherited search loop is unchanged. Each unique support passes eight native/capture/sparse state-parity checks on 16 inputs, followed by exact integer field/energy checks. The fresh cohort is official `full-all[320000,330000)`, 10,000 shared puzzles from the reused corpus.

The 64 logical comparisons reduce to **30 unique support/settings searches**. Repeated models map to the same outcomes with their original weights retained. Four GPU workers complete the queue in **37.854 seconds**, within the frozen 300-second/20-GPU-minute envelope. Live receipts show completed-job counters increasing and active final workers at 100% utilization; idle GPUs in the final tail have no pending eligible jobs. There are no new fits or duplicate searches.

An independent audit verifies **300,000 returned boards**, givens, all Sudoku units, integer fields and energies, saved-state/output correspondence and all stopped states. The public verifier reconstructs every original/completed support from the earlier certificates, checks exact hashes and logical mappings, and reproduces all three intervals from compact validity/stopping bits. The primary uses 5,000 paired input resamples with RNG 20261115555. Secondary intervals are descriptive and are not simultaneous bounds.

```sh
uv run --with numpy python research/20261114/foureyes/verify_logical_completion.py
```

[Frozen protocol](logical_completion.json) · [Preparation](prepare_logical_completion.py) · [Search adapter](logical_completion_search.py) · [Bounded launcher](launch_logical_completion.py) · [Saved-state audit](audit_logical_completion.py) · [Public verifier](verify_logical_completion.py) · [Summary and source hashes](../../../results/research/foureyes/logical_completion/summary.json) · [Per-model results](../../../results/research/foureyes/logical_completion/per_model.csv) · [Queue execution receipt](../../../results/research/foureyes/logical_completion/launch-result.json)
