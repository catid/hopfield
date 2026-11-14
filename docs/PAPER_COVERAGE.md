# Paper result coverage

The active scope is CRAM's main results, controls, and ablations, with published baseline comparisons. This inventory follows arXiv:2611.17309v1; the much broader initial source inventory also contains baseline training and legacy exports that are outside this scope. A queued experiment is not a completed result.

| Paper item | Completed local evidence and declared limits |
|---|---|
| Full paper, equations, references, appendices | [Reviewed Markdown](../references/2611.17309v1.md), [conversion audit](../references/CONVERSION_REVIEW.md), original PDF, LaTeX, bibliography, and vector figures retained. |
| Tables 1 and 5: parameterizations and recipes | Independent implementation and numerical checks in [implementation notes](IMPLEMENTATION.md); pinned recipes and exact run settings retained. |
| Table 2: Sudoku and Maze-Unique | Fresh three-seed fits and complete test sets, hand-set controls, local warm/cold fit timing, and paper-convention arithmetic counts completed. [Sudoku audit](SUDOKU_OPERATION_AUDIT.md), [maze results](MAZE_REPRODUCTION.md). |
| Table 3: N-Queens and graph coloring | Fresh models and hand-set controls completed at both sizes and all three seeds. [Multi-solution results](MULTI_SOLUTION_REPRODUCTION.md) separates every measured and published cell. |
| Table 4: stopping rules | Fresh CRAM full-test counts and published baseline comparisons completed. [Stopping-rule report](HALTING_REPRODUCTION.md). |
| Table 6: objective definitions | Twelve objectives implemented in the pinned shared harness; independent mathematical checks and a separate EqProp gradient diagnostic retained. |
| Table 7: twelve training objectives | All 84 runs complete: 72 standard, nine intermediate/long, and three 3,840,000-update full-BPTT runs. Final metrics, development crossings and operation counts are validated in [objective report](OBJECTIVE_REPRODUCTION.md). |
| Table 8 and Appendix F: Maze-Hard | Fresh CRAM+z fits and all 1,000 test mazes completed for seeds 0–2. Every seed solves 999, matching the paper. The separate frozen-recipe 900-fit/100-development check solves 100/100 for all three seeds, with independent saved-board validation. [Maze report](MAZE_REPRODUCTION.md). Baselines remain published-only. |
| Figure 1: schematic | Original vector diagram retained in the reviewed paper; the implementation realizes its training/search separation. This illustration is not a separate experiment. |
| Figure 2a: compute curves | Exported from saved fresh full-test records, with explicitly imported baseline sweeps. Matching the paper's accuracy rule gives 275.4× versus published EBR, compared with the paper's 282×. |
| Figure 2b: learned weights | Fresh learned tensors inspected, plotted, and checked against task rules. [Coupling audit](LEARNED_COUPLING_AUDIT.md). Weak non-rule weights remain visible. |
| Figure 3a: difficulty | Complete. Three full zero-energy searches passed the 10,000-puzzle development gate and exported the rows 10000–19999 curves. The main table's fitted-threshold records remain separate. [Protocol details](REMAINING_CRAM_CONTROLS.md). |
| Figure 3b: data efficiency | All 102 runs and 17 six-seed points validated; final plot and independent count/hash audit complete. The runs use the explicitly seeded recreation of the missing reference arrays. [Results and protocol](DATA_EFFICIENCY_REPRODUCTION.md). |
| Figure 4: training curves | All 100 runs and 9,740 probes complete, with all 12 dense-clock checks validated. [Protocol audit](FIGURE4_AUDIT.md). |
| Figure 5: hand-set weights | Generated from the actual hand-set energy definitions, with independent constraint-support checks. Full inference controls are reported separately. |
| Figure 6: latent Maze-Hard dynamics | Fresh seed-0 model replay completed, including hidden field, pointer structure, answer snapshots, stopping, and continued energy descent. |
| Appendix C.4: Maze-Unique dead-end and empty-board claims | Learned-kernel enumeration and development-set energy comparisons completed in the coupling audit. |
| Appendix C.5: N-Queens free pairs and preconditioning | Full frozen-recipe controls completed at both sizes and all three seeds. Their outputs are retained separately from the transferred size-8 main recipe. |
| Appendix D.2: implicit differentiation at 60,000 updates | All nine development-only jobs complete (three frozen learning rates × three seeds); no test tuning. Maximum 3/512 solved. [Supplemental audit](D2_SUPPLEMENTAL_AUDIT.md). |
| Appendix F: failed latent ablations | All four controlled reconstructions complete: token source 0/300, edge gates 0/100. Missing historical recipes and seed identities prevent an exact reproduction claim. [Audit](REMAINING_CRAM_CONTROLS.md). |

The [main report](../REPRODUCTION.md) and per-family machine-readable collectors carry measured values and current progress. This inventory is a coverage map, not a substitute for their completion checks.

The following boundaries remain explicit:

- The paper's H200 training-time ratios cannot be independently established by local Blackwell CRAM timing and published baseline timing. Cold/warm timing protocols also differ in the paper.
- The original data-efficiency reference was generated without a seed and was not released. The seeded recreation preserves the recipe, not the original sampled rows.
- Appendix D.2's exact gradient-cosine diagnostic and beta=0.5 divergence experiment lack a checkpoint/batch/trajectory recipe. A new declared gradient check is available; it is not the missing experiment.
- Appendix F's plain-CRAM Maze-Hard failure statement lacks a frozen fit/search recipe. It remains a published limitation claim. Applying the Maze-Unique model to Maze-Hard would not reproduce that unspecified training experiment.
- `training_curves.py` also exports 100 older Maze/Adam curves. They are absent from this paper's figures and numerical claims, and their historical choices are missing. They are an optional release artifact, outside the active paper-results scope; the required Maze/SGD objective comparison is included.
- Baseline training and a new hyperparameter search are outside the confirmed scope. Published frozen choices are reused without local test-driven selection.
