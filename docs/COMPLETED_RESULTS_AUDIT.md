# Completed result audit

Audit snapshot: 2026-11-14T06:39:38.930614+00:00. The captured `REPRODUCTION.md` SHA-256 is `2e99e2be426f6303a2f8c7e96f0cc3075d18a4a26354ac8fc573d666a2a4fd80`.

This CPU-only audit independently checks saved N-Queens and graph-coloring boards, enumerates exact solution counts, repeats the maze BFS/path checks, and recomputes Sudoku status/FLOP aggregates. It revalidates completed older-controller markers against their model and output hashes. No fitting or inference was rerun.

| Evidence family | Audit result |
|---|---|
| sudoku_learned | validated |
| sudoku_hand | validated |
| mazes | validated |
| nqueens_8 | validated |
| nqueens_10 | validated |
| graph_coloring_8 | validated |
| graph_coloring_10 | validated |
| completed_controller_markers | validated |
| tests_and_compute_comparison_receipts | validated |
| maze_objective_receipts | validated |
| table4_halting | validated |
| timing_and_optimization_receipts | validated |

Runtime: 9.65 seconds. Full metrics, errors, per-seed control availability and hash-linked evidence are in [completed_controls_audit.json](../results/reproduction/completed_controls_audit.json).

Sudoku full-test records retain correctness flags but no final boards; aggregate agreement does not provide independent Sudoku board verification. Objective runs retain evaluator summaries and receipts but no saved per-maze boards. Test receipts are checked against source hashes without rerunning their tests. Published baseline comparisons remain imported evidence.
