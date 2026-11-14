# Exact checkpoint-resume benchmark

The native frozen-source continuation is bitwise identical across the tested 300 W Max-Q and 450 W Workstation GPUs, but the measured 1.246–1.260× throughput gain falls below the requested 1.3× migration threshold. No production migration was performed.

| Training seed | Max-Q seconds / check | Workstation seconds / check | Speedup | Full state bytes equal |
|---:|---:|---:|---:|---|
| 4 | 1.342809 | 1.077795 | 1.2459× | Yes |
| 5 | 1.340551 | 1.064307 | 1.2596× | Yes |

Each check advances all 10,000 active puzzles with four chains by 4,096 moves. Seeds 4 and 5 began at copied checkpoint moves 9,285,632 and 10,510,336. Each device ran two independent repetitions of 16 checks from each identical initial checkpoint. One complete warm-up check was excluded, then the original checkpoint was restored before each timed repetition. Preparation, compilation, checkpoint restoration and state hashing were outside the timing.

Every GeneralSearch.STATE tensor, generator state, move index, restart interval and next-restart index matched by dtype, shape, torch.equal and raw bytes; only elapsed time was excluded. The reconstructed Sg, Bc, neighbor indices, R blocks, search signature and static scalars also matched. Exactness is established for these bounded traces, not proved for every future trajectory.

Both runs used the unchanged 46-file pinned source at commit 62bbe830c7d6f87f2ba18f30a28039258b986454, Python 3.13.13, PyTorch 2.11.0+cu128, CUDA 12.8 and Triton 3.6.0. Dataset and all copied inputs were hash-verified. Physical GPU ownership was exclusive under both lock conventions; delegated data-efficiency jobs completed before benchmarking. Production searches and outputs were untouched.

Commands and outputs are retained in the lane launch receipts and runs/{foureyes-3,work-0}/result.json. Exact tensor comparisons are in comparison.json; compare.py reproduces them from saved isolated final states. inputs.json records the copied checkpoint identity, source hashes and full test-file hash.

The separately requested two-GPU Workstation sharding benchmark passed all 23 exact-state comparisons, including pre-refresh kernel state, eight checks from the original checkpoint, eight checks after a forced-compaction fixture, repeated timing outputs and final native/sharded equality. It changes only move-kernel dispatch; original checks, refreshes, compaction, RNG chain identities and budgets remain centralized and unchanged.

For 32 checks, native times were 34.6221, 35.2480 and 35.3697 seconds; two-GPU sharded times were 17.6505, 17.7052 and 17.9135 seconds. Alternating-order medians give **1.99083×** throughput, excluding compilation, with copies included. The bounded benchmark took 190.9 seconds. Both GPU locks were released afterward.

This supports considering the reviewed two-GPU execution mode for seed5, subject to root's separate production-resume gates and latest-checkpoint provenance. The cross-host native proof and the same-host sharding proof are separate requirements. Neither proof covers an untested seed3 checkpoint or guarantees bitwise equivalence for all possible future states. No production migration occurred in this benchmark task. Evidence: work-sharded-seed5.json, its log and launch receipt, plus the source AST audit retained by root.


A subsequent native cross-host check for seed4 on ripper GPU0 also passed: 16 checks advanced the same copied state from move 9,285,632 to 9,351,168. All 24 state fields excluding elapsed, every tensor byte, the RNG state, static model tensors/scalars and native signature match the foureyes result. One timed repetition measured 1.29042 seconds/check; its purpose is the cross-host correctness gate. The process exited and both GPU locks were released. Proof: `foureyes-to-ripper-seed4-proof.json` (SHA256 `b8eb1206ad530e777e2fdeaf57f3006b7b4df1103e7d95baab9dec407a6f57b2`). This covers captured seed4 only; seed3 remains outside the measured cross-host evidence.
