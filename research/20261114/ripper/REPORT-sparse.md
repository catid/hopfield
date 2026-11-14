# H1d: exact sparse backend and measured speed

Warm inference geometric-mean speedup: **2.824×**, timing bootstrap95% interval **[2.415, 3.293]×**. The frozen ≥1.5× hypothesis is **supported_for_frozen_protocol**. Constructor plus inference geometric-mean speedup is **2.786×**.

The18models passed1,170 complete25field state comparisons plus best/certificate board arrays. Each used256development puzzles and64checks of4096moves. All63 paired warmup/timed full searches matched final state and boards exactly. This is extensive bounded empirical parity, not a universal floating-point proof.

| k | Puzzles | General inference (s) | Sparse inference (s) | Inference speedup | Complete speedup |
|---:|---:|---:|---:|---:|---:|
| 10 | 128 | 0.6083 | 0.2468 | 2.465× | 2.415× |
| 10 | 512 | 1.5414 | 0.4085 | 3.773× | 3.718× |
| 10 | 2048 | 6.1660 | 1.6112 | 3.827× | 3.813× |
| 17 | 128 | 0.5170 | 0.2479 | 2.086× | 2.047× |
| 17 | 512 | 0.8624 | 0.3356 | 2.569× | 2.529× |
| 17 | 2048 | 3.4521 | 0.9629 | 3.585× | 3.561× |
| 21 | 128 | 0.5149 | 0.2493 | 2.066× | 2.022× |
| 21 | 512 | 0.8399 | 0.3444 | 2.439× | 2.403× |
| 21 | 2048 | 3.1875 | 0.9619 | 3.314× | 3.292× |

Stock SparseSearch fails its uniform-degree assertion on these learned graphs. The derivative pads with unique zero-weight neighbors, forces bias-aware arithmetic with unit hscale, preserves floating fields and the original GeneralSearch refresh, and changes no model energy/focus offsets. No one-level integer approximation is used.

Timing uses the same exclusive GPU for paired backends, alternating order across six repetitions after one warmup per backend. Each search uses262144moves per chain and fourchains. Reports include constructor/conversion setup separately. Loading input files and constructing the shared dense S tensor are outside timing. Capture, compaction and periodic refresh are included. CUDAevent move-call timings include device preparation.

[Frozen protocol](protocol-sparse.json) · [Backend](sparse_backend.py) · [Preflight](sparse_preflight.py) · [Benchmark](sparse_benchmark.py) · [Public evidence](../../../results/research/ripper/sparse.json)
