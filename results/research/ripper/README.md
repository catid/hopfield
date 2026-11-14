# Ripper research evidence

The strongest confirmed result is digit-label symmetry projection: **+48.31 percentage points** (approximate paired 95% interval **43.11–53.65**) over unchanged fitted CRAM parameters, on new development rows 512–2559, k17/21 and six model/search seeds. The fixed search budget is 262,144 moves per chain, four chains; this is not the paper main-result budget.

| Method/control | Scope |
| --- | --- |
| BP AdamW | Outside this post-training intervention protocol; not run |
| Original CRAM | Measured matched controls, same models, puzzle rows and move budgets |
| Digit-symmetric CRAM | Mean diagonal/off-diagonal pair tables and digit-mean unary terms; no supplied Sudoku peer graph |
| Canonical repulsive CRAM | Drop the constant energy gauge, then clip negative equal-digit couplings |

The preregistered repulsion hypothesis of at least +10 points was falsified: observed **+1.87 points**, interval **0.86–2.82**. Repulsion removed all observed wrong stops in its study. All 18 repulsive models also satisfy a post-hoc sufficient exact-energy validity condition; this is not a worst-case floating-point drift guarantee. Low-data k1/3 models still solve zero puzzles and projection worsens their stopping behavior.

`summary.json` contains grouped plot inputs, hypotheses, decisions, confidence intervals, scope and hashes. `per_seed.csv` is the common source for per-seed plots. The four `audit-*.json` files preserve compact audited outcomes. `certificate.csv` and `certificate.json` retain the post-hoc energy-gap evidence and numerical caveat.

Frozen protocols and implementation are under [research/20261114/ripper](../../../research/20261114/ripper/). Detailed reports there describe matched controls, saved-board validation and protocol errata. Large saved boards, fitted parameters, training examples and source dataset files remain under ignored `artifacts/` and `upstream/` paths; a source-only checkout does not include these inputs. Reconstruct them using the repository reproduction/data setup and the frozen per-input SHA256 manifests before running the research workers. The research entrypoints intentionally require matching archived inputs and Ripper host ownership.

The compact exports also include the audited sparse-backend and fixed held-out studies described below.

The sparse backend audit now reports **2.824×** geometric-mean warm inference speedup with exact measured state/board parity; see `sparse.json` and `sparse_timings.csv`. The fixed held-out study reports **+47.83pp**,95% interval **42.72–52.85pp**, with **0** repulsive wrong stops. See `heldout.json` and `heldout_per_seed.csv`.
