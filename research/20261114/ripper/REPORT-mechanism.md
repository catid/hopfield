# H1c: gauge and repulsion mechanism

All54 searches and110,592 saved boards validated. Clipping negative canonical couplings changes accuracy by **1.8663pp** versus the signed canonical arm; paired two-way bootstrap95% interval **[0.8571, 2.8239]pp**. The preregistered >=10pp hypothesis is **falsified_for_frozen_protocol**.

Secondary canonical-versus-raw-projection effect: **-0.2414pp**, interval[-0.8627, 0.3364]. Gauge equivalence concerns energies; the global focus reference can change trajectories. All arms use the same raw-projection thermal/focus scale and recompute training-only tau/ebar.

| k | Arm | Solved per seed (of2048) | Mean accuracy (%) | Wrong stops |
|---|---|---|---:|---:|
| — | BP AdamW: outside post-training protocol | Not run | — | — |
| 10 | projected | 702 / 557 / 565 / 670 / 595 / 522 | 29.3864 | 1342 |
| 10 | canonical | 702 / 557 / 565 / 670 / 590 / 521 | 29.3376 | 1338 |
| 10 | canonical_repulsive | 768 / 662 / 686 / 795 / 701 / 636 | 34.5703 | 0 |
| 17 | projected | 1440 / 1406 / 1450 / 1398 / 1421 / 1418 | 69.4417 | 0 |
| 17 | canonical | 1435 / 1377 / 1400 / 1387 / 1423 / 1430 | 68.7826 | 0 |
| 17 | canonical_repulsive | 1376 / 1416 / 1403 / 1420 / 1438 / 1424 | 68.9860 | 0 |
| 21 | projected | 1394 / 1392 / 1421 / 1424 / 1409 / 1422 | 68.8639 | 0 |
| 21 | canonical | 1394 / 1385 / 1437 / 1390 / 1429 / 1425 | 68.8477 | 0 |
| 21 | canonical_repulsive | 1412 / 1379 / 1414 / 1414 / 1429 / 1432 | 69.0104 | 0 |

No Sudoku spatial peer graph enters any intervention. The canonical transformation is explicit; broad invariance under arbitrary original pair/unary gauge changes is not claimed. Paired seeds use the same initialization but model-specific compaction may change later random streams.

[Frozen protocol](protocol-mechanism.json) · [Runner](mechanism.py) · [CPU audit](audit_mechanism.py) · [Receipt](../../../results/research/ripper/audit-mechanism.json)

Training-size convention: k counts unique base puzzles. Each base puzzle was fitted with the original and 99 stored symmetry variants, so k10 uses 1,000 training boards. Those inherited variants include digit relabeling, transposition, and valid band/stack/row/column permutations. The post-training transforms and search receive no additional Sudoku peer graph; the original augmentation pipeline already uses Sudoku symmetries.
