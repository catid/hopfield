# Conservative noise calibration: implementation pilot

This six-bank pilot checked implementation and cost. N1500 and the 10 pp fresh-study noninferiority margin were fixed before it ran. Actual label noise stays at 1%, with fixed q1=1/9 and 25 queried digits per board.

| Assumed rate | Exact graphs / 6 | Missing true edges | Extra edges |
|---|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — |
| 1% exact | 3 | 0 | 3 |
| 2% conservative | 0 | 0 | 36 |
| 0.5% unprotected | 5 | 0 | 1 |

All 27,000 masks and 87,480,000 log-LR values were independently replayed. The exact arm reproduced the original pilot’s first 1,500 masks, queried values, log paths, and support exactly. No inference was run.

The conservative calibration retains the true-edge deletion bound under the stated law. The underbound lacks that guarantee even though this finite pilot observed no missing true edges. No pilot outcomes are pooled with the fresh study.

[Protocol](protocol-noise-bound-pilot.json) · [Proof](NOISE_UPPER_BOUND_EPROCESS.md) · [Pilot](noise_bound_pilot.py) · [Audit](audit_noise_bound_pilot.py)
