# H1as: adapt acquisition to the supplied exact equality null

The frozen conjunction is **supported_finite_bank_conjunction**: 32/32 banks retain every true edge, with 2,588 extra edge-bank pairs versus the ceiling of 7,114.

| Mask policy at exact q0 | All true edges retained | Exact graphs | Missing true edges | Retained extras |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| archived_fixed_mask_exact_law | 32/32 | 0/32 | 0 | 14229 |
| adaptive_exact_law | 32/32 | 0/32 | 0 | 2588 |

Pooled extras changed by 81.8118% relative to the frozen H1ar exact-law replay. The empirical target requires at least a 50% reduction and at least 29 banks retaining every true edge; exact graph recovery is descriptive.

Every new acquisition uses the supplied exact permutation-law q0 = 9/1600 from the first row, with unchanged q1 = 1/9, boundary 648000, exposure selector and 25 queries per training board. The current mask is fixed before the current corruption draw. All 32 source banks, potential event/permutation draws and original row-indexed tie streams match H1ap. Histories and queried masks may now diverge under the new exclusions.

The independent auditor reconstructed all 35,520 masks, noise draws and 115,084,800 cumulative log-likelihood values, plus all 103,680 first-crossing paths. 34,919 masks differ from the archived acquisition. All 888,000 new training queries are charged, including labels on corrupted rows; no calibration is fitted or queried.

Known-law information is an explicit supplied prior. The exact-null martingale argument applies to the independent board-permutation law with predictable current masks; this study does not claim arbitrary correlated-noise robustness. The comparison uses already-seen source banks and common latent draws, not fresh-source confirmation.

Each bank spends 27,750 training queries. No new source board, solver input or GPU work is used. The archived H1ar fixed-mask comparator is not reacquired.

[Original scientific protocol](protocol-adaptive-null.json) · [Separately authorized continuation](protocol-adaptive-null-continuation.json) · [Independent eager replay](audit_adaptive_null_eager.py) · [Results](../../../results/research/ripper/adaptive_null.json)

The original run failed its 120 CPU-second gate: the monitor triggered at 120.004219 CPU-seconds and 15.055996 wall seconds. Complete final CPU accounting for that attempt is unavailable. Eight acquisitions finished; no aggregate scientific summary was produced. A lazy NPZ-loading bug repeatedly decompressed arrays during audit. Original code, failed logs and artifacts remain preserved. A separate additional-budget continuation eagerly loads the same arrays, reuses all eight completed acquisitions byte-identically, and acquires only the remaining 24 banks. Its bounded equivalence and one-bank preflight are charged to the new budget; the original gate remains failed.

[Eager-only diff](adaptive-null-eager-audit.diff) · [Preserved original artifacts](adaptive-null-original-artifacts.json)

The separately authorized corrected continuation used 25.027003 additional CPU-seconds and 4.591258 execution wall seconds, including its 1.480072 CPU-second / 1.504345-second preflight. This cost is reported separately from the failed original attempt; no combined exact CPU total is available.
