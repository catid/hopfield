# H1ap: a histogram-preserving corruption evades the balance filter

The frozen target of at least 16 banks losing a true edge is **supported_finite_bank_target**: 21/32 banks lose edges.

| Corruption and filter | Missing true edges | Retained extras | Rejected training rows | Accepted corrupt rows |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Archived H1an collapse + balance filter | 0 | 54 | 1961 | 0 |
| Uniform position permutation + balance filter | 41 | 45 | 0 | 2071 |

Each board draws one Bernoulli(9/160) corruption event and one uniform permutation of its 81 cell positions. All current queries and repeated calibration labels share that shuffled board. The current adaptive mask is fixed before either draw. Every digit still occurs exactly nine times, so no 25-label subset can violate the supplied digit-balance prior. The filter accepts every row.

The exact fixed-cell marginal error is (9/160)(8/9) = 5%; each wrong digit has probability 1/160. A uniform permutation maps two distinct queried positions uniformly to 6,480 ordered source-position pairs, of which 648 have equal labels. Thus true-pair equality is (9/160)(1/10) = 9/1600 = 0.5625%, while calibration repeats always agree. These laws were checked independently using exact fractions and exhaustive ordered-pair counts, not factorial permutation enumeration.

All 35,520 adaptive masks, noise draws, observed labels, log-likelihood rows and first crossings were independently replayed. Every permuted board preserved its full digit histogram. An independent replay without filtering matched the complete filtered state on every row. All 2,071 corrupted training rows passed the filter; no row was rejected.

The observed adaptive-query error was 5.1773%; realized true-pair equality was 0.5837%. These measurements are distinct from the exact fixed-cell and conditional-pair laws. The iid-based calibration estimate remains below nominal 5% in all 32 banks.

This is a prospective corruption-law boundary test on already-seen H1an source banks, with new observations and frozen noise streams. Each new fit uses M1000/N1110 and 29,750 queries. There are 32 new calibrations/acquisitions; the identical unfiltered replay is an audit, not another independent experiment. No new source board, solver input or GPU work is used.

The supplied multiplicity prior can detect shared collapse, but it cannot certify cleanliness under arbitrary shared corruption. This study measures graph damage only; it does not measure invalid solver certificates.

[Protocol](protocol-permuted-noise.json) · [Exact law check](test_permuted_noise_law.py) · [Independent replay](audit_permuted_noise_acquisition.py) · [Results](../../../results/research/ripper/permuted_noise.json)

Execution used 0.8695 aggregate CPU-min and 6.6759 seconds wall time, within the frozen15 CPU-min/180-second ceilings. Up to eight CPU workers ran; no GPUs were allocated.
