# Noisy adaptive acquisition: exploratory old-bank feasibility

Six previously used3000-boardbanks,not fresh confirmation; no solver inference.

eta=.01, q1=1/9, alpha=.01 per learned model. Every25-cellmask precedes all current-row noisy labels. Source labels are100%complete stored simulated oracles; the policy sees only its25queriednoisyvalues.

| Policy | N | Exact / 6 | Total missing true edges | Total extra edges |
|---|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — |
| uniform_e | 600 | 0 | 0 | 2624 |
| exponential_e | 600 | 0 | 0 | 1706 |
| exponential_naive | 600 | 0 | 1054 | 0 |
| uniform_e | 1000 | 0 | 0 | 496 |
| exponential_e | 1000 | 0 | 0 | 134 |
| exponential_naive | 1000 | 0 | 1670 | 0 |
| uniform_e | 1500 | 0 | 0 | 51 |
| exponential_e | 1500 | 3 | 0 | 3 |
| exponential_naive | 1500 | 0 | 2280 | 0 |
| uniform_e | 3000 | 6 | 0 | 0 |
| exponential_e | 3000 | 6 | 0 | 0 |
| exponential_naive | 3000 | 0 | 3655 | 0 |

All54,000masks,174,960,000log-LRvalues, potential noise realizations and first-crossing paths were independently replayed. Exact rational crossing thresholds were checked independently with70-digit Decimal arithmetic.

The e-process theorem protects against removing true edges with probabilityatmost1%permodel under the specified noise law. It does not guarantee exact recovery, extras, solve accuracy, or arbitrary stopping certificates. The first-equality control does not have that protection. This exploratory study selects a later independent test endpoint and is not pooled with fresh results.

[Predictable-mask proof](NOISY_ACQUISITION_EPROCESS.md) · [Protocol](protocol-noisy-pilot.json) · [Policy](noisy_policy.py) · [Acquisition](noisy_pilot.py) · [Audit](audit_noisy_pilot.py)
