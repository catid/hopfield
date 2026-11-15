# H1x: independent256-bank N600 recovery confirmation

The frozen >=90% exact-recovery target is **supported_at_least_90percent_recovery**. Exponential recovered243/256exact graphs atN600 (94.9219%). The exact binomial one-sided95%lower bound is92.0475%; the separate one-sided95%upper is96.9705%. Support requires at least239/256exact. These are not a95%two-sided interval.

| Policy | Exact N600 graphs / 256 | Total extra edges |
|---|---:|---:|
| BP AdamW: outside this structural study; not run | — | — |
| uniform | 13 | 788 |
| candidate | 189 | 85 |
| exponential | 243 | 13 |

There are150,504unique source boards among153,600bank-row instances, with3,096cross-bank repeated instances. Every bank samples600boards without replacement, independently of other banks. All110,220previously used source rows are excluded. Exact binomial interpretation assumes this independent sampling/tie-stream mechanism.

All policies remain fixed, with25querieddigitsperboard and15,000labelsperbank. Source-draw, tie, and shadow RNG ranges are disjoint. The full study and sampling protocols froze before input generation; the input hash registration preceded all acquisition. All460,800mask decisions were independently replayed;768supports and1,536hidden-value recounts passed.

This study has a construction-only primary. Candidate and uniform controls are descriptive. It does not establish superiority, universal recovery, or noise tolerance. The earlier61/64H1w result remains inconclusive and is not pooled or extended.

445exact supports were independently matched to the already-audited unit1,B0,true810-edge energy. No new solver runs or accuracy observations were generated; exact energy identity does not create independent test evidence. CPU wall time from preparation through final audit was289.39s; GPU usage was0minutes.

[Protocol](protocol-recovery-confirm.json) · [Acquisition](recovery_confirm_acquire.py) · [Independent replay](audit_recovery_confirm_acquisition.py) · [Final audit](audit_recovery_confirm.py) · [Public results](../../../results/research/ripper/recovery_confirm.json)
