# H1o: direct counting versus fixed pseudolikelihood recipes on raw boards

The four frozen count-versus-PL contrasts at 100 and 300 raw complete boards give **supported_joint_noninferiority** under the five-point margin. Each contrast uses a 98.75% paired bootstrap interval, with Bonferroni correction for the four-contrast family. All 480,000 returned boards were independently checked.

| Raw training boards | Method | Mean accuracy (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside this fixed-PL-recipe comparison; not run | — | — |
| 30 | count | 4.7433 | 0 |
| 30 | pl_raw | 0.0000 | 36 |
| 30 | pl_digit | 0.0000 | 60000 |
| 100 | count | 80.4033 | 0 |
| 100 | pl_raw | 0.0000 | 23641 |
| 100 | pl_digit | 0.0417 | 59936 |
| 300 | count | 80.4033 | 0 |
| 300 | pl_raw | 0.0000 | 59550 |
| 300 | pl_digit | 26.6850 | 40706 |

Both PL arms use the original 32-update proximal pseudolikelihood recipe and unchanged fitted weights. `pl_raw` has no augmentation; `pl_digit` uses the original per-update digit relabeling. All arms preserve the official puzzle masks associated with the sampled complete boards. No spatial augmentation or largest-gap transformation is used.

Count graphs at 100 and 300 boards are byte-identical, with the same unit, stopping/focus statistics, search seed and test inputs. The six count searches are explicitly reused across these sizes. The six 30-board count graphs remain distinct.

The oracle column measures which fixed test solutions have zero count energy. It is a ceiling for correct certified stops; overall returned-board accuracy can exceed it because an unstopped best board may be correct at positive energy. The diagnostic never feeds back to graph construction or search.

Evaluation uses full-all rows [60,000,70,000), six seeds, four chains and 262,144 moves per chain. A documented pre-inference assertion correction enforces the already frozen literal count unit of one when a floating mean reduction returns 0.99999994. Completed earlier runs were unchanged.

[Frozen protocol](protocol-raw-compare.json) · [Original runner](raw_compare.py) · [Corrected runner](raw_compare_v2.py) · [Erratum](ERRATUM-raw-compare.json) · [Audit](audit_raw_compare.py) · [Tracked results](../../../results/research/ripper/raw_compare.json)
