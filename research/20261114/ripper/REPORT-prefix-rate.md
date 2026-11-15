# H1ax: 600 observations are insufficient for reliable smoothed-rate recovery

The frozen hypothesis is **falsified**. At N=600, the degree-20 LP produces only **7/32 independently certified integral exact graphs**, below the required 29. The other 25 outcomes are resolved procedure failures: 19 certified integral nonexact graphs and six certified fractional optima. There are no unresolved runs. At N=100 and N=300 no bank is recovered exactly; the archived N=1110 comparator remains 32/32 exact.

| Procedure | Observations per bank | Certified exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — |
| Smoothed-rate degree-20 LP | 100 | 0/32 | 21 | 11 | 0 |
| Smoothed-rate degree-20 LP | 300 | 0/32 | 16 | 16 | 0 |
| Smoothed-rate degree-20 LP, primary | 600 | 7/32 | 19 | 6 | 0 |
| Archived smoothed-rate degree-20 LP | 1110 | 32/32 | 0 | 0 | 0 |

The paired top-810 control recovers 0/32 exact graphs at each new prefix, with respectively 11,850, 2,824, and 395 missing true edge-bank pairs and the same number of extras. Its archived N=1110 result is 16/32 exact, with 23 missing and 23 extra edge-bank pairs.

Each prefix is the literal first N rows of each fixed H1ap stream, with 25 queried cells per row. Costs remain the exact half-even rounded Laplace rates `roundEven(1e6*(equalities+1)/(coqueries+2))`. All 3,240 candidate pairs remain eligible. Every cell is constrained to degree 20; this degree and the implied 810-edge cardinality are supplied structural priors. The top-810 control sorts the same costs, with lexicographic pair-index tie breaking. Geometry is used only to assess recovered graphs.

The primary was fixed at N=600; N=100 and N=300 are descriptive. These prefixes are nested, dependent observations from already-examined banks. They are not independent replication or fresh-source evidence. Failure establishes insufficiency of this particular procedure and budget on these streams, not impossibility of all lower-query recovery methods.

The feature producer loads only observed masks and values once per bank, reads rows only through each checkpoint, and saves independent count snapshots. New fits run in the frozen order 600, 300, 100, without warm starts or cross-prefix inputs. The archived N=1110 model is loaded only afterward. An independent pair-vector recount checks every produced count and rechecks archived counts against all 1,110 rows. For each of 96 bank-prefix pairs, replacing all future masks and labels leaves prefix counts and costs unchanged. Historical acquisition masks were finalized before their current labels, but their construction used calibration.

All **96 new exact optimality certificates** and **32 archived certificates** pass independent integer checks. The public verifier rejects 384 mutated certificates without invoking an LP solver. No primal values are snapped or rounded. A certified fractional returned optimum is counted as a failure of the frozen procedure; it does not establish that no integral co-optimum exists. Certificates establish optimality, not uniqueness. The primary success lower and upper bounds are both seven.

| Alternative historical prefix budget | Training queries per bank | Training queries, all 32 | Shared calibration, all 32 | Total acquisition, all 32 |
|---|---:|---:|---:|---:|
| 100 rows | 2,500 | 80,000 | 64,000 | 144,000 |
| 300 rows | 7,500 | 240,000 | 64,000 | 304,000 |
| 600 rows | 15,000 | 480,000 | 64,000 | 544,000 |
| 1110 rows, archived | 27,750 | 888,000 | 64,000 | 952,000 |

These are alternative nested budgets, not additive new spending. The recovery costs do not take noise probabilities as inputs, but acquisition is not calibration-free. This study uses zero new queries, source rows, latent draws, puzzle inputs, or GPU work.

Execution used **5.256441 aggregate CPU-seconds and 1.136500 wall seconds**, within frozen 120 CPU-second and 60-second limits, using eight one-thread workers with thread limits set before numerical imports. Timing includes preparation, 96 fits, independent counts, suffix gates, proof checks, archived comparator verification, and summary generation. Synthetic preflight and publication packaging are separate. Public proof verification checks exported counts and certificates; reproducing the raw count lineage and future-suffix gates requires the archived observations and independent bank auditor.

[Frozen protocol](protocol-prefix-rate.json) · [Independent audit](audit_prefix_rate.py) · [Public verifier](verify_prefix_rate_public.py) · [Results](../../../results/research/ripper/prefix_rate.json) · [Figure](../../../assets/figures/prefix-rate.png)
