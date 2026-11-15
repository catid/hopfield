# H1ay: 900 observations recover all 32 fixed graphs

The frozen N=900 hypothesis is **supported**: the smoothed-rate degree-20 LP recovers **32/32 exact graphs**, each with an independently verified integer primal-dual certificate. N=1000 also recovers 32/32. N=750 yields 25 certified integral exact graphs, six certified integral nonexact graphs, and one certified fractional optimum. No run is unresolved.

| Procedure | Observations per bank | Certified exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — | — |
| Archived H1ax degree-20 LP | 600 | 7/32 | 19 | 6 | 0 |
| Degree-20 LP | 750 | 25/32 | 6 | 1 | 0 |
| Degree-20 LP, primary | 900 | 32/32 | 0 | 0 | 0 |
| Degree-20 LP | 1000 | 32/32 | 0 | 0 | 0 |
| Archived H1aw degree-20 LP | 1110 | 32/32 | 0 | 0 | 0 |

This is an **adaptive follow-up on the same 32 streams** after H1ax failed at N=600. The new checkpoints 750, 900, and 1000, primary N=900 threshold of at least 29 certified integral exact graphs, solver recipe, controls, and compute ceilings were frozen before any new checkpoint counts or outcomes were inspected. The sources have already informed the research program; this is not fresh-source confirmation or a population sample-complexity bound. It identifies an observed transition among these prespecified checkpoints, not a minimum sufficient query count.

The same-rate top-810 control recovers 0/32, 3/32, and 11/32 exact graphs at N=750, 900, and 1000. It has respectively 169, 78, and 38 missing true edge-bank pairs and the same number of extras. The archived controls are 0/32 at N=600 and 16/32 at N=1110. Thus supplied degree structure still changes recovery materially at N=900 despite identical pair costs.

Costs are unchanged exact half-even rounded Laplace rates `roundEven(1e6*(equalities+1)/(coqueries+2))`; all 3,240 pairs are eligible. The LP enforces degree exactly 20 at every cell. The top-810 control has the same costs and lexicographic pair-index ties. Degree 20 and its implied cardinality remain supplied priors; row, column, and box identities enter only evaluation.

Raw observed masks and labels are loaded once per bank. Count snapshots contain only rows before their checkpoint; fits run in order 1000, 900, 750 without warm starts or cross-prefix inputs. Archived N=600 and N=1110 models are loaded afterward and independently reverified without new solves. Independent pair-vector recounts match all counts. All **96 future-suffix substitution gates**, **96 new optimality certificates**, and **64 archived certificates** pass. The public verifier checks the proofs without an LP solver and rejects 480 altered certificates.

A certified fractional returned optimum counts as a failure of the fixed procedure; it does not rule out integral co-optima. No primal rounding is permitted. Certificates prove optimality, not uniqueness. The N=900 primary success bounds are both 32.

| Alternative historical budget | Training queries per bank | Training queries, all 32 | Shared calibration, all 32 | Total acquisition, all 32 |
|---|---:|---:|---:|---:|
| N600, archived | 15,000 | 480,000 | 64,000 | 544,000 |
| N750 | 18,750 | 600,000 | 64,000 | 664,000 |
| N900, primary | 22,500 | 720,000 | 64,000 | 784,000 |
| N1000 | 25,000 | 800,000 | 64,000 | 864,000 |
| N1110, archived | 27,750 | 888,000 | 64,000 | 952,000 |

These budgets are nested alternatives and must not be added together. N=900 uses 18.9% fewer training queries than N=1110, or 17.6% fewer total historical acquisition queries when the same calibration overhead is included. Recovery costs need no supplied noise probabilities, but historical mask acquisition used calibration; this is not end-to-end calibration-free acquisition. This study consumes zero new queries, source rows, latent draws, solver inputs, or GPU work.

Execution used **6.482687 aggregate CPU-seconds and 1.358693 wall seconds**, within 120 CPU-second and 60-second ceilings, with eight one-thread workers and thread limits set before numerical imports. This includes preparation, 96 new fits, independent recounts, suffix gates, new and archived proof checks, and summary generation. Synthetic gates and packaging are separate. Public verification checks exported counts and proofs; raw count lineage and suffix isolation require the archived observation audit.

[Frozen protocol](protocol-intermediate-rate.json) · [Independent audit](audit_intermediate_rate.py) · [Public verifier](verify_intermediate_rate_public.py) · [Results](../../../results/research/ripper/intermediate_rate.json) · [Figure](../../../assets/figures/intermediate-rate.png)
