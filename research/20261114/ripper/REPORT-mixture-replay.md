# H1at: fixed likelihood mixture on existing adaptive observations

The preregistered conjunction is **falsified**. The mixture preserves every true edge in all 32 banks but retains 4,064 extras, exceeding the ceiling of 1,941. Original q1 = 1/9 retains 2,588 extras: the mixture has **57.03% more** on these fixed observations.

| Evidence rule | All true edges retained | Exact graphs | Missing true edges | Retained extras |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Original q1 = 1/9 | 32/32 | 0/32 | 0 | 2,588 |
| Uniform four-alternative mixture | 32/32 | 0/32 | 0 | 4,064 |

The mixture assigns weight 1/4 to q1 ∈ {1/36, 1/18, 1/9, 2/9}, with supplied q0 = 9/1600 and boundary 648000. For each pair with m coqueries and c equal observed labels, its evidence is the arithmetic mean of the four likelihood ratios:

\[
E(m,c)=\frac14\sum_{q\in\{1/36,1/18,1/9,2/9\}}
\left(\frac{q}{9/1600}\right)^c
\left(\frac{1-q}{1591/1600}\right)^{m-c}.
\]

All components continue updating without individual stopping. A pair is removed permanently at the first row with E ≥ 648000, including exact equality. Unexposed pairs remain retained. There is no maximum over components, threshold tuning, mask reselection, or replacement by a final-only decision.

The original H1as masks and labels are reused in their original order. These masks were fixed before each current noise draw; using the existing predictable masks remains valid even though the original acquisition algorithm used q1 = 1/9. Under the supplied independent board-permutation law, each queried true pair has equality probability 9/1600. Each unstopped likelihood process has conditional mean multiplier one, and their fixed mean is a nonnegative martingale starting at one. Ville's inequality gives a crossing probability at most 1/648000 per true pair under this law. This is not a guarantee for arbitrary correlated noise.

The finite-bank target was zero missing true edge-bank pairs **and** at least a 25% reduction from 2,588 extras (integer ceiling 1,941). Both arms retain all true edges; neither is exact in any bank. Descriptively, the mixture removes no pair that the original method retained and retains 1,476 additional pairs. It has more extras in every bank, with increases of 36–64. This fixed mixture does not improve evidence efficiency on these observations. The result does not test every possible mixture or an acquisition policy driven by mixture evidence.

All 32 banks are included. Exact integer likelihood comparisons certify each threshold and predecessor for both arms at every exposure count 0–1110. An independent 80-digit Decimal logsumexp audit checks 4,435 attainable endpoints/predecessors; the smallest log-distance to the boundary is 0.00050778. Independent reconstruction checks 115,084,800 cumulative pair states, replays all 103,680 original first-crossing paths exactly, and checks every mixture crossing. The original q1 trajectory also matches the archived cumulative log-likelihoods. The public verifier independently checks exported support statistics and accounting; full likelihood verification requires the preserved raw-observation replay.

The observations represent 27,750 historical training queries per bank, or 888,000 in total. This study adds **zero** queries, calibration fits, source rows, solver inputs, and GPU work. Execution on Ripper used **13.513215 aggregate CPU-seconds and 2.361958 wall seconds**, within the frozen 120 CPU-second / 60-second ceilings, using up to eight single-thread workers. These execution costs cover preparation, threshold certification, all 32 producer/auditor jobs, and aggregation. Synthetic-only development preflight and publication packaging are separate.

The source banks and observations were already examined in H1as. This is a post-hoc mechanism experiment with a frozen finite-bank target, not fresh-source confirmation or a population-confidence claim.

[Frozen protocol](protocol-mixture-replay.json) · [Independent raw auditor](audit_mixture_replay.py) · [Public verifier](verify_mixture_replay_public.py) · [Results](../../../results/research/ripper/mixture_replay.json) · [Figure](../../../assets/figures/mixture-replay.png)
