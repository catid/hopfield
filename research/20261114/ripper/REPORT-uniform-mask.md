# H1ba: uniform acquisition fails the N900 reliability target

The frozen hypothesis is **falsified**: independent uniform 25-cell masks recover **23/32 certified integral exact graphs**, below the required 29. Eight returned optima are certified integral nonexact graphs and one is certified fractional. There are no unresolved runs. The paired archived adaptive acquisition recovers all 32 graphs with the same recovery costs and degree constraint.

| Procedure | Certified integral exact | Certified integral nonexact | Certified fractional | Unresolved |
|---|---:|---:|---:|---:|
| BP AdamW: outside this source-only study; not run | — | — | — | — |
| Archived adaptive masks + degree-20 LP | 32/32 | 0 | 0 | 0 |
| Uniform masks + degree-20 LP | **23/32** | **8** | **1** | **0** |

The uniform same-cost top-810 control recovers 0/32 graphs and has 257 missing true edge-bank pairs and 257 extras. Among the 31 integral uniform LP outputs, there are 19 missing and 19 extra edge-bank pairs; the fractional bank has no rounded graph and is not included in that edge total.

This is a paired acquisition intervention on already-examined H1az sources and corruption latents. It generates no fresh source identity or latent draw and carries no new distribution-generalization claim. The exact source rows and full corrupted board at each row are shared with the adaptive comparator. The recovery procedure remains fixed: all 3,240 pairs eligible, exact half-even rounded Laplace equality rates, and degree exactly 20 at every cell. Degree 20 and the implied 810-edge cardinality remain supplied structural priors.

Each bank uses an independently seeded RNG; every row takes the first 25 positions of a fresh permutation of all 81 cells. **All 900 masks are generated and saved before loading selected source labels or archived corruption arrays.** Masks use no calibration, likelihood-ratio state, previous labels, or geometry. The archived adaptive masks are loaded only after the uniform LP fit, for query accounting. This checks a genuinely calibration-free acquisition recipe, but its frozen recovery target fails. It does not establish that calibration is necessary for all successful acquisition rules: the intervention also removes the adaptive exposure allocation and exclusion feedback.

Common noise is reconstructed from the archived mask-independent uniform and position-permutation draws. Each row therefore exposes the same corrupted board in both arms. Independent replay regenerates all 28,800 uniform masks and all 28,800 archived latent draws, checks selected labels, recounts every pair, and requires every reused answer to equal the archived answer. All **32 new and 32 archived exact optimality certificates** pass; no archived LP is refitted. The public verifier checks masks, pair exposures, overlap accounting and proofs, and rejects 192 mutated certificates. Equality-label provenance and original adaptive masks require the separate raw archive audit.

No primal is snapped or rounded. The certified fractional output is a failure of the fixed procedure and does not prove that no integral co-optimum exists. Certificates establish optimality, not uniqueness. The primary success lower and upper bounds are both 23.

| Query accounting | Count |
|---|---:|
| Alternative uniform deployment: training | 720,000 |
| Alternative uniform deployment: calibration | 0 |
| Alternative adaptive deployment: training + calibration | 784,000 |
| Answers reused in the actual paired replay | 222,073 |
| **Newly exposed answers in the actual paired replay** | **497,927** |
| Actual union, including prior adaptive calibration | **1,281,927** |

Reused plus newly exposed answers equal the uniform arm's 720,000 training queries. Only the 497,927 newly exposed answers are incremental physical queries for this paired replay. The union is the previous 784,000 adaptive queries plus that increment. Alternative deployment budgets must not be added together, and the hypothetical zero-calibration budget is not the actual cumulative research spending.

Execution used **5.151850 aggregate CPU-seconds and 1.272411 wall seconds**, within the frozen 120 CPU-second / 60-second ceilings. Eight one-thread workers cover acquisition, recovery, independent masks/noise/counts, paired-query accounting, exact new and archived proofs, and summary. Synthetic preflight and packaging are separate. Observed arrays are materialized once per bank; no redundant calibration, archived refits, or filler GPU jobs were run. The source ledger remains **1,988,296 IDs**, with no new solver inputs or GPU work.

[Frozen protocol](protocol-uniform-mask.json) · [Mask generator](uniform_mask_features.py) · [Independent audit](audit_uniform_mask.py) · [Public verifier](verify_uniform_mask_public.py) · [Results](../../../results/research/ripper/uniform_mask.json) · [Figure](../../../assets/figures/uniform-mask.png)
