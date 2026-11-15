# H3cb proposal: spend unused membership budget on unresolved candidate sets

Status: design only, awaiting parent review. No implementation, source draws, frozen protocol or reserved seeds. H3ca artifacts and conclusions remain unchanged.

H3ca's common MAP/tree policy fails on sixteen of 1,024 pairs. Every failure stops with more than sixteen candidates after four paid MAP queries, leaving four calls of the declared eight-call allowance unused. Candidate-count exclusion avoids expensive exact optimization but does not establish that the true candidate cannot be identified on its realized path. This proposal tests whether a cheap, explicitly incomplete continuation can recover most of that loss at a small total query cost. It changes the terminal policy and the scientific objective; it is not a larger candidate-count setting or a relaxed exact-tree certificate.

## Falsifiable question

Across a fresh source bank for each of all 1,522 eligible supplied truths, can deterministic balanced-partition continuation recover at least 75% of the comparator's unresolved cases while increasing total actual membership calls by at most 3%?

For baseline failures F, newly recovered failures R, baseline calls B, and extended-policy calls E, the proposed single joint empirical endpoint is:

`F >= 8 and B > 0 and 4*R >= 3*F and 100*E <= 103*B`.

Fewer than eight comparator failures or zero comparator calls is prospectively inconclusive, not success or falsification. Otherwise a complete cohort failing either recovery or overhead condition falsifies the joint finite-cohort claim. All 1,522 truths and all their cases remain in cost and recovery totals. There is no selected-failure cohort, second source bank, redraw, threshold revision, pilot tuning or extension of the eight-call per-arm allowance.

The recovery and overhead thresholds are proposed after observing H3ca's failure mechanism. They apply only to new prospectively frozen evidence. H3ca's failures will not be executed through this fallback as a hidden pilot.

## Population and comparison

Use the supplied full class of 1,541 models, all 1,522 models whose N12 source population contains at least eight positives, and the complete 73,712-board N13 membership universe. Process each eligible truth exactly once in a fresh, uniformly randomized permutation. Each case receives one fresh bank of eight distinct positives sampled uniformly without replacement from that truth's complete source population. This balances truth coverage without replacement; it is a new cohort design and is not a same-seed replication of H3ca. The permutation seed and separate source seeds remain unspecified pending the then-current scoped seed audit. The truth schedule is oracle-only: no policy/selector receives the permutation, truth ID, remaining unused truth IDs or earlier cases' true labels. Policies start each case from its own source bank and fixed score tables; no across-case posterior learning is permitted. Cases are therefore not claimed to be independent truth draws, and the endpoint is solely the retained finite cohort.

| Method | Policy before unresolved stop | Additional policy | Measurement |
|---|---|---|---|
| BP AdamW | Not applicable | Not applicable | Unmeasured in this finite exact-search scope |
| B | H3ca T: hierarchical MAP, matched-posterior exact trees and guards | Stops at original unresolved outcome | Fresh comparator |
| E | Identical H3ca T policy | Up to four deterministic balanced queries after an unresolved four-MAP stop | Fresh intervention |

Both arms execute independently from the same source bank, empty history and separate selector/compiler/cache/proof namespaces. Both pay every actual membership label, even when board choices agree. Execute all B cases then all E cases. No label or proof cache is donated between arms. Shared source positives count once as preparation and twice as arm appearances.

## Deterministic fallback

Only after the unchanged H3ca T policy stops unresolved after four MAP calls, with more than sixteen candidates and no accepted handoff, E may continue. From current candidates C and unqueried boards q, compute positive and negative prediction counts. Choose the board minimizing `max(n_positive, n_negative)` among informative boards, with ascending board index as the sole tie rule. Labels filter the full current set exactly. Stop on a singleton, on no informative query, or after four additional calls. Never guess a model from an unresolved set. The selector does not use the hidden truth, orbit count, future labels, H3ca cases, or cohort frequencies.

This is a greedy policy, not an exact depth-four optimizer and not a claim of all-candidate recovery. If |C|>16, no depth-four binary tree can identify every member of C, by its sixteen-leaf limit. Realized singleton recovery can nevertheless occur for many truths, and the proposed 75% cohort condition remains uncertain. Preserve every remaining set and every unresolved outcome, including branches that still contain truth after all eight calls.

The decision rule uses candidate counts to isolate the mechanism of spending unused queries. It does not reuse the posterior as a newly tuned fallback parameter. All remaining true candidates stay in support under noiseless labels, so E cannot lose an exact result already produced by B when the shared prefix is implemented correctly.

## Prospective correctness invariants

The two arms must have identical original-policy actual queries/labels, candidate sets, MAP handoffs, exact-tree choices and original stopping outcome on every pair. Both have the same hierarchical selector and matched tree scores, deterministic ties and complete certificates. If B identifies a singleton, E must stop at exactly that same point with the same paid cost. E's additional labels can occur only after a qualifying unresolved B outcome and cannot increase a candidate set or remove the true candidate. No false terminal singleton is acceptable. An invariant violation invalidates the whole paired study and produces no empirical primary decision.

The full existing exact-tree and counterfactual auditors remain required for original-policy portions. A separate fallback auditor recomputes every board's two candidate counts from the supplied complete predictions and verifies the globally minimal worst branch and first-board tie. It reconstructs every label, candidate update, stopping reason, call count, casewise gain and joint endpoint without invoking the producer selector.

Fixed, nonrandom fixtures should cover: one unique balanced optimum; multiple tied optima; exclusion of previously queried boards; no-information stop; singleton stop; four-query exhaustion with unresolved candidates; a >16-candidate catalog with both rescued and unresolved truth paths; and a comparator exact case where no fallback is called. A truth-leakage mutation, wrong tie mutation, free cross-arm label mutation and premature singleton mutation must each be rejected by named independent checks.

## Bounded computation and efficiency

Propose retaining H3ca's 600 aggregate CPU-second / 900 wall-second budget, 2,000,000 memo states, 50,000,000 exact-search work steps and 100,000-bit integer limit. Raise only the study's natural declared paid-label ceiling to `2*1522*8 = 24,352`, with 12,176 per arm, because the cohort has 1,522 pairs rather than 1,024. These values are proposals for review, not authorization to launch.

Additionally charge fallback search explicitly: one board/candidate prediction examination per active candidate per considered board, with a prospective whole-study ceiling of 500,000,000 such examinations. This avoids presenting NumPy's vectorized scan as zero work or equating its units with the compiler's existing steps. Record fallback CPU/wall time, examined candidate-board cells, selection requests and maximum active-set size separately. Initial hashing/imports, preparation, runtime gates, selectors, proof serialization and monitoring remain inside the whole-study CPU/wall envelope.

The likely new bottleneck is repeated full-universe column counting. A vectorized Boolean gather/sum gives the same exact integer counts as a fixed synthetic scalar reference. Benchmark both on a public, fixed synthetic prediction table before freeze, preserving agreement and exact tie identity. Do not use fresh sources or H3ca's selected failures for timing/tuning. CPU scan work is small enough that GPU transfer/orchestration is not presumptively useful; measure eligibility before allocating a GPU. Retain existing proved exact-search optimizations.

Any CPU/wall/state/work/integer/fallback-scan cap is technical incompleteness of the whole study, not zero recovery, falsification, or permission to skip cases. Preserve paid labels, partial work and every failure. No changed-cap continuation, substitute cohort or inferred primary is allowed. Postrun audits/export are separately timed and may overlap.

## Required report

Report the one joint recovery/overhead endpoint, all 1,522 truth-specific results, every original and final unresolved set, all B/E calls and paid repeated boards, total and per-rescued-case added labels, source duplicates/historical matches, full exact-transition counts, and source seeds/provenance. Separate the logically guaranteed no-loss invariant from the measured 75% rescue claim. Include failures with up to eight paid calls and zero gain in the overhead denominator. Full source coverage is not evidence for arbitrary unseen rule families, noisy labels, or general positive-only learning.
