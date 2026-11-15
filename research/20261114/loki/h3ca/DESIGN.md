# H3ca: change tree weights while preserving hierarchical MAP guidance

Status: prospective design only. No implementation, frozen protocol, reserved seeds, fresh truth/source draws, or actual oracle calls. Parent review is required before implementation and a later source/gate review is required before freeze or draws. H3bz remains unchanged.

H3bz's joint hypothesis was falsified: fitting the uniform prior throughout the policy saved 350 of 3,121 calls (11.214354%) but lost one exact identification achieved by the hierarchical policy. Its five discordant cases all stopped after four MAP queries with more than sixteen candidates; none reached a terminal tree. This motivates isolating the tree objective while retaining the same MAP guidance and routing opportunity in both arms. This is a new intervention and fresh cohort, not a relaxation of H3bz's endpoint.

## Falsifiable question

On 1,024 fresh paired cases with a uniform truth prior over the 1,522 eligible models, does changing only the tree objective to the matched posterior reduce actual membership calls by at least 2%?

Let H be the original deeper policy using hierarchical MAP guidance and hierarchical tree weights. Let T retain that MAP guidance and use matched uniform-prior tree weights. The sole empirical primary condition is `H > 0 and 100*T <= 98*H`, using totals across all 1,024 pairs. A complete valid cohort that fails the inequality falsifies this finite-cohort hypothesis. Zero H is inconclusive. No pilot, redraw, extension, excluded failure, or alternative threshold is allowed.

The experiment also requires identical exact-identification flags for every pair, identical MAP queries and labels up to handoff, and identical handoff checkpoint/candidate set/requested depth (or identical no-handoff outcome). These are prospective correctness invariants justified in SAME_MAP_LEMMA.md, not empirical improvements allowed to trade against cost. A violation invalidates the implementation or proof conditions and leaves the whole study without a primary decision. It must be retained and investigated. In particular, a T-only or H-only exact outcome cannot be silently treated as evidence for this cost hypothesis.

## Population and intervention

The supplied class remains all 1,541 zero/one/two D4 leaper-orbit models, using the existing exact N12 source populations and the full 73,712-board N13 query universe. Truth is uniform over the 1,522 models whose source population M is at least eight. Each pair receives eight distinct N12 positives sampled uniformly without replacement from its truth's population. The shared source bank is paid/prepared once; each arm has 8,192 source-positive appearances, for 16,384 arm appearances. Inference filters the entire class only by compatibility with those positives and later noiseless labels. It never uses true orbit count or cohort frequencies.

For model c with k selected orbits and source population M, define hierarchical score `s_H(c)=C(55,k)*C(M,8)` and matched score `s_T(c)=C(M,8)`. Weights are exactly reciprocal scores, converted to positive integers by the existing LCM procedure. Both tables are undefined exactly on the nineteen models with M<8, which cannot remain compatible with eight distinct positives.

| Method | MAP selector score | Tree and guard weights | Status |
| --- | --- | --- | --- |
| BP AdamW | Not applicable | Not applicable | Unmeasured; outside this finite exact-search experiment |
| H | s_H | reciprocal s_H | Fresh measured comparator |
| T | s_H | reciprocal s_T | Fresh measured intervention |

The selector file must explicitly contain s_H in both namespaces. The compiler receives a distinct tree-score file containing s_H or s_T. File names and every receipt/cache identity must distinguish selector scores from tree scores; a generic `scores` field must not hide different meanings. Counterfactual MAP partitions also use s_H in both arms. Counterfactual terminal depth-four trees and each arm's cost guard use that arm's tree weights.

## Unchanged policy and accounting

At m paid MAP calls, a singleton stops. A set larger than sixteen is ineligible for tree compilation; this is not an infeasibility certificate. Otherwise the arm attempts an exact tree of requested depth 8-m. For a feasible tree, it computes both branches of the fixed continuation consisting of 4-m hierarchical MAP calls and then an exact depth-four terminal tree. Handoff requires every candidate's continuation to identify exactly and the exact integer cost guard to pass. Equality accepts. The guard must pass whenever that continuation is complete; at m=4 it must tie. After four MAP calls there is no further MAP fallback. Each case/arm uses at most eight actual queries.

All H cases execute before all T cases. Each arm has a separate selector subprocess, compiler object, counterfactual object, proof directory, score bindings and caches. Both start each case with source-only knowledge. Even identical selected boards are independently queried and paid. Cross-arm history annotates reacquisitions only after selection and never supplies a free label or changes a choice. No cross-arm proof/cache reuse is permitted in this intervention.

Every source draw, query-before-answer record, label, duplicate bank, historical source match, failed outcome and technical failure is retained in the durable hash-chained ledger. Historical-bank checking must include H3bz as well as all previously declared bank histories. No fresh source is inspected before the prospective protocol and gates are approved.

## Mathematical claim and its resource limit

On completed exact calculations, candidate-count eligibility, tree feasibility and the all-candidate continuation-admissibility predicate are independent of strictly positive tree weights. Because an admissible continuation is itself an allowed tree of depth at most 8-m, each weighted optimum passes its own guard. Identical MAP guidance therefore gives the same actual MAP prefix and handoff. Both accepted trees identify every remaining model, so paired exact flags agree.

Conditioned on a common source bank and common MAP history, the true posterior is proportional to reciprocal C(M,8). Thus the T tree has no larger conditional expected remaining query count than H's tree at that handoff. This expectation result does not imply a realized per-case improvement, a realized 2% cohort saving, a population 2% improvement, or a runtime improvement. Individual T paths may be longer. All such outcomes remain in the cost total.

Weight-dependent Huffman bounds, ordering, pruning, memoization and integer sizes can change computation substantially. Equality of the scientific routing decisions is not equality of solver traces or time-to-solution. In particular, one arm can exhaust the budget where the other finishes. No capped calculation is an infeasibility certificate. Any CPU/wall/state/work/integer cap or missing exact certificate makes the whole paired study incomplete, even if one arm or many cases have completed. Never substitute an incumbent, skip the case, resume under higher caps, or continue MAP because a compiler capped.

## Prospective budget and gates

Use the same 600 aggregate CPU-second / 900 wall-second envelope, 2,000,000 memo-state cap, 50,000,000 work-step cap, and 100,000-bit integer cap as H3bz. Charge both arms, runtime fixtures, preparation/imports/hash checks/serialization, counterfactuals and selector descendants under one monitor. Actual calls are capped at 16,384 total and 8,192 per arm. Postrun independent audit/export is separately timed. Any limit leaves all artifacts and statuses intact and yields no primary decision.

Before a source packet can be approved, gates must establish:

1. All source populations, full candidate support, all 1,522 uniform truth tickets, both score formulae and each role-specific score binding are reconstructed independently without RNG draws.
2. The same hierarchical selector is used in actual MAP and counterfactual MAP calls in both arms. Mutations substituting matched scores into either T selector path fail at a named check.
3. Fixed synthetic catalogs exercise feasible/infeasible trees, all-branch admissibility rejection, candidate-count ineligibility, no-information stopping, initial singleton, m=4 guard ties, unequal tree paths, and equal MAP/handoff/exact outcomes across positive weight choices. Independent exhaustive reference trees verify feasibility and weighted optimality.
4. Weight-sensitive pruning gates preserve exact optima and lexicographic ties under both score tables. The inherited full proof replay verifies every examined or pruned option, inline Huffman trace, cache identity and reused complete tree proof.
5. Forced state/work/CPU/wall/integer caps are observable technical incompleteness, including a fixture where one weight ordering completes and another caps. No false infeasibility, incumbent routing, fallback, paid call after failure, or partial primary decision is accepted.
6. A separate fixed paired namespace fixture pays repeated cross-arm boards independently, isolates caches, and verifies the common-prefix/handoff invariants from full ledger replay. An independent auditor reconstructs both score roles and rejects swapped files or missing role identities.
7. The independent postrun auditor checks all pairs' prefix/handoff/exact invariants and, at every handoff, evaluates both chosen trees under exact matched-posterior weights to confirm the conditional expectation inequality. These mathematical checks do not replace full optimality certificates.

Truth/source seed ranges remain deliberately unspecified until a new seed audit. That audit must inspect the then-current full registry, explicit integer seeds, all declared Loki formula ranges, and nearby Ripper solver formulas. Document precisely which generators and formulas were checked; numerical seed uniqueness is not a claim of PRNG independence. No proposed range is reserved by this design.

## Useful performance work and reporting

Retain the already verified inline Huffman bounds, depth normalization, complete-proof cache reuse within each arm, compact RAM routing cache, and deterministic parallel postprocessing export. Do not change the scientific planner during this intervention. The known expensive work is CPU exact-tree/counterfactual search, serialization, and full-proof audit; GPUs have no eligible role. Measure per-arm actual states, partitions, Huffman merges, CPU/wall time, proof bytes and audit/export time. Compare these measured quantities without assuming similar routes imply similar work. Any additional optimization needs a separate source change, correctness comparison and measured fixed-fixture timing before freeze.

Report the sole finite-cohort cost endpoint, all exact outcomes, all remaining candidate sets, duplicates/historical matches, paid reacquisitions, realized per-case gains/losses/ties, common handoff distribution, and exact conditional expected costs. Orbit and handoff strata are descriptive only. BP AdamW is first and unmeasured in method tables. This remains a supplied finite-class/noiseless-oracle experiment, not a general positive-only rule learner or optimizer comparison.
