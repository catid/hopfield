# Conditions for common MAP routing under different tree objectives

This is a design proof for H3ca, not a claim that its future implementation has passed validation. It distinguishes the exact mathematical policy from a budgeted implementation.

## Assumptions

Both arms share the same nonempty source-compatible set C, full deterministic prediction catalog, paid noiseless truth labels, MAP score table s_H, candidate-ID and board-ID tie rules, queried-board exclusion rule, maximum four MAP calls, sixteen-candidate compilation eligibility limit and total depth allowance eight. Only the strictly positive tree weights differ. The true candidate remains in C.

Every tree request that affects a routing decision must return a complete, correct optimality or infeasibility certificate for the same allowed nonconstant binary-query tree class and requested depth. The implementation serializes and hash-binds the complete producer certificate before routing. Full independent certificate validation is retrospective, after the study; this lemma applies only to runs whose certificates pass that validation, and does not claim an independent auditor executes inside the query loop. Terminal continuation requests use depth four and the same class. Counterfactual evaluation retains all candidate branches and uses s_H for every MAP partition. It declares admissibility exactly when every candidate is identified, not from a weighted average or a mass cutoff. Numeric comparisons use exact integer arithmetic. No technical cap is mapped to infeasibility or to a MAP continuation.

## 1. Feasibility does not depend on positive weights

For a candidate set S and depth d, the admissible tree set depends only on predictions, S and d. The weights define the objective of choosing from this set. Hence completed exact optimizers with different positive weights agree whether the set is empty. They may choose different trees and different per-candidate depths.

The implementation's weighted branch-and-bound can certify this property only if its bounds are valid and its run completes. A Huffman lower bound is a lower bound on each allowed tree's positive weighted external path length. It never prunes before a feasible incumbent exists. Once an incumbent exists, a cost-dominated partition cannot improve the required optimum; pruning it cannot erase the fact that a feasible tree already exists. For an infeasible state no incumbent ever exists, so every nontrivial partition must be ruled out by a weight-independent information bound or certified infeasible child. Sorting by weighted bounds changes traversal and work, not the certified feasible/infeasible answer. Equal-cost pruning requires the retained lexicographic condition. These are obligations for the independent certificate auditor, not assumptions justified merely by executing the same code twice.

## 2. Continuation admissibility does not depend on tree weights

Fix S, queried history Q, and r=4-m remaining MAP calls. If S is a singleton, both continuations identify it. If r=0, each continuation asks whether a depth-four separating tree exists. Part 1 gives identical exact flags for every candidate at that leaf. If r>0, common s_H and deterministic ties choose the same informative MAP board and partition S identically. Apply induction to both nonempty label branches. Thus both continuations identify the same candidates, and their all-candidate admissibility predicates agree. Their costs can differ because their terminal optimal trees can differ.

A feasible ancestor implies that every distinct candidate remains distinguishable; consequently any nontrivial subset admits an informative board. Previously queried boards are constant over the current set and cannot be selected as a nonconstant tree partition or informative MAP query. The continuation therefore does not consume a free repeated label. Every counterfactual branch is a model calculation, not an actual oracle call.

## 3. An admissible continuation always passes its own cost guard

An admissible continuation is a complete separating policy with at most r MAP calls followed by at most four terminal-tree calls. It is an allowed binary decision tree of depth at most r+4=8-m, after deleting any constant queries. The new exact optimal tree under that arm's current positive weights can cost no more. Restricting static reciprocal-score weights to a noiselessly observed subset changes only their normalization, not within-subset preferences, so terminal optimization is consistent with the root objective.

Therefore `tree feasible AND continuation admissible` is sufficient for guard acceptance. A pure cost rejection when admissible is a correctness error. At m=4 both compared trees are optimal at depth four under identical within-arm weights; their exact costs must tie. The cost values need not match across arms.

## 4. Equal prefixes, handoffs and exact flags on completed runs

Proceed by induction over actual MAP steps. Before any query the paired source sets agree. At each shared prefix, candidate-count eligibility agrees. If eligible, feasibility agrees by Part 1 and continuation admissibility agrees by Part 2. By Part 3 the two arms either both hand off or both reject. On rejection they choose the same next MAP query and receive the same noiseless truth label, so the next candidate sets agree.

At a handoff the checkpoint m, candidate set S, query history Q and requested depth 8-m agree. The actual trees may differ, but both identify every candidate in S; in particular both identify the true candidate. If no handoff occurs, both follow the same MAP prefix to the same stopping set. An initial/early singleton is shared; a no-information stop is shared; a four-MAP unresolved set is shared. Thus every paired exact flag agrees. Complete routing logs can differ in weighted costs, proof identities, explored states and chosen tree paths without violating this lemma.

## 5. Matched-posterior expected query count

The actual truth prior is uniform over eligible models. An unordered set of eight distinct source positives has likelihood 1/C(M,8) under any compatible model with source population M. An ordered sample contributes the same model-independent factor 1/8!, so the posterior ratios are unchanged. Subsequent noiseless membership observations restrict support. Deterministic common MAP choices add no truth information beyond their observed labels. Therefore at any shared handoff, the actual conditional posterior is exactly proportional to reciprocal C(M,8) on S.

Let d_H(c) and d_T(c) be the two tree depths for c at that handoff and let w_T(c)>0 be exact matched weights. H's complete tree belongs to T's allowed tree class, so optimality gives

`sum_c w_T(c)*d_T(c) <= sum_c w_T(c)*d_H(c)`.

The paid MAP prefix is equal. Conditional averaging and then averaging over common source/history paths gives nonincrease in expected scientific query count for the fully evaluated mathematical policy. It gives neither a 2% margin nor pointwise path dominance. The finite-cohort 2% endpoint remains falsifiable. The result is about the uncensored mathematical policy; do not infer an unconditional performance claim for a cap-limited implementation or for a selected subset of completed cases.

## 6. Resource limits break algorithmic equivalence

Different positive weights can change all of: LCM size, Huffman merge values, option order, incumbent arrival, pruning, visited states, memo/cache population, proof size, serialization time and aggregate CPU/wall cost. A shared global budget additionally makes remaining resources depend on execution order. Nothing above proves that both arms finish under the same finite caps, or that failure occurs at the same checkpoint.

A resource exception is thus a third state, distinct from certified feasible and certified infeasible. It terminates the whole study as technically incomplete. A found feasible incumbent without a complete proof is insufficient for any routing decision. Partial completed cases, the already completed H arm, and paid labels are retained, but no paired primary decision is emitted. No retry or increased-cap continuation uses that cohort. This avoids turning computational censoring into a spurious scientific divergence or excluding expensive outcomes.

The prospective gate must demonstrate this distinction with a deterministic catalog whose weight-sensitive traversal makes one run finish and another hit a chosen fixed cap. The study monitor and auditor must reject converting that cap into false feasibility, false infeasibility, a fallback MAP call, or a cost endpoint. Exact cross-arm invariants are evaluated only when both sides have the complete certificates required by the policy; global completion is still necessary for any primary decision.
