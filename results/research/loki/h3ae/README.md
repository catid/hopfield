# H3ae: model-only exact zero-F compilation

Validated exports for [the full report](../../../../research/20261114/loki/h3ae/REPORT.md). The empirical endpoint is at least10× measured compile/load/three-query speed relative to three frozen GPU permutation calls, including independent completeness verification. Quality is a correctness/completeness gate. Capped distinct-solution coverage divides distinct valid returns by sum min(compatible true solutions,20), averaged over three seeds; 100% does not mean unrestricted enumeration of every compatible completion per query.

The summary retains all costs, scope, model/cache identities, audits and decisions. Family folders contain the sealed complete cache, model, compilation receipt and fresh-cohort metadata. Per-seed CSVs separate GPU controls and exact queries. Fixed N13, two fixed families, three batches of4096queries; no equal-FLOP, single-query or scaling claim.
