# H3y tracked evidence

The fixed all-18 transfer hypothesis is **falsified**: 12 unique exact transfers, four empty sources, one nine-way minimum ambiguity, and one unique minimum with an incorrect transferred graph. All cases completed and remain in the denominator. Source completeness, D4 symmetry and minimum-description priors are explicit; there is no target-dependent tie selection.

The unique wrong case is N10 queens plus camel and zebra. Its four complete source boards uniquely select camel alone. A separately frozen descriptive census proves that the selected N11 rule admits 56 complete boards, of which **12 are invalid under the true rule**. Every invalid witness and both full solution sets are preserved. No GPU search or optimizer fit was performed.

- [Full report](../../../../research/20261114/loki/h3y/REPORT.md) and [frozen structural protocol](../../../../research/20261114/loki/h3y/protocol.json).
- [Structural summary](summary.json) and [all 18 cases](per_case.csv).
- [Complete-source generation receipt](generation.json), [all source seals](source-seals.json), and the `sources/`, `cases/` and `cache/` exports contain complete source arrays, per-case certificates/witnesses, and independently validated cached universes. `summary.json` hashes every structural export.
- [Descriptive census protocol](../../../../research/20261114/loki/h3y/census-protocol.json), [census result](q11-census.json), and [all census boards](q11-census-boards.npz). The census preserves the structural-summary hash and leaves the original result unchanged.
- [Figure](multifamily-stress.png), [SVG](multifamily-stress.svg), and [deterministic plot provenance](plot-provenance.json).

Rebuild the figure with `artifacts/plotting-venv/bin/python research/20261114/loki/h3y/plot_public.py` from the repository root. It reads only tracked structural-summary and census exports.
