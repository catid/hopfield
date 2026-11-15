# H1af: descriptive exposure diagnostic

This descriptive analysis was added after protocol freeze and during acquisition, before inference. It does not change H1af endpoints, guide tuning, or establish a causal mechanism.

| Selector / bound | Retained nonedge-bank pairs | Never queried nonedge-bank pairs | Median exposures among retained nonedges | Mean exposures among retained nonedges |
|---|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — |
| exposure_exact | 191 | 0 | 272 | 271.46 |
| bounded_exact | 189 | 0 | 273 | 271.42 |
| exposure_learned | 744 | 0 | 258 | 258.44 |
| bounded_learned | 688 | 0 | 259 | 257.47 |

Counts pool 32 banks, each with 2,430 possible nonedges. They are pair-bank instances, not distinct pairs or independent test puzzles. Every arm has exactly 666,000 joint pair exposures per bank. Final counts include exposures before and after exclusion.

The distributions describe how much attention retained nonedges received under each selector. They can indicate neglected pairs but do not isolate a causal mediator of accuracy. True-edge and nonedge labels were used only in this postfit analysis, never by the acquisition policy.

[Frozen study](BOUNDED_EVIDENCE_SELECTOR.md) · [Final inference report](REPORT-bounded-selector.md) · [Diagnostic script](diagnose_bounded_selector.py) · [Diagnostic data](../../../results/research/ripper/bounded_selector_exposure_diagnostic.json)
