# H1ae: descriptive exposure diagnostic

This analysis was added after acquisition and while inference was finishing. It is descriptive and was not used to tune or change H1ae. It does not establish a causal mechanism.

| Selector / bound | Retained nonedge-bank pairs | Never queried nonedge-bank pairs | Median exposures among retained nonedges | Mean exposures among retained nonedges |
|---|---:|---:|---:|---:|
| BP AdamW: outside this study; not run | — | — | — | — |
| exposure_exact | 217 | 0 | 271 | 271.28 |
| lr_exact | 4,230 | 552 | 24 | 40.92 |
| exposure_learned | 750 | 0 | 258 | 257.87 |
| lr_learned | 10,128 | 3,386 | 2 | 25.79 |

Counts pool 32 banks, each with 2,430 possible nonedges. They are pair-bank instances, not distinct pairs or independent test puzzles. Every arm has exactly 666,000 joint pair exposures per bank. Final counts include exposures before and after exclusion.

The LR selector leaves many nonedges with few or no observations. This is consistent with allocation concentrating on pairs that already accumulated positive evidence. The exposure control has no unqueried pairs and its residual nonedges have substantially more observations. True-edge and nonedge labels were used only in this postfit analysis, never by the acquisition policy.

[Frozen study](LR_EVIDENCE_SELECTOR.md) · [Final inference report](REPORT-lr-selector.md) · [Diagnostic script](diagnose_lr_selector.py) · [Diagnostic data](../../../results/research/ripper/lr_selector_exposure_diagnostic.json)
