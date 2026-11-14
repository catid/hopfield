# H3f: training-count and complete-bank controls

**Frozen exclusion-energy noninferiority hypothesis: supported.** All 14,200 complete n=12 solution patterns already occur among the positive-weight fitting labels. The 8,192 evaluation partial inputs are held out, but their complete solution patterns are not. The compatible-bank control makes this support overlap explicit.

| Method | Optimizer updates | Original N20 (%) | Certificate-first N20 (%) | Coverage (%) | Wrong stops | Capped chains | Mean search moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP AdamW, decay0.01 | 4,000 | 99.788411 | 99.788411 | 54.172327 | 0 | 212,163 | 13320.23 |
| BP Adam, no decay | 250 | 100.000000 | 100.000000 | 71.566672 | 0 | 19,955 | 4878.40 |
| BP Adam, no decay | 1,000 | 100.000000 | 100.000000 | 72.323870 | 0 | 15,655 | 4508.89 |
| BP Adam, no decay | 4,000 | 100.000000 | 100.000000 | 71.632876 | 0 | 32,272 | 6000.44 |
| SGD + geometry preconditioner | 4,000 | 99.926758 | 99.926758 | 72.493537 | 0 | 14,569 | 4322.40 |
| Training-count exclusion + fixed energy | 0 | 100.000000 | 100.000000 | 75.952803 | 0 | 1,729 | 1947.55 |
| Compatible training-bank retrieval | 0 | 100.000000 | 100.000000 | 76.381270 | — | — | — |

Exclusion versus Adam250: accuracy difference +0.000000 percentage points, one-sided 95% endpoints [0.000000, 0.000000]; coverage difference +4.386131 points, endpoints [4.249731, 4.520206]. Exact upper probability for any paired input regression: 0.036562% (0 observed). Frozen margins: 0.1 point accuracy, 2 points coverage.

The constructor counts offset co-occurrences using only complete fitting labels and their exact positive expanded-row multiplicities. It infers q=12 from their common cardinality. Zero observed count marks an offset forbidden; coefficients are fixed as J=q×forbidden−1 and unary0. Rewarding every allowed pair by−1 and penalizing every forbidden pair byq−1 is an added fixed energy-design prior, not learned-weight recovery. The offset architecture supplies geometry. No attack category or held-out label chooses coefficients.

The separate post hoc rule audit finds 44 inferred forbidden offsets and 220 allowed offsets, with exact attack-support recovery. All training solutions have energy−66 and single-cell margin at least11. Temperature unit 25.741927615, threshold and margin are training-derived. The saved-output audit explicitly counts invalid boards below, equal to and above q, separately for stopped and capped chains; dense invalid occupancies are not silently ignored.

The bank stores all14,200 complete training boards in 255,600 packed bytes. For each partial input it filters compatible stored boards and samples20 times with replacement. It does not consult puzzle rules or a held-out solution list. Every sampling identity was independently replayed. Its CPU timing is retained separately from GPU search timing; no cross-hardware speedup is claimed. Bank output storage flags are not learned stopping certificates.

All 3,440,640 saved boards, learned comparisons, selection identities, support counts and bank draws passed the independent audit. Original learned-model searches were reused only with pinned hashes and unchanged full settings. This is adaptive reuse of the H3e evaluation split, not new confirmation or solution-pattern generalization.

[Conditional stopped-state proof](CERTIFICATE_PROOF.md), [Frozen protocol](protocol.json), [constructor and workers](driver.py), [auditor](analyze.py), [tracked summary](../../../../results/research/loki/h3f/summary.json), [per-seed counts](../../../../results/research/loki/h3f/per_seed.csv).
