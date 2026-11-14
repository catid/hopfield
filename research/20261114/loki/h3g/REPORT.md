# H3g: held-out complete patterns and bank-incompatible inputs

**Frozen 512-pattern primary hypothesis: supported.** Every valid completion of every evaluation input lies outside all 512 patterns in the largest training bank. No evaluation input overlaps any restricted-bank training input or the previous H3e evaluation inputs. A compatible training-bank baseline therefore has 0% accuracy and 0% coverage by construction, independently verified by matching every input.

| Training patterns | Method | Updates | Original N20 (%) | Certificate-first N20 (%) | Coverage (%) | Wrong stops | Capped chains |
|---|---|---:|---:|---:|---:|---:|---:|
| 512 | BP AdamW, decay0.01 | 4,000 | 99.507650 | 99.532064 | 53.103257 | 0 | 222,591 |
| 512 | BP Adam, no decay | 250 | 99.991862 | 99.991862 | 67.929639 | 0 | 29,608 |
| 512 | SGD + geometry preconditioner | 4,000 | 99.995931 | 99.995931 | 69.825403 | 0 | 25,482 |
| 512 | Training-count exclusion + fixed energy | 0 | 100.000000 | 100.000000 | 76.864761 | 0 | 3,469 |
| 32 | BP Adam, no decay | 250 | 69.335938 | 80.098470 | 15.793356 | 3,620 | 293,662 |
| 32 | SGD + geometry preconditioner | 4,000 | 73.657227 | 84.077962 | 18.279486 | 3,025 | 267,512 |
| 32 | Training-count exclusion + fixed energy | 0 | 97.387695 | 97.387695 | 57.866327 | 0 | 18,152 |
| 128 | BP Adam, no decay | 250 | 99.743652 | 99.833171 | 58.109458 | 0 | 36,942 |
| 128 | SGD + geometry preconditioner | 4,000 | 99.776204 | 99.881999 | 61.950264 | 0 | 35,802 |
| 128 | Training-count exclusion + fixed energy | 0 | 99.890137 | 99.890137 | 75.144573 | 0 | 4,081 |

Primary count-energy versus Adam250 difference: accuracy +0.008138 percentage points (one-sided 95% endpoints [0.000000, 0.020345]); coverage +8.935122 points ([8.783620, 9.090677]). Exact upper probability for any paired input regression: 0.036562%. Frozen margins were 0.1 point accuracy and 2 points coverage. Search seeds remain clustered inside each resampled input.

The conditioned test distribution contains 8,192 unique inputs: 0 with one clue, 16 with two clues and 8,176 with three clues. It is sampled from 111,781 inputs incompatible with the largest training bank and absent from prior H3e evaluation. It is deliberately different from the original GRAM distribution. The nested 32/128/512 banks are prefixes of one fixed random permutation of complete patterns, not independent training replications.

Each selected training pattern contributes all 298 one/two/three-clue masks. The banks give 9,536/38,144/152,576 expanded labeled rows and 8,923/31,109/90,239 unique training inputs. Outside patterns never enter fitting, support counts or calibration. All learned models retain the n=10 learning rates and 4,000-step cosine horizon; Adam stops at the fixed 250-update prefix. Count energies retain J=q×forbidden−1, unary0, with q inferred from training cardinality. Allowed-pair reward and offset geometry remain explicit fixed priors.

Independent checks covered all 4,915,200 saved boards, preserved givens, selection identities, exact completion counts and model calibration. Every returned valid board was checked to be absent from the entire 512-pattern bank. Stopped/capped invalid boards are separated into below/equal/above 12 queens, with float64 certificate-margin diagnostics. The lower 32/128-bank results and their comparisons are secondary; no setting was selected from them.

This study demonstrates only the measured generalization relative to these new restricted banks and this conditioned split. Earlier studies used the full solution universe; no claim is made that these mathematical patterns had never been considered anywhere in the research process. The dataset and all recipes were frozen before the new fits or outcomes.

[Frozen protocol](protocol.json), [split builder](prepare.py), [runner](driver.py), [auditor](analyze.py), [tracked summary](../../../../results/research/loki/h3g/summary.json), [per-seed counts](../../../../results/research/loki/h3g/per_seed.csv), [per-clue counts](../../../../results/research/loki/h3g/per_clue_seed.csv).
