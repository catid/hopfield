# H3: AdamW versus geometry-derived preconditioning

**Preregistered n=10 result: falsified.** Selection on 512 discovery inputs chose `adamw-lr0.3`; confirmation used 3,179 different unique partial boards, each with 20 chains under each of 3 paired search seeds. Models use the same 97,554 fitting rows, 4,000 full-batch updates, zero initialization and exact pseudolikelihood. No test data were used.

BP AdamW here differentiates the local conditional loss with respect to energy parameters. It does not backpropagate through search. The SGD controls use the frozen paper LR 30 with geometry preconditioning on/off; AdamW uses the selected LR and weight decay 0.01. All arms use the same normalized temperature and stopping policy, with statistics fitted from their own training rows.

| Size | Method | N20 accuracy (%) | Coverage (%) | Wrong stopped chains | Capped chains | Complete fit (s) |
|---|---|---:|---:|---:|---:|---:|
| 10 | BP AdamW, no preconditioner | 99.538639 | 92.733910 | 0 | 11,522 | 8.014 |
| 10 | SGD + preconditioner | 100.000000 | 94.086455 | 0 | 2,921 | 8.095 |
| 10 | SGD, no preconditioner | 95.092796 | 88.895293 | 57,698 | 772 | 8.073 |
| 8 | BP AdamW, no preconditioner | 100.000000 | 97.629010 | 0 | 11 | 3.205 |
| 8 | SGD + preconditioner | 100.000000 | 97.559275 | 0 | 8 | 2.423 |
| 8 | SGD, no preconditioner | 99.032258 | 96.792190 | 1,005 | 3 | 1.826 |

Against preconditioned SGD at n=10, AdamW changes accuracy by **-0.461361 percentage points** and coverage by **-1.352546 points**. One-sided 95% bootstrap lower/upper bounds are [-0.660585, -0.272622] for accuracy and [-1.616492, -1.069151] for coverage. These endpoints form a 90% two-sided interval; they are the preregistered one-sided tests.

The accuracy safeguard found 15 unique inputs with at least one paired-seed regression. Its exact one-sided 95% upper probability is 0.725628%, giving the conservative accuracy-difference lower bound -0.725628 points. Support required this bound above −0.1 point and the coverage lower bound above −2 points. Absence of evidence is reported as inconclusive.

Coverage counts distinct valid completions, capped at min(all completions, 20), and pools unique inputs by that denominator. N20 keeps the lowest-energy returned chain, including a capped chain’s best board under the unchanged paper convention. Search-seed repeats are clustered inside each unique input for uncertainty. The n8 result is a secondary transfer check with only 310 confirmation inputs, not a new LR selection.

Fit times are one observation per deterministic model on Loki Blackwell, with CUDA context already initialized, and include progress logging. They are not warmed benchmarks or H200 comparisons. The seven-arm discovery sweep took separate fits; selection overhead is excluded from the final model fit column.

The official published SGD choice was previously selected using the original validation set; it is a frozen external control here. The new AdamW choice used only this campaign’s discovery subset. The training and validation sets are disjoint by partial input, but completed solution labels can repeat across different partial inputs, as in the original GRAM task.

[Frozen protocol](protocol.json), [runner](study.py), [CPU auditor](analyze.py), [tracked results and hashes](../../../results/research/loki/summary.json), [per-seed counts](../../../results/research/loki/per_seed.csv).
