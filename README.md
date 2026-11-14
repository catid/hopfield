# Hopfield: small energies, better reasoning

**99.9567% Sudoku accuracy with a count-based energy and no optimizer.**

An independent implementation, reproduction and experimental extension of [*Classical Energies Learn to Reason 1,000× Faster than Looped Models*](https://arxiv.org/abs/2611.17309v1), by Benjamin Hoover, Hugh Blayney and James C. R. Whittington. CRAM learns a classical energy function, then searches for low-energy solutions. Our follow-ups test what the optimizer, data augmentation and search each contribute.

[Paper in reviewed Markdown](references/2611.17309v1.md) · [Research results](results/README.md) · [Reproduction](REPRODUCTION.md) · [Run the code](docs/RERUN_GUIDE.md)

## Simple counts match the transformed Sudoku energy

![Count-based Sudoku energy: matched short-budget accuracy and longer-budget failure curves](assets/figures/count-energy.png)

Connect two cells if they **never share a digit in the training solutions**. Give each retained pair weight one, then run the same focused Gibbs search. This simple estimator recovers the same 810-edge constraint graph as our transformed learned energies, without an optimizer or a supplied Sudoku peer graph.

On a reserved **10,000-puzzle official test slice**, the count energy solves **59,974/60,000 evaluations (99.9567%)** across six search seeds, with **zero wrong stops**, using four chains and **16,777,216 moves per chain**. The transformed learned energy reaches 99.9417% under the same budget. The paired difference is +0.015 points; its one-sided 95% lower bound of −0.0151 points passes the frozen 0.1-point noninferiority margin. At the shorter **262,144-move** cap, the count method solves **81.50%**.

**The augmentation prior matters.** These training counts use ten unique base solutions **plus 99 stored Sudoku-symmetry variants per base**. The transformations supply task knowledge. A larger one-base-solution study recovered the exact graph in **8,105/8,192 fresh trials** with 99 spatial variants. This uses augmentation-group knowledge; it is not one unaugmented example learning Sudoku. [Coverage theorem and fresh-RNG validation →](research/20261114/ripper/REPORT-coverage.md) The count comparison also reuses the official corpus and a follow-up cohort. [Protocols, full budget curve, stopping proof and limitations →](research/20261114/foureyes/REPORT-budget.md) · [Count and augmentation controls →](research/20261114/ripper/REPORT-count.md)

**Spatial augmentation is not required with more raw solutions.** Counting recovers the exact graph from **100 distinct, unaugmented training solutions in all six samples**. A stronger matched control then compared counting with the fixed transformation pipeline applied to digit-relabeling PL fits. At 262,144 moves per chain, counting solves **80.79%** versus **80.46%** for transformed PL, with zero wrong stops; PL passes the frozen 2-point noninferiority margin. The simpler count estimator matches a properly transformed learned model. [Raw-count structure →](research/20261114/ripper/REPORT-raw-count.md) · [Matched transformation control →](research/20261114/foureyes/REPORT-raw-transform.md)

![Constraint recovery from raw solutions and the matched transformed-PL control](assets/figures/raw-count-transformed.png)

Hidden solution labels can also be removed: counts from **3,000 partially observed training puzzles** recover the same graph in all six samples. Masked digits never enter estimation or calibration. These are official clue masks, whose selection may carry useful information. [Clues-only experiment and leakage checks →](research/20261114/ripper/REPORT-clues.md)

**Label noise exposes a failure—and a simple repair.** Corrupting one digit in three of 300 training solutions drops strict-count accuracy from 82.01% to 55.12%. Tolerating one observed equality recovers the original graph and restores 82.01%, with zero wrong stops. This tests one declared noise level and threshold. [Noise protocol and audit →](research/20261114/foureyes/REPORT-noise.md)

The earlier intervention result remains useful: digit symmetry and repulsion increased low-data accuracy by **47.83 points** on a separate reserved slice. Counting explains why the transformed models recover the same graph. [Earlier matched results →](research/20261114/ripper/REPORT-heldout.md)

## The same transformed model, faster inference

![Measured sparse backend speedup by batch size and training size](assets/figures/sparse-speed.png)

A sparse backend runs the same repulsive energies **2.82× faster** in warm inference, or **2.79× including search setup**, on RTX PRO 6000 Blackwell Max-Q. All 63 paired warmup/timing calls produced identical final states and boards. Broader bounded checks covered 18 models, followed by a full 10,000-puzzle integration comparison. Compilation is excluded from warm timing. [Timing data, parity checks and limitations →](research/20261114/ripper/REPORT-sparse.md)

## Larger mazes, fewer inference steps

![Maze step-size intervention: fewer updates and more solved mazes within a fixed budget](results/research/work/eta-acceleration.png)

Changing the inference step size from **0.2 to 0.5**, with the learned 29-parameter maze energy unchanged, reduces median answer updates by **2.38× at 60×60** and **2.42× at 90×90**. Measured hidden-plus-answer runtime improves **2.09× and 2.21×**. Both arms solve all 768 generated mazes at the 8,192-update cap, across three frozen models, with no lost solves.

At a tighter 2,048-update cap, joint success rises from **351/384 to 384/384** at 60×60 and **285/384 to 370/384** at 90×90. These generated tree/loopy mazes differ from the original Maze-Hard distribution. [Frozen protocol, confidence intervals and causal branch-depth experiments →](research/20261114/work/REPORT.md)

## N-Queens: unseen solutions from training counts

![N-Queens-12 accuracy and solution coverage on inputs incompatible with every training pattern](results/research/loki/h3g/pattern-holdout.png)

With **512 complete training patterns**, a count-based exclusion energy reaches **100% accuracy and 76.86% solution coverage** on 8,192 new N-Queens-12 partial inputs. Every valid completion lies outside the training bank, so retrieving compatible training examples cannot solve any of these inputs. Adam at 250 updates reaches 99.9919% accuracy and 67.93% coverage. The count method gains **8.94 coverage points**; the paired one-sided 95% lower bound is +8.78 points. Ordinary BP AdamW at 4,000 updates reaches 99.5321% accuracy and 53.10% coverage.

These measurements use 20 search chains and three search seeds, with a certificate-first output rule. The count method uses no optimizer, but retains a fixed energy design and displacement-sharing architecture. The test distribution is conditioned on incompatibility with the training bank and consists almost entirely of three-clue inputs; it differs from the original benchmark. The 512 patterns expand into 152,576 labeled training examples. [Frozen split, all controls and independent 4.9-million-board audit →](research/20261114/loki/h3g/REPORT.md)

Sharing is consequential: with 32 training patterns, removing displacement sharing drops accuracy from 97.39% to 0%. False exclusions can also make invalid initial boards pass the learned certificate. [Matched architecture ablation and certificate audit →](research/20261114/loki/h3h/REPORT.md)

The earlier study included every complete pattern in training, where simple bank retrieval was already competitive. That control motivated this stricter follow-up. [Full-bank controls and limitations →](research/20261114/loki/h3f/REPORT.md)

Earlier N-Queens-10 experiments showed faster fitting with Adam and a weight-decay gap. A learned-certificate-first output policy then fixed all 77 failed AdamW input–seed cases without extra search, showing that the gap depends on output selection. [Optimizer experiments →](research/20261114/loki/h3c/REPORT.md) · [Selection control →](research/20261114/loki/h3d/REPORT.md)

## Failures are part of the result

The maze step-size benefit has a limit: a new maze with many tied shortest paths produces certified branched or disconnected answers for two models. Simple tied-cycle controls still solve perfectly, so ties alone do not explain the failure. [Saved graph, boards and causal controls →](research/20261114/work/REPORT.md)

Exhaustively scoring all **352,212 shortest paths** in that maze finds that a disconnected endpoint-only state has lower energy than every valid shortest path, for all three models. On 4,096 fresh cyclic mazes, checking candidate path feasibility rescues energy-based selection by 42.58 points—but does not improve the native initialization baseline. [Energy landscape and fresh selector controls →](research/20261114/work/REPORT.md)

The learned Sudoku pipeline's four-million-move **99% all-seed joint-success hypothesis was inconclusive**, despite 99.635% mean accuracy. The sixteen-million-move secondary endpoint passed. Projection still fails at extremely low data; largest-gap pruning collapses without spatial augmentation; and maze mass rescaling gives no improvement. [All hypotheses, including negative and inconclusive outcomes →](results/README.md)

Fixed Sudoku restarts also failed their frozen +0.1-point improvement target: the gain was +0.012 points, with a paired 95% interval [−0.068, +0.093]. [Restart control and full curve →](research/20261114/foureyes/REPORT-restart.md)

## What was reproduced?

The original campaign reproduced **99.996689% Sudoku accuracy** over three full held-out runs, **100% Maze-Unique**, **99.9% Maze-Hard**, and every graph-coloring quality cell. It also completed all **84 objective runs, 100 training curves, 102 data-efficiency runs**, and the declared controls and supplemental ablations. The independent implementation passed 27 CPU tests; the pinned authors' implementation passed 91 tests.

![Original six-seed data-efficiency reproduction](results/reproduction/data_efficiency/figure3b.png)

This curve uses a seeded recreation of unreleased training arrays. Baselines are clearly labeled **published results**. The paper's H200 **“1,000×” training-speed claim is not independently established** by these Blackwell experiments. [Measured reproduction and limits →](REPRODUCTION.md) · [Paper coverage →](docs/PAPER_COVERAGE.md)

## Get started

```sh
git clone https://github.com/catid/hopfield.git
cd hopfield
git clone https://github.com/Prefrontal-AI/CRAM.git upstream
git -C upstream checkout --detach 62bbe830c7d6f87f2ba18f30a28039258b986454
uv sync --project upstream --frozen --python 3.13.13
uv pip install --python upstream/.venv/bin/python --no-deps -e .
CUDA_VISIBLE_DEVICES= upstream/.venv/bin/python -m pytest -q tests/test_reference.py tests/test_counting.py
```

Build the optimizer-free baseline from your own completed training states:

```python
from hopfield_cram import fit_never_equal

# Integer tensor (examples, 81), digits 0..8; training data only.
energy, equality_counts = fit_never_equal(training_solutions, n_values=9)
```

This [small public API](src/hopfield_cram/counting.py) supplies no cell geometry. For the declared noisy-label control, use `fit_rarely_equal(training_solutions, n_values=9, max_equal_count=1)`. Positive thresholds may add false exclusions and give training states positive energy; choose inference calibration separately. With insufficient data, even strict counts can exclude unseen valid boards. The [published binary model](results/research/foureyes/count_budget/model.json) retains the training provenance behind the measured Sudoku result. All 38 reference/counting tests pass; the tolerant API exactly matches all 24 noise-study model artifacts.

The readable implementation is in [src/hopfield_cram](src/hopfield_cram/). Full-scale runs use the pinned upstream kernels and the isolated [research runners](research/20261114/). [Setup, data preparation and rerun instructions](docs/RERUN_GUIDE.md) explain their hardware and dataset requirements; [implementation notes](docs/IMPLEMENTATION.md) explain the algorithms.

Public [JSON/CSV results](results/README.md), [figure sources](docs/FIGURES.md), protocols and audit hashes are included. Large datasets, checkpoints and raw board archives are excluded from Git. The [development history and original completion snapshot](results/reproduction/README.md) are retained separately. Research is ongoing; only audited results appear above.

Code: [Apache 2.0](LICENSE). Paper and its converted assets: CC BY 4.0. [Attribution and scope](NOTICE.md).
