# H1p: constraint recovery from given clues only

The frozen exact-recovery hypothesis is **supported_on_all_six_banks**: 6 of six banks recover exactly the 810 Sudoku peer edges at 3,000 original clue-only training puzzles. Hidden completion digits never enter the estimator or search calibration.

| Training puzzles | Arm | Extra edges across six banks | Exact graphs |
|---:|---|---|---:|
| 100 | clues | 730, 724, 771, 733, 753, 681 | 0/6 |
| 100 | full | 0, 0, 0, 0, 0, 0 | 6/6 |
| 300 | clues | 80, 75, 81, 57, 78, 69 | 0/6 |
| 300 | full | 0, 0, 0, 0, 0, 0 | 6/6 |
| 1000 | clues | 0, 0, 0, 0, 0, 0 | 6/6 |
| 1000 | full | 0, 0, 0, 0, 0, 0 | 6/6 |
| 3000 | clues | 0, 0, 0, 0, 0, 0 | 6/6 |
| 3000 | full | 0, 0, 0, 0, 0, 0 | 6/6 |

| Training puzzles | Method | Mean solved (%) | Wrong stops |
|---:|---|---:|---:|
| — | BP AdamW: outside this count-only study; not run | — | — |
| 300 | clues | 0.9200 | 0 |
| 300 | full | 80.9517 | 0 |
| 3000 | clues | 80.9517 | 0 |
| 3000 | full | 80.9517 | 0 |

Evaluation uses fresh official full-all rows [100000,110000), six paired bank/search seeds, four chains and 262,144 moves per chain. All 120000 unique returned boards were independently checked. All true peer edges remain present, so a zero-energy certificate implies a valid Sudoku; there were no wrong stops.

Clue-only arrays contain observed values, given masks and source rows. Full completions are stored separately for positive controls and auditing. The estimator removes a pair only after observing equal digits with both cells given. It does not receive row, column or box geometry. All supports were saved before the geometry audit. Missing-value sentinel, random replacement, shuffled-observed replacement and empty-mask gates passed at every prefix. A separate post-fit audit also filled masked entries with their true or shuffled completion digits and verified unchanged supports and counts at all 24 prefixes.

Search uses literal unit 1, tau 0 and ebar 0, with zero biases. It never calibrates against hidden training completions. General/sparse state parity passed before each unique search; changing the hidden evaluation targets preserved all trajectory state except target-dependent scoring fields.

The N3000 clue-only versus full-control difference is 0.0000 pp, with the registered 95% interval [0.0, 0.0]. All N3000 clue graphs and full controls are byte-identical for each seed, so this equality follows from the model/settings equivalence rather than duplicate runs. There were 24 requested model/seed cells and 12 unique searches. The two-point secondary noninferiority criterion is satisfied.

A test target having zero model energy is a ceiling for correct certified stops only. A returned best board can match the solution without stopping at zero energy; the oracle diagnostic is kept separate from accuracy.

These are six nested banks from the official dataset, with associated generator-produced masks. They are not uniform draws over all Sudoku grids or independent evidence for arbitrary missingness. The method assumes the cell and digit alphabet, equal-digit repulsion and a zero-energy rule. It does not establish distribution-free sample complexity.

[Protocol](protocol-clues.json) · [Preparation](prepare_clues.py) · [Estimator](clues_estimator.py) · [Search](clues_search.py) · [Audit](audit_clues.py) · [Public results](../../../results/research/ripper/clues.json)
