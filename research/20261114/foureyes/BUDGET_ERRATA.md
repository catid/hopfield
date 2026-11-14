# Budget protocol clarifications

Frozen protocol bytes and hashes are preserved.

The H4d `training_k: 10` refers to **ten unique base puzzles plus 99 stored variants per puzzle** (1,000 augmented examples), not ten unaugmented boards. The exact training input is pinned by the original fit and continuation receipts.

H4e was derived from H4d's protocol. Its inherited `limits` sentence about timing excluding a prior prefix applies only to H4d. H4e executes a **fresh count search from move zero**, as specified by its controlling `continuation` field and implemented by `count_budget.py`. Its timing excludes validation, setup and saved-output serialization. The inherited `primary_accuracy_fraction: 0.99` is not H4e's primary decision: the explicit `hypothesis`, `primary_margin_pp`, `uncertainty` and `decision` fields control the 0.1-point comparative noninferiority test. Absolute 99% joint success is secondary.
