# Loki: optimizer and weight-decay experiments

The fixed-learning-rate weight-decay hypothesis was supported on 6,513 unique official n=10 test inputs with 20 chains and three paired search seeds. BP AdamW (decay 0.01) reached 99.605916% N20 accuracy; setting decay to zero reached 100%, and preconditioned SGD reached 99.984646%. Coverage was 92.283227%, 93.559127%, and 93.585165%, respectively.

The original AdamW-versus-preconditioned-SGD hypothesis was falsified on its disjoint development confirmation cohort. H3b then tested the decay intervention at the already-selected learning rate 0.3. It used the same 97,554 fitting rows, 4,000 full-batch updates and normalized inference temperature. All models differentiate the local conditional loss; BP does not unroll search.

The no-decay accuracy difference versus preconditioned SGD was +0.015354 percentage points, with one-sided 95% endpoints [0, 0.046062]. Coverage difference was -0.026039 points, endpoints [-0.206310, 0.142826]. The conservative exact regression bound was 0.045986%, below the preregistered 0.1-point margin. Search seeds are repeated draws within inputs, not independent training seeds.

[summary.json](summary.json) includes decisions, aggregate results, uncertainty, source/model/protocol/outcome hashes and limits. [per_seed.csv](per_seed.csv) includes integer success and coverage counts for every search, plus hashes. The larger saved boards and checkpoints remain in the local artifact archive. These files do not assert an optimizer advantage across tasks or a hardware speedup.

The separate 1,024-input budget control captures 20,000 and 80,000 moves from the same trajectory. The AdamW-decay accuracy changed from 99.348958% to 99.479167%; no-decay Adam and preconditioned SGD remained at 100%. This fixed canonical subset is not a random representative sample of the full test. Snapshot parity was verified against the upstream solver; multi-block snapshot capture is rejected because shared RNG state couples blocks.

Protocols and full reports: [H3](../../../research/20261114/loki/REPORT.md), [H3b](../../../research/20261114/loki/h3b/REPORT.md). Regenerate after independent audits with `upstream/.venv/bin/python research/20261114/loki/export_public.py`.
