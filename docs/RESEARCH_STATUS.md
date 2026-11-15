# Research stopped after the final round

The user requested a stop on 2026-11-15. The current scientific runs finished; subsequent experiments were held. Published hypotheses retain their original protocols, failures and audit evidence. Prepared code is not evidence of an experimental result.

## Last measured results

| Study | Result | Scope |
|---|---|---|
| H4be: repeated measurements | 256/256 exact graphs versus prior 253/256; three gains, zero losses | Examined synthetic graph/source cohort, new noise; both fixed targets pass |
| H1ci: source diversity after correlated noise | 72/72 from distinct boards versus 0/72 from correlated repeats | Fresh official-source banks; equal noisy-label budgets, three times as many source boards; monitoring deviation disclosed |
| H2bs: packed maze dispatcher | 31.34% median complete-cost saving versus the original dispatcher | Examined inputs, six paired timing blocks; all outputs and trajectories preserved |
| H3ca: matched exact-tree weights | 6.64% fewer paid calls; both arms identify the same 1,008/1,024 truths | Fresh pairs, common MAP routing; 16 shared failures remain |

These studies address different questions and do not share denominators. BP AdamW controls and scope differences are retained in the individual reports. The original paper reproduction is described separately in [REPRODUCTION.md](../REPRODUCTION.md).

- [H4be result, accounting and portable replay](../research/20261114/foureyes/REPORT-repeat-diversity.md).
- [H1ci result, source freshness and monitoring deviation](../research/20261114/ripper/REPORT-diversity-after-correlation.md).
- [H2bs measured result and later Work stop state](../research/20261114/work/REPORT.md).
- [H3ca query result and proof-archive limits](../research/20261114/loki/h3ca/REPORT.md).

## Prepared studies that were not run

| Study | State at stop | Missing evidence |
|---|---|---|
| H2bt: packed dispatcher versus LP96 and always-cascade | Protocol frozen; functional gates passed on both GPUs; timing never launched | No timed comparison or cost conclusion |
| H2bu: fresh packed-dispatch confirmation | Design only; no source draw or implementation | No fresh accuracy or timing result |
| H3cb: bounded fallback membership queries | Implementation and intermediate synthetic checks retained; no scientific freeze or source/permutation draw | Final registry/packet approval and verification/export/lifecycle integration remain unfinished |
| Fresh confirmation of H4be repeated measurements | Not implemented or run | No fresh-cohort confirmation of the 256/256 result |

The Work report records its frozen and design-only packets. [H3cb's prepared-only status](../research/20261114/loki/h3cb/PREPARED-UNRUN.md) lists the incomplete checks. Resuming any unrun study requires renewed authorization, current source/hardware checks and the outstanding review; do not treat these draft runners as approved scientific results.

H1ci's monitor thread encountered a process-exit race after its last regular sample. The unchanged final synchronous counters, recorded after all 14 child commands were reaped, establish 39.278874 wall seconds and 237.985004 aggregate CPU seconds, below the 180/900 limits. Its report retains the traceback and all 385 samples. This is completed science with a disclosed monitoring deviation, not a claim that supervision remained uninterrupted.
