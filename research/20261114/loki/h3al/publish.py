"""Report per-fit reliability separately from the independent-stream product."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT/'research/20261114/loki/h3al'
PUBLIC = ROOT/'results/research/loki/h3al'


def main():
    s = json.loads((PUBLIC/'summary.json').read_text())
    lines = []
    add = lambda value='': lines.append(value)
    add('# H3al: exact finite-population reliability of positive-witness repair')
    add()
    add('**The prospective per-fit reliability hypothesis is supported for all four fixed BP AdamW fits.** Drawing 64 distinct positives uniformly without replacement from the fixed 1,932-board N13 population gives at least 95% exact N15 capacity with probability 97.232629% for banks 0, 1 and 3, and 95.667695% for bank 2. These are exact combinatorial probabilities, conditional on these fixed fits, this finite source population and the previously examined N15 evaluation universe.')
    add()
    add('![Exact acquisition reliability](../../../../results/research/loki/h3al/acquisition-reliability.png)')
    add()
    add('| BP AdamW fixed fit | P(capacity ≥95%) at 64 positives (%) | Expected capacity (%) | P(exact graph recovery) (%) | P(no repair) (%) | Primary decision |')
    add('|---|---:|---:|---:|---:|---|')
    for r in s['primary']:
        add(f"| Bank {r['bank']} | {r['success_probability_pct']:.6f} | {r['expected_capacity_pct']:.6f} | {r['exact_graph_probability_pct']:.6f} | {r['unchanged_support_probability_pct']:.6f} | {r['decision']} |")
    add()
    add('The primary statement is a conjunction of four **per-fit probabilities ≥90%**. It is not a claim that all four independent acquisition streams succeed jointly with probability ≥90%. The descriptive independent-stream product is **87.943023%**. Within each board, witness classes can co-occur; their full joint frequencies are retained. No independence assumption is made between witness classes.')
    add()
    add('| Fixed fit | 0 positives | 1 positive | 8 positives | 32 positives | 64 positives, primary |')
    add('|---|---:|---:|---:|---:|---:|')
    for bank in range(4):
        by = {r['dose']:r for r in s['rows'] if r['bank'] == bank}
        add('| Bank '+str(bank)+' | '+' | '.join(f"{by[d]['success_probability_pct']:.6f}%" for d in (0,1,8,32,64))+' |')
    add()
    add('The lower doses are descriptive. Expected capacity, exact graph recovery, unchanged-support probability and the independent-stream joint product are also descriptive. Crossing the 95% input-capacity threshold does not imply full graph recovery: the latter has only 38.628055% probability for banks 0/1/3 and 35.675240% for bank 2 at 64 positives. These observations do not alter the frozen primary endpoint.')
    add()
    add('The learner is unchanged from H3ak: remove an inherited finite displacement class only when a selected valid N13 positive witnesses it. There is no refit, target-dependent deletion, new exclusion, exposure recalibration, minimum-rule choice or symmetry augmentation. All other weights, unary terms, cardinality and source temperature unit remain fixed. The full source population is used by this study to evaluate the sampling distribution; an individual 64-positive repair receives only its own selected positives. Positive labels plus size-invariant translation-shared pair prohibitions imply deletion soundness. Sufficient recovery from a finite sample remains an empirical question.')
    add()
    add('The source-only process first validates all 1,932 positive labels and forms the joint witness mask for each board. Banks 0, 1 and 3 each have four observable inherited classes; bank 2 has five. All 80 bank masks are retained, including unchanged and zero-mass outcomes. They give 32 byte-distinct full support graphs, the sole deduplication unit. All source statistics and deletion outcomes are sealed before target scoring. The prospective complexity gates allow at most eight classes per bank, 512 bank masks and 128 full supports; exceeding any cap would stop the whole study without pruning or extension.')
    add()
    add('For each mask S, let n(S) be the number of population boards whose witness mask is contained in S. Then choose(n(S), k) counts selected k-subsets whose union is contained in S. Integer subset Möbius inversion produces the number whose union is exactly S. Every mass must be nonnegative, reconstruct the contained-union counts and sum to choose(1932, k). Combinatorial integers and reduced rational numerators/denominators are stored as decimal strings. Capacity and reliability decisions use integer cross-products: 20 × covered ≥19 ×831605 and 10 × favorable draws ≥9 ×total draws. There are no Monte Carlo standard errors or sampling-seed replications in this exact distribution calculation.')
    add()
    add('Before filtering any support, the evaluator verifies that **every repaired F contains the true F**. Thus every repaired zero-F cardinality-15 state belongs to the complete true census, and exact occupied-pair filtering of all 26,960 true states is complete. All 575 one/two/three-clue subsets of each retained state contribute to an exact Boolean key union over the same 831,605 eligible H3aj inputs. No new target cohort is created, and this is not an untouched-test or learning-population generalization claim.')
    add()
    add(f"The monitored stage used **{s['aggregate_CPU_seconds']:.6f} aggregate CPU seconds** and **{s['wall_seconds']:.6f} wall seconds**, within the frozen 120 CPU-second /180 wall-second gate. This includes the 225 exhaustive small-distribution validation cases, all source label checks and masks, exact probability masses, all support constructions, the completeness gate, all capacities and serialization. GPU work is zero; the exact finite-state calculation has no eligible GPU stage. The separate independent publication audit took {s['audit']['elapsed_seconds']:.6f} wall seconds.")
    add()
    add('The independent public-data audit uses a different grouped union dynamic program: for each joint mask type with frequency n, selecting j boards contributes choose(n, j) and unions that type into the state. It cross-checks every mass at every declared dose, not merely the totals. It also reconstructs every source mask and every deletion outcome, independently re-enumerates both complete true populations with a direct queen/camel solver, proves all repaired-support inclusions before scoring, rebuilds integer clue multiplicities for all 32 supports, compares every packed state/key flag, and recomputes all probabilities, rational thresholds and the descriptive joint product. It reads published inputs only.')
    add()
    add('[Frozen protocol](protocol.json), [source-only outcomes](source_outcomes.py), [integer Möbius distribution](distribution.py), [capacity evaluator](study.py), [publication audit](analyze.py), [public-only independent verifier](verify_public.py), [tracked summary](../../../../results/research/loki/h3al/summary.json), [all probabilities](../../../../results/research/loki/h3al/reliability.csv), [all source masks and exact masses](../../../../results/research/loki/h3al/source-outcomes.json), [figure provenance](../../../../results/research/loki/h3al/plot-provenance.json).')
    (SRC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'README.md').write_text('# H3al: exact positive-acquisition reliability\n\n[Full report](../../../../research/20261114/loki/h3al/REPORT.md). All four fixed BP AdamW fits support the prospective per-fit ≥90% probability of ≥95% exact N15 capacity after 64 uniform distinct N13 positives: 97.232629% for banks 0/1/3 and 95.667695% for bank 2. The descriptive independent-stream joint probability is 87.943023%, a different endpoint. All 80 masks, 32 full supports and exact masses are retained. Independent grouped-union DP, true-population re-enumeration and all integer capacity censuses agree. Same seen finite target universe; full source population evaluates the distribution rather than being supplied to each 64-positive learner. 7.743600 CPU s /7.717084 wall s, zero GPU work.\n')


if __name__ == '__main__':
    main()
