"""Build a reviewable protocol; freezing requires the coordinator's source review."""
import argparse
from common import ROOT, SRC, OUT, read, save, sha, now


def main(draft=False):
    assert not (SRC/'protocol.json').exists() and not (OUT/'preparation.json').exists()
    prior = read(SRC.parent/'h3br/protocol.json')
    previous = read(SRC.parent/'h3bw/protocol.json')
    pre = read(SRC/'preflight.json')
    seed = read(SRC/'seed-audit.json')
    audit = read(SRC/'component-audit.json')
    monitor = read(SRC/'monitor-gate.json')
    assert pre['status'] == 'passed_prefreeze_synthetic_gates'
    assert pre['independent_inline']['status'] == 'passed_independent_synthetic_artifact_audit'
    assert pre['independent_baseline']['status'] == 'passed_independent_baseline_artifact_audit'
    assert pre['wrapper_gate']['status'] == 'passed_independently_verified_baseline_wrapper_gate'
    assert pre['actual_oracle_calls'] == pre['new_source_draws'] == pre['new_truth_draws'] == 0
    assert not seed['explicit_collisions'] and not seed['expanded_collisions']
    assert audit['status'] == 'passed_independent_complete_verifier_components'
    assert all(audit['paid_query_binding_gate']['all10_rehashed_semantic_mutations_rejected'].values())
    for name, digest in audit['source_sha256'].items():
        assert sha(SRC/name) == digest, name
    for name, digest in audit['paid_query_binding_gate']['source_sha256'].items():
        assert sha(SRC/name) == digest, name
    assert monitor['status'] == 'passed_monitor_accounting_probes'
    assert monitor['gate_source_sha256'] == sha(SRC/'gate.py')
    assert monitor['probe_source_sha256'] == sha(SRC/'monitor_gate.py')
    for name, digest in pre['scientific_source_sha256'].items():
        assert sha(SRC/name) == digest, name
    for name in ['compiler.py', 'policy.py', 'counterfactual.py', 'selector.py', 'weights.py', 'ledger.py']:
        assert sha(SRC/name) == sha(SRC.parent/'h3bw'/name), name
    for new, old in [('baseline_compiler.py', 'compiler.py'), ('baseline_firstlex.py', 'firstlex.py'), ('prior.py', 'prior.py')]:
        assert sha(SRC/new) == sha(SRC.parent/'h3br'/old), new
    inputs = set(previous['source_data_sha256']) | set(seed['historical_protocol_sha256'])
    inputs.update(['research/20261114/loki/h3bw/protocol.json', 'results/research/loki/h3bw/summary.json',
                   'results/research/loki/h3br/source-banks.json', 'results/research/loki/h3bm/source-pairs.json',
                   'results/research/loki/h3bq/source-banks.json', 'research/20261114/loki/h3bi/verify_public.py',
                   'research/20261114/loki/h3bj/verify_public.py', 'research/20261114/loki/h3bl/verify_public.py',
                   'research/20261114/loki/h3bv/verify_tree.py',
                   'research/20261114/loki/archive_export_layer.py', 'research/20261114/loki/export_layer.py'])
    inputs.update(str((SRC/name).relative_to(ROOT)) for name in
                  ['DESIGN.md', 'BOUND_LEMMA.md', 'GUARD_LEMMA.md', 'DEPTH_LEMMA.md', 'LEMMA.md',
                   'preflight.json', 'prior-gate.json', 'seed-audit.json', 'component-audit.json', 'monitor-gate.json'])
    sources = sorted(SRC.glob('*.py'))+[SRC.parent/'h3at/source.py']
    p = dict(
        id='H3bx-fresh-matched-prior-deeper-policy-confirmation',
        frozen_utc=None if draft else now(),
        hypothesis='Across all 4096 fresh paired matched-prior cases, the fixed H3bw deeper handoff uses at least 2% fewer total actual membership calls than the separately executed original H3br four-MAP/depth-four posterior policy. Sole finite-cohort scientific endpoint: 100*D <= 98*B. No inherited historical comparator total.',
        primary=dict(multiplier=100, comparator_multiplier=98, threshold_percent=2,
                     cost_scope='All actual MAP and tree calls in each arm, across all 4096 cases',
                     zero_comparator='inconclusive_zero_comparator',
                     missing_or_capped='whole_study_incomplete_no_primary_decision'),
        candidate_prior=dict(prior['candidate_prior'], definition='All 1541 ordinary-queen plus zero/one/two D4 leaper-orbit models retained; source prior conditions M>=8, with no restriction by true orbit count in inference'),
        normalizers=prior['normalizers'], truths=prior['truths'],
        seed_bases=dict(truth=2026900000, source=2027000000),
        historical_protocol_sha256=seed['historical_protocol_sha256'],
        truth_sampling='For i=0..4095: default_rng(2026900000+i).integers(4436); exact half-open cumulative prior tickets in candidate ID order. Eligible integer weights 1485/27/1 for orbit counts 0/1/2, conditional M>=8. Exactly 1522 eligible and 19 ineligible class rows; retain all frequencies including zero. No pilot draws, seed tuning, redraw or sample extension.',
        source_sampling='For i=0..4095: independent default_rng(2027000000+i).choice(M,8,replace=False) for the sampled truth. Draw and durably log all sources before any prospective target query. Retain every repeated truth, unordered duplicate bank and historical eight-bank match. One shared source bank per pair: 32768 new positive appearances, 65536 arm appearances.',
        routing=previous['routing'], weighting=previous['weighting'],
        compilation=previous['compilation'], counterfactual=previous['counterfactual'],
        cost_guard=previous['cost_guard'], expected_cost_property=previous['expected_cost_property'],
        baseline='After all deeper-arm executions, independently start each original baseline case from its eight positives and empty query history. Up to four unchanged MAP calls then original posterior-optimal depth-four tree; original firstlex feasibility and full-enumeration weighted compiler, unchanged lexicographic ties. Infeasible terminal sets make zero additional tree calls and retain exact=False. Its separate cache never reuses deeper-arm or counterfactual computations.',
        execution='All runtime gates precede prospective draws; all 4096 truth/source draws precede queries; all deeper-arm executions precede all separately executed original-baseline executions; paired outcome comparison occurs last. Every actual oracle read in both arms is separately paid, including a repeated board or common MAP prefix. Query metadata can annotate previous same-case queries only after selection. No other-arm answers or candidate states are inherited.',
        correctness='Every one of 4096 paired exact flags must agree. No fixed success/failure count is assumed. Admissible baseline containment and expected nonincrease under the prescribed posterior are correctness properties, distinct from realized >=2% saving. Pure cost rejection for an admissible complete baseline is an error. All failures and duplicates remain in the primary denominator.',
        ledger=previous['ledger'], ledger_gate=previous['ledger_gate'],
        source_gate='Independently recount all source cardinalities, enumerate all 4436 exact tickets, preserve the full1541 prior table. Verify 8192 proposed seeds against explicit seed-valued fields in pinned prior protocols and expanded known prior base+candidate/bank/case formula ranges, including conservative unused combinations. No claim about undocumented streams. No prospective RNG calls during preflight.',
        gates='Repeat H3bw synthetic bound/pruning/canonical-reuse/cost/exact/ledger and restriction fixtures inside monitored budget. Add 4252 fixed depth-four original firstlex/full-enumeration/all-inline comparisons with identical feasibility, costs and every candidate route; independently replay all three serialized certificates. Four baseline-wrapper fixtures cover feasible, weighted, exhaustive infeasible and n17 information-bound cases, separate zero-work cache hits and compact paths. All scientific gate work is charged before the first fresh draw.',
        budget=dict(aggregate_CPU_seconds=600, wall_seconds=900, GPUs=0,
                    max_memo_states_total=2000000, max_split_evaluations_total=50000000,
                    max_integer_bits=100000, membership_calls=65536,
                    membership_calls_per_arm=32768, planned_cases=4096,
                    new_positive_appearances=32768, positive_arm_appearances=65536,
                    enforcement='Aggregate both realized arms, independent baseline compiler, all counterfactuals, all runtime gates and reference computations, preparation, serialization, hashing and monitor. Charge tree/counterfactual/bound memo states, canonical partition attempts, every recomputed Huffman merge and counterfactual MAP partition. No persistent Huffman bound cache in the deeper compiler. Caps/errors make the entire4096case study incomplete; preserve all statuses, paid labels and complete artifacts. No continuation, cap extension or partial endpoint. Post-run compression and independent audit separately timed.'),
        publication='Complete cases, both separately paid arms, append-only query chain, protocol/source identities, plots and full audit receipt public. Large proof payloads outside Git with every raw/compressed identity; portable verification explicitly cannot replay omitted optimality certificates.',
        interpretation='Fresh finite-cohort confirmation of the already-fixed deeper policy under the declared matched prior. Passing does not imply a uniform population effect size, arbitrary-rule learning, positive-only learning or learned-optimizer superiority. Binary membership and supplied predictions are stronger supervision. BP AdamW not measured and first in method comparison tables.',
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sources},
        source_data_sha256={rel: sha(ROOT/rel) for rel in sorted(inputs)}, target_data_sha256={})
    destination = SRC/('proposal-protocol.json' if draft else 'protocol.json')
    save(destination, p)
    print('DRAFT' if draft else 'FROZEN', sha(destination), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true')
    main(parser.parse_args().draft)
