"""Portable fresh-source and paid-label replay; external proofs remain explicit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from common import load
from audit_queries import PaidQueries
from audit_sources import verify as verify_sources
from audit_costs import choose
from verify_complete import verify as verify_complete


def verify(root):
    R = Path(root).resolve()
    S, P = R/'research/20261114/loki/h3by', R/'results/research/loki/h3by'
    read = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    s, p, g = read(P/'summary.json'), read(S/'protocol.json'), read(P/'capacity-gate.json')
    checks = 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    check('validated_complete_summary', s['status'] == 'validated' and s['cases'] == 4096 and
          s['protocol_sha256'] == sha(S/'protocol.json') and s['verifier_source_sha256'] == sha(S/'verify_public.py') and
          s['analyze_source_sha256'] == sha(S/'analyze.py'))
    for category in ['source_sha256', 'source_data_sha256']:
        for rel, digest in p[category].items():
            check('frozen_identity '+rel, sha(R/rel) == digest)
    for rel, digest in s['public_sha256'].items():
        check('public_metadata '+rel, sha(P/rel) == digest)
    for rel, digest in s['audit_dependency_sha256'].items():
        check('audit_source '+rel, sha(R/rel) == digest)
    boundary, manifest = read(P/'archive-boundary.json'), read(P/'proof-export.json')
    check('external_proof_boundary', boundary['full_proof_payloads_in_git'] is False and
          boundary['full_verification_requires_external_archive'] and
          boundary['proof_manifest_sha256'] == sha(P/'proof-export.json') and
          not any(P.rglob('proof.json')) and not any(P.rglob('proof.json.gz')) and
          len(manifest['proofs']) == manifest['proof_files'] == boundary['omitted_compressed_proof_files'] and
          sum(r['raw_bytes'] for r in manifest['proofs']) == boundary['omitted_raw_proof_bytes'] and
          sum(r['compressed_bytes'] for r in manifest['proofs']) == boundary['omitted_compressed_proof_bytes'])
    raw = gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes())
    chain = load(S.parent/'h3bj/verify_public.py', 'h3by_portable_chain').chain
    events = chain(raw)
    receipt = read(P/'ledger-export.json')
    check('exact_paid_ledger', hashlib.sha256(raw).hexdigest() == receipt['raw_sha256'] == s['ledger_sha256'] and
          receipt['raw_bytes'] == len(raw) and receipt['gzip_sha256'] == sha(P/'query-ledger.jsonl.gz') and
          len(events) == s['ledger_records'])
    bits = np.load(R/'results/research/loki/h3bc/predictions-bits.npy')
    boards = np.load(R/'results/research/loki/h3bc/ordinary13.npy')
    pred = np.unpackbits(bits, axis=1, count=len(boards), bitorder='little').astype(bool)
    scores = read(P/'score-values8.json')
    sources = read(P/'source-banks.json')
    replay = PaidQueries(events, pred, boards, check)
    check('start_identity', replay.take('study_start')['data'] == dict(protocol_sha256=sha(S/'protocol.json'),
          preparation_sha256=sha(P/'preparation.json')))
    source_audit = verify_sources(R, P, p, sources, replay, read, sha, check)
    results = {'deeper': read(P/'deeper-results.json'), 'baseline': read(P/'baseline-results.json')}
    completed = dict(deeper=0, baseline=0)
    active = queried = maps = trees = None
    current_arm = current_case = None
    deeper_finished = baseline_finished = False
    while replay.at < len(events):
        event = events[replay.at]
        kind, data = event['kind'], event['data']
        if kind == 'arm_source_bound':
            arm, case = data['arm'], data['case']
            check('fixed_arm_case_order', arm in completed and case == completed[arm] and
                  current_arm is None and (not deeper_finished if arm == 'deeper' else deeper_finished and not baseline_finished))
            source = sources[case]
            check('source_only_arm_initial_state', data == dict(case=case, arm=arm, candidate=source['candidate'],
                  source_record_sha256=source['source_record_sha256'], initial_candidates=source['initial_candidates'], inherited_queries=0))
            current_arm, current_case = arm, case
            active, queried, maps, trees = list(source['initial_candidates']), [], [], []
            replay.take(kind)
        elif kind == 'query':
            arm, case = current_arm, current_case
            check('query_inside_its_current_arm', arm is not None and data['arm'] == arm and data['case'] == case)
            request, selection, phase = data['request'], data['selection'], data['phase']
            if phase == 'map':
                check('independent_exact_MAP_selection', not trees and len(maps) < 4 and selection == choose(pred, scores, active, queried) and
                      request == dict(bits_path='predictions-bits.npy', scores_path='score-values8.json',
                                      board_count=len(boards), candidates=active, queried=queried))
            else:
                check('tree_label_scope', phase == 'tree' and len(trees) < (4 if arm == 'baseline' else 8-len(maps)) and
                      selection == dict(query_index=selection['query_index'], candidate_predictions={
                          str(c): bool(pred[c, selection['query_index']]) for c in active}))
            active, step = replay.ask(case, sources[case]['candidate'], arm, phase, active, queried, request, selection)
            (maps if phase == 'map' else trees).append(step)
            queried = queried+[step['query_index']]
        elif kind in ['deeper_case_result', 'baseline_case_result']:
            arm = 'deeper' if kind == 'deeper_case_result' else 'baseline'
            check('whole_completed_arm_record', arm == current_arm and data == results[arm][current_case] and
                  data['case'] == current_case and data['map_steps'] == maps and data['tree_steps'] == trees and
                  data['remaining_candidates'] == active and data['membership_calls'] == len(queried) <= 8 and
                  data['map_calls'] == len(maps) and data['tree_calls'] == len(trees) and
                  data['exact_true_singleton'] == (active == [sources[current_case]['candidate']]))
            if arm == 'baseline':
                check('all_cross_arm_reacquisitions_paid', data['independently_paid_cross_arm_reacquisitions'] ==
                      sum(bool(step['previous_same_case_query_sequences']) for step in maps+trees))
            completed[arm] += 1
            current_arm = current_case = None
            replay.take(kind)
        elif kind == 'all_deeper_executions_complete':
            check('full_deeper_phase_completed_before_baseline', completed == dict(deeper=4096, baseline=0) and
                  current_arm is None and data['paid_calls'] == replay.calls['deeper'])
            deeper_finished = True
            replay.take(kind)
        elif kind == 'all_baseline_executions_complete':
            check('full_baseline_phase_completed', completed == dict(deeper=4096, baseline=4096) and
                  current_arm is None and data['paid_calls'] == replay.calls['baseline'])
            baseline_finished = True
            replay.take(kind)
        else:
            check('only_declared_nonquery_records', kind in ['handoff_ineligible_candidate_count',
                  'online_compiled_tree_bound', 'posterior_cost_guard_bound', 'no_informative_query',
                  'original_baseline_tree_bound', 'paired_case_result', 'study_complete'])
            replay.take(kind)
    D, B = replay.calls['deeper'], replay.calls['baseline']
    decision = 'inconclusive_zero_comparator' if B == 0 else 'supported' if 100*D <= 98*B else 'falsified'
    pairs = read(P/'case-results.json')
    for new, old, pair in zip(results['deeper'], results['baseline'], pairs):
        check('every_paired_outcome_record', pair == dict(case=new['case'], candidate=new['candidate'],
              deeper_calls=new['membership_calls'], baseline_calls=old['membership_calls'],
              deeper_exact=new['exact_true_singleton'], baseline_exact=old['exact_true_singleton'],
              all_actual_calls=new['membership_calls']+old['membership_calls'],
              independently_paid_cross_arm_reacquisitions=old['independently_paid_cross_arm_reacquisitions']))
    check('all4096_pairs_and_fresh_primary', completed == dict(deeper=4096, baseline=4096) and len(pairs) == 4096 and
          D == s['deeper_policy_calls'] == sum(r['deeper_calls'] for r in pairs) and
          B == s['baseline_policy_calls'] == sum(r['baseline_calls'] for r in pairs) and
          D+B == s['actual_membership_calls'] and s['decision'] == decision and
          s['primary'] == dict(left=100*D, right=98*B, passing_maximum=(98*B)//100 if B else None, threshold_percent=2) and
          all(new['exact_true_singleton'] == old['exact_true_singleton'] for new, old in zip(results['deeper'], results['baseline'])))
    check('all_exact_failures_and_call_composition', s['exact_cases'] == sum(r['deeper_exact'] for r in pairs) and
          s['retained_failures'] == sum(not r['deeper_exact'] for r in pairs) and
          s['MAP_calls'] == {arm: sum(r['map_calls'] for r in rows) for arm, rows in results.items()} and
          s['tree_calls'] == {arm: sum(r['tree_calls'] for r in rows) for arm, rows in results.items()} and
          s['distinct_N12_boards'] == source_audit['distinct_N12_boards'])
    descriptive = []
    for k in [0, 1, 2]:
        subset = [row for row in pairs if len(p['truths'][row['candidate']]['orbits']) == k]
        dd, bb = sum(r['deeper_calls'] for r in subset), sum(r['baseline_calls'] for r in subset)
        descriptive.append(dict(true_orbit_count=k, cases=len(subset), deeper_calls=dd, baseline_calls=bb,
                                exact_cases=sum(r['deeper_exact'] for r in subset), saving_calls=bb-dd,
                                scope='Descriptive true-orbit stratum only; not an additional scientific endpoint'))
    check('descriptive_strata_and_every_zero_frequency_row', s['descriptive_by_true_orbit_count'] == descriptive and
          s['truth_frequencies'] == [dict(candidate=r['candidate'], eligible=r['eligible'],
              sampling_integer_weight=r['sampling_integer_weight'], cases=source_audit['truth_counts'].get(r['candidate'], 0)) for r in p['truths']])
    note = read(P/'seed-scope-note.json')
    overlap = []
    for source in sources:
        grid, degree = divmod(source['seed']-2027111502, 41)
        if 0 <= grid < 4096 and 1 <= degree <= 40 and not(grid < 2048 and grid % 32 == 0):
            overlap.append(dict(case=source['case'], source_seed=source['seed'], signal_solver_grid=grid, signal_solver_degree=degree))
    check('numerical_solver_seed_overlap_disclosed', note['intersections'] == overlap and
          note['source_seed_numerical_intersections'] == len(overlap) == 3996 and
          note['historical_solver_protocol_sha256'] == sha(R/note['historical_solver_protocol']) and
          note['different_PRNG_applications'] and note['no_redraw_or_retry'])
    check('all_complete_case_statuses', read(P/'case-statuses.json') ==
          [dict(case=i, source='complete', deeper='complete', baseline='complete') for i in range(4096)])
    full = read(P/'full-archive-audit.json')
    check('full_audit_receipt_and_manifest', full['scientific_audit'] == s['audit'] and
          full['scientific_audit']['status'] == 'passed_complete_fresh_two_arm_audit' and
          full['scientific_audit']['deeper_paid_queries'] == D and full['scientific_audit']['baseline_paid_queries'] == B and
          full['proof_manifest_sha256'] == sha(P/'proof-export.json') and full['protocol_sha256'] == sha(S/'protocol.json'))
    for rel, digest in full['audit_source_sha256'].items():
        check('full_audit_source_identity', sha(R/rel) == digest)
    plot = read(P/'plot-provenance.json')
    for rel, digest in plot['input_sha256'].items():
        check('plot_input', sha(R/rel) == digest)
    for rel, digest in plot['output_sha256'].items():
        check('plot_output', sha(P/rel) == digest)
    return dict(status='passed_portable_fresh_source_and_paid_query_audit', checks=checks, cases=4096,
                primary_decision=decision, deeper_paid_queries=D, baseline_paid_queries=B,
                actual_paid_queries_replayed=D+B, all_fresh_source_choices_reconstructed=True,
                full_optimality_certificates_replayed=False,
                limitation='Full tree, original-baseline and counterfactual optimality replay requires the separately retained external archive; its prior full audit receipt and every proof hash are public')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--archive-dir', type=Path)
    args = parser.parse_args()
    result = dict(portable=verify(args.root))
    if args.archive_dir is not None:
        result['external_full_audit'] = verify_complete(args.root, args.archive_dir)
    print(json.dumps(result))
