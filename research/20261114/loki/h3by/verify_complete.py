"""Full independent fresh-source, two-arm paid-query and proof replay."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from common import load
from audit_io import AuditIO
from audit_queries import PaidQueries
from audit_sources import verify as verify_sources
from audit_runtime import verify as verify_runtime
from audit_trees import TreeAudit
from audit_original import OriginalAudit
from audit_costs import GraphAudit, choose, exact_guard


def verify(root, archive_dir):
    started = time.perf_counter()
    R, P = Path(root).resolve(), Path(archive_dir).resolve()
    S = R/'research/20261114/loki/h3by'
    checks = 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    io = AuditIO(P, S, check)
    read, sha = io.read, io.sha
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    p, g, raw, prep = read(S/'protocol.json'), read(P/'capacity-gate.json'), read(P/'study-result.json'), read(P/'preparation.json')
    check('complete_inside_unchanged_aggregate_caps', g['status'] == raw['status'] == 'complete_verified' and
          not g['monitor_expired'] and g['aggregate_cpu_seconds'] <= 600 and g['wall_seconds'] <= 900 and
          g['global_memo_states'] <= 2000000 and g['global_split_evaluations'] <= 50000000 and
          g['full_monitor_process_lifetime_CPU_charged'] and g['initial_source_hashing_inside_wall_and_CPU_budget'] and
          g['subprocess_inclusive_CPU_accounting'] and all(g[k] == v for k, v in raw.items() if k != 'utc'))
    check('declared_cases_and_budgets', p['budget']['planned_cases'] == 4096 and
          p['budget']['membership_calls'] == 65536 and p['budget']['membership_calls_per_arm'] == 32768 and
          p['seed_bases'] == dict(truth=2027100000, source=2027200000))
    for category in ['source_sha256', 'source_data_sha256']:
        for relative, digest in p[category].items():
            check('frozen_identity '+relative, sha(R/relative) == digest)
    for name in ['case-results', 'case-statuses', 'source-banks', 'deeper-results', 'baseline-results',
                 'groups', 'baseline-groups', 'counterfactual-groups', 'preparation']:
        check('complete_artifact_seal '+name, g[name.replace('-', '_')+'_sha256'] == sha(P/(name+'.json')))
    check('protocol_and_scientific_source_seals', g['protocol_sha256'] == prep['protocol_sha256'] == sha(S/'protocol.json') and
          g['source_sha256'] == sha(S/'study.py') and prep['preflight_sha256'] == sha(S/'preflight.json'))
    for name in ['population-parity', 'ledger-gate', 'synthetic-gate', 'restriction-gate',
                 'pruning-gate', 'cache-gate', 'cost-gate', 'baseline-gate', 'wrapper-gate']:
        check('runtime_gate_seal '+name, prep[name+'_sha256'] == sha(P/(name+'.json')))
    check('source_gate_seals', prep['source_prior_gate_sha256'] == sha(P/'source-gates/prior-gate.json') and
          prep['fitted_weight_gate_sha256'] == sha(P/'source-gates/fitted-weight-gate.json') and
          prep['source_gate'] == read(P/'source-gates/seed-audit.json') and prep['all_runtime_gates_before_prospective_draws'])
    bits = np.load(R/'results/research/loki/h3bc/predictions-bits.npy')
    boards = np.load(R/'results/research/loki/h3bc/ordinary13.npy')
    pred = np.unpackbits(bits, axis=1, count=len(boards), bitorder='little').astype(bool)
    scores = read(P/'score-values8.json')
    check('complete_universe_and_scores_pinned', pred.shape == (1541, 73712) and
          scores == read(R/'results/research/loki/h3bm/score-values8.json'))
    ledger_bytes = io.ledger()
    chain = load(S.parent/'h3bj/verify_public.py', 'h3by_independent_chain').chain
    events = chain(ledger_bytes)
    check('exact_raw_append_only_ledger_identity', hashlib.sha256(ledger_bytes).hexdigest() == g['ledger_sha256'] and
          len(events) == g['ledger_records'] and events[-1]['sha256'] == g['ledger_last_record_sha256'] and
          p['frozen_utc'] < prep['utc'] < events[0]['utc'] < raw['utc'])
    gate = read(P/'ledger-gate.json')
    synthetic_bytes = io.data(P/'synthetic-ledger.jsonl')
    synthetic_events = chain(synthetic_bytes)
    check('synthetic_durability_gate_identity', sha(P/'synthetic-ledger.jsonl') == gate['synthetic_ledger_sha256'] and
          len(synthetic_events) == 16 and gate['oracle_callbacks_observed_durable_query_before_label'] == 8 and
          all(gate['tamper_cases_rejected'].values()) and gate['source_sha256'] == sha(S/'ledger_gate.py') and
          gate['ledger_source_sha256'] == sha(S/'ledger.py'))
    for i, (query_path, answer_path) in enumerate(p['ledger_gate']['archival_query_answer_paths']):
        oldq, olda = read(R/query_path), read(R/answer_path)
        query, answer = synthetic_events[2*i:2*i+2]
        receipt = gate['paired_format_parity'][i]
        check('synthetic_archived_selection_and_answer', query['data'] == dict(index=i,
              request=oldq['request'], selection=oldq['selection'], board_zero_based=oldq['board_zero_based']) and
              answer['data'] == dict(index=i, label=olda['label'], candidates_before=olda['candidates_before'],
              candidates_after=olda['candidates_after'], query_sha256=query['sha256']) and
              receipt['archival_query_sha256'] == sha(R/query_path) and receipt['archival_answer_sha256'] == sha(R/answer_path) and
              receipt['new_query_record_sha256'] == query['sha256'] and receipt['new_answer_record_sha256'] == answer['sha256'] and
              query['monotonic_ns'] < receipt['oracle_callback_monotonic_ns'] < answer['monotonic_ns'])
    gates = verify_runtime(P, S, prep, check, io, canonical)
    sources = read(P/'source-banks.json')
    new_rows, old_rows, paired = read(P/'deeper-results.json'), read(P/'baseline-results.json'), read(P/'case-results.json')
    check('entire_paired_cohort_and_all_statuses', len(new_rows) == len(old_rows) == len(paired) == len(sources) == 4096 and
          [row['case'] for row in new_rows] == [row['case'] for row in old_rows] == [row['case'] for row in paired] == list(range(4096)) and
          read(P/'case-statuses.json') == [dict(case=i, source='complete', deeper='complete', baseline='complete') for i in range(4096)])
    ledger = PaidQueries(events, pred, boards, check)
    check('study_start_seal', ledger.take('study_start')['data'] == dict(protocol_sha256=sha(S/'protocol.json'),
          preparation_sha256=sha(P/'preparation.json')))
    source_audit = verify_sources(R, P, p, sources, ledger, read, sha, check)
    ids = dict(predictions_sha256=sha(R/'results/research/loki/h3bc/predictions-bits.npy'),
               scores_sha256=sha(P/'score-values8.json'), selector_sha256=sha(S/'selector.py'),
               tree_compiler_sha256=sha(S/'compiler.py'), tree_wrapper_sha256=sha(S/'policy.py'))
    groups = read(P/'groups.json')
    trees = TreeAudit(P, S, groups, pred, scores, ids['predictions_sha256'], check, read, sha, canonical)
    selection_cache = {}
    def select(candidates, queried):
        key = tuple(candidates), tuple(queried)
        if key not in selection_cache:
            selection_cache[key] = choose(pred, scores, candidates, queried)
        return selection_cache[key]
    def request_for(candidates, queried):
        return dict(bits_path='predictions-bits.npy', scores_path='score-values8.json',
                    board_count=len(boards), candidates=candidates, queried=queried)
    costs = GraphAudit(P, read(P/'counterfactual-groups.json'), pred, select, trees.get,
                       ids, check, read, sha, canonical, io.logical_cost_groups)
    for source, stored in zip(sources, new_rows):
        case, truth = source['case'], source['candidate']
        candidates, queried = list(source['initial_candidates']), []
        maps, steps, routes, guards = [], [], [], []
        handoff = handoff_candidates = handoff_depth = terminal = None
        check('deeper_initial_full_source_binding', ledger.take('arm_source_bound')['data'] == dict(
              case=case, arm='deeper', candidate=truth, source_record_sha256=source['source_record_sha256'],
              initial_candidates=candidates, inherited_queries=0))
        while len(candidates) > 1:
            depth, row, proof, decision = 8-len(maps), None, None, None
            if len(candidates) > 16:
                data = dict(case=case, MAP_calls=len(maps), requested_depth=depth, candidates=candidates,
                            queried=queried, reason='candidate_count_eligibility_limit', limit=16,
                            not_an_infeasibility_certificate=True)
                event = ledger.take('handoff_ineligible_candidate_count')
            else:
                event = ledger.take('online_compiled_tree_bound')
                row, proof = trees.get(candidates, event, depth)
                group = row['group']
                data = dict(case=case, MAP_calls=len(maps), requested_depth=depth, candidates=candidates,
                            queried=queried, group=group, feasible=proof['feasible'], proof_sha256=row['proof_sha256'],
                            group_result_sha256=sha(P/f'group{group}/result.json'), cache_key=row['cache_key'])
            check('deeper_routing_receipt_and_input', event['data'] == data)
            routes.append(dict(data, bound_sequence=event['sequence'], bound_sha256=event['sha256']))
            if proof is not None and proof['feasible']:
                event = ledger.take('posterior_cost_guard_bound')
                baseline_cost = costs.verify(event['data']['baseline_cost_group'], candidates, queried, 4-len(maps), event)
                decision = exact_guard(proof, baseline_cost)
                data = dict(case=case, MAP_calls=len(maps), requested_depth=depth, tree_group=group,
                            baseline_cost_group=baseline_cost['group'],
                            baseline_result_sha256=sha(P/f'cost{baseline_cost["group"]}/result.json'),
                            baseline_cache_key=baseline_cost['cache_key'], **decision)
                check('every_guard_candidate_cost_and_exact_flag', event['data'] == data and
                      (not decision['baseline_admissible'] or decision['cost_comparison_passes']))
                if len(maps) == 4:
                    check('MAP4_exact_tie', decision['handoff'] and decision['tie'])
                guards.append(dict(data, sequence=event['sequence'], sha256=event['sha256']))
            if proof is not None and proof['feasible'] and decision['handoff']:
                handoff, handoff_candidates, handoff_depth = len(maps), list(candidates), depth
                key = proof['root_state']
                while proof['states'][key]['reason'] != 'singleton_leaf':
                    node = proof['states'][key]
                    index = node['chosen_split']['board_index']
                    request = dict(group=group, proof_sha256=row['proof_sha256'], state=key,
                                   candidates=candidates, queried=queried)
                    selection = dict(query_index=index, candidate_predictions={str(c): bool(pred[c, index]) for c in candidates})
                    candidates, step = ledger.ask(case, truth, 'deeper', 'tree', candidates, queried, request, selection)
                    steps.append(step)
                    queried = queried+[index]
                    key = node['positive_state'] if step['label'] else node['negative_state']
                break
            if len(maps) == 4:
                break
            request, selection = request_for(candidates, queried), select(candidates, queried)
            if selection['query_index'] is None:
                event = ledger.take('no_informative_query')
                check('deeper_no_information', event['data'] == dict(case=case, arm='deeper', request=request, selection=selection))
                terminal = dict(sequence=event['sequence'], sha256=event['sha256'])
                break
            candidates, step = ledger.ask(case, truth, 'deeper', 'map', candidates, queried, request, selection)
            maps.append(step)
            queried = queried+[step['query_index']]
        result = dict(case=case, candidate=truth, source_record_sha256=source['source_record_sha256'],
                      initial_candidates=source['initial_candidates'], routing=routes, cost_guards=guards,
                      map_steps=maps, tree_steps=steps, handoff_MAP_calls=handoff,
                      handoff_candidates=handoff_candidates, handoff_depth=handoff_depth,
                      terminal_no_information=terminal, remaining_candidates=candidates,
                      exact_true_singleton=candidates == [truth], map_calls=len(maps),
                      tree_calls=len(steps), membership_calls=len(queried))
        check('complete_deeper_case_replay', result == stored == ledger.take('deeper_case_result')['data'] and
              len(maps) <= 4 and len(queried) <= 8)
    check('all_deeper_runs_before_any_baseline_query', ledger.take('all_deeper_executions_complete')['data'] == dict(
          cases=4096, paid_calls=ledger.calls['deeper'], deeper_results_sha256=sha(P/'deeper-results.json'),
          groups_sha256=sha(P/'groups.json'), counterfactual_registry_sha256=sha(P/'counterfactual-groups.json')))
    baseline_groups = read(P/'baseline-groups.json')
    original = OriginalAudit(P/'baseline', S, baseline_groups, pred, scores, ids['predictions_sha256'],
                             check, read, sha, canonical)
    for source, stored in zip(sources, old_rows):
        case, truth = source['case'], source['candidate']
        candidates, queried, maps, steps = list(source['initial_candidates']), [], [], []
        terminal = bound = None
        check('baseline_starts_from_source_with_no_inherited_queries', ledger.take('arm_source_bound')['data'] == dict(
              case=case, arm='baseline', candidate=truth, source_record_sha256=source['source_record_sha256'],
              initial_candidates=candidates, inherited_queries=0))
        for _ in range(4):
            if len(candidates) == 1:
                break
            request, selection = request_for(candidates, queried), select(candidates, queried)
            if selection['query_index'] is None:
                event = ledger.take('no_informative_query')
                check('original_no_information', event['data'] == dict(case=case, arm='baseline', request=request, selection=selection))
                terminal = dict(sequence=event['sequence'], sha256=event['sha256'])
                break
            candidates, step = ledger.ask(case, truth, 'baseline', 'map', candidates, queried, request, selection)
            maps.append(step)
            queried = queried+[step['query_index']]
        checkpoint_candidates, checkpoint_queries = list(candidates), list(queried)
        if len(candidates) > 1:
            event = ledger.take('original_baseline_tree_bound')
            row, proof = original.get(candidates, event)
            group = row['group']
            data = dict(case=case, group=group, candidates=candidates, queried=queried,
                        feasible=proof['feasible'], proof_sha256=row['proof_sha256'],
                        group_result_sha256=sha(P/f'baseline/group{group}/result.json'), cache_key=row['cache_key'])
            check('original_actual_tree_binding', event['data'] == data)
            bound = dict(data, sequence=event['sequence'], sha256=event['sha256'])
            if proof['feasible']:
                key = proof['root_state']
                while proof['states'][key]['reason'] != 'singleton_leaf':
                    node = proof['states'][key]
                    index = node['chosen_split']['board_index']
                    request = dict(group=group, proof_sha256=row['proof_sha256'], state=key,
                                   candidates=candidates, queried=queried)
                    selection = dict(query_index=index, candidate_predictions={str(c): bool(pred[c, index]) for c in candidates})
                    candidates, step = ledger.ask(case, truth, 'baseline', 'tree', candidates, queried, request, selection)
                    steps.append(step)
                    queried = queried+[index]
                    key = node['positive_state'] if step['label'] else node['negative_state']
        result = dict(case=case, candidate=truth, source_record_sha256=source['source_record_sha256'],
                      initial_candidates=source['initial_candidates'], map_steps=maps, tree_steps=steps,
                      checkpoint_candidates=checkpoint_candidates, checkpoint_queries=checkpoint_queries,
                      compiled_tree_bound=bound, terminal_no_information=terminal,
                      remaining_candidates=candidates, exact_true_singleton=candidates == [truth],
                      map_calls=len(maps), tree_calls=len(steps), membership_calls=len(queried),
                      independently_paid_cross_arm_reacquisitions=sum(bool(step['previous_same_case_query_sequences']) for step in maps+steps))
        check('complete_separately_paid_original_case', result == stored == ledger.take('baseline_case_result')['data'] and
              len(maps) <= 4 and len(steps) <= 4 and len(queried) <= 8)
    check('original_execution_complete_seal', ledger.take('all_baseline_executions_complete')['data'] == dict(
          cases=4096, paid_calls=ledger.calls['baseline'], baseline_results_sha256=sha(P/'baseline-results.json'),
          baseline_groups_sha256=sha(P/'baseline-groups.json')))
    for new, old, pair in zip(new_rows, old_rows, paired):
        check('paired_all_case_exact_flag_preservation', new['case'] == old['case'] and
              new['candidate'] == old['candidate'] and new['exact_true_singleton'] == old['exact_true_singleton'])
        for i, step in enumerate(new['map_steps']):
            check('common_MAP_prefix_separately_paid', step['query_index'] == old['map_steps'][i]['query_index'] and
                  step['label'] == old['map_steps'][i]['label'] and step['query_sequence'] != old['map_steps'][i]['query_sequence'])
        expected = dict(case=new['case'], candidate=new['candidate'], deeper_calls=new['membership_calls'],
                        baseline_calls=old['membership_calls'], deeper_exact=new['exact_true_singleton'],
                        baseline_exact=old['exact_true_singleton'], all_actual_calls=new['membership_calls']+old['membership_calls'],
                        independently_paid_cross_arm_reacquisitions=old['independently_paid_cross_arm_reacquisitions'])
        check('whole_paired_case_record', pair == expected == ledger.take('paired_case_result')['data'])
    ns, nt = trees.finish()
    nc, np_ = costs.finish()
    bs, bt = original.finish()
    D, B = ledger.calls['deeper'], ledger.calls['baseline']
    check('all_actual_paid_calls_in_both_arms', D == sum(row['deeper_calls'] for row in paired) <= 32768 and
          B == sum(row['baseline_calls'] for row in paired) <= 32768 and D+B <= 65536)
    decision = 'inconclusive_zero_comparator' if B == 0 else 'supported' if 100*D <= 98*B else 'falsified'
    outcome = dict(decision=decision, primary=dict(left=100*D, right=98*B,
                   passing_maximum=(98*B)//100 if B else None, threshold_percent=2), cases=4096,
                   deeper_policy_calls=D, baseline_policy_calls=B, actual_membership_calls=D+B,
                   MAP_calls={arm: sum(r['map_calls'] for r in rows) for arm, rows in [('deeper', new_rows), ('baseline', old_rows)]},
                   tree_calls={arm: sum(r['tree_calls'] for r in rows) for arm, rows in [('deeper', new_rows), ('baseline', old_rows)]},
                   exact_cases=sum(r['deeper_exact'] for r in paired), retained_failures=sum(not r['deeper_exact'] for r in paired),
                   all4096_exact_flags_identical_correctness_invariant=True, new_truth_draws=4096, new_source_draws=4096,
                   source_positive_union_appearances=32768, positive_arm_appearances=65536,
                   distinct_N12_boards=source_audit['distinct_N12_boards'],
                   source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),
                   source_cases_matching_archived_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),
                   independently_paid_cross_arm_reacquisitions=sum(r['independently_paid_cross_arm_reacquisitions'] for r in paired),
                   truth_frequencies=[dict(candidate=r['candidate'], eligible=r['eligible'], sampling_integer_weight=r['sampling_integer_weight'],
                                      cases=source_audit['truth_counts'].get(r['candidate'], 0)) for r in p['truths']],
                   handoff_MAP_call_histogram={str(i): sum(r['handoff_MAP_calls'] == i for r in new_rows) for i in range(5)},
                   cases_without_tree_handoff=sum(r['handoff_MAP_calls'] is None for r in new_rows),
                   distinct_compiled_inputs=len(groups), baseline_compiled_inputs=len(baseline_groups),
                   tree_memo_states=ns, compiler_work_steps=nt,
                   huffman_merge_operations=sum(r['huffman_merge_operations'] for r in groups),
                   canonical_computation_reuse_hits=trees.reuse_hits, baseline_memo_states=bs, baseline_work_steps=bt,
                   counterfactual_memo_states=nc, counterfactual_MAP_partitions=np_,
                   runtime_gate_memo_states=gates['memo_states'], runtime_gate_work_steps=gates['work_steps'],
                   global_memo_states=gates['memo_states']+ns+nc+bs,
                   global_split_evaluations=gates['work_steps']+nt+np_+bt)
    check('sole_fresh_comparator_primary_and_every_counter', ledger.take('study_complete')['data'] == outcome and
          all(g[k] == v for k, v in outcome.items()) and ledger.at == len(events))
    storage = io.finish()
    return dict(status='passed_complete_fresh_two_arm_audit', checks=checks, cases=4096,
                primary_decision=decision, deeper_paid_queries=D, baseline_paid_queries=B,
                actual_paid_queries=D+B, all_fresh_prior_tickets_and_source_indices_verified=True,
                all4096_exact_flags_verified=True, every_failure_and_duplicate_retained=True,
                separate_baseline_queries_and_original_proofs_verified=True,
                all_deeper_and_counterfactual_proofs_verified=True,
                original_baseline_candidate_paths=original.paths, deeper_tree_candidate_paths=trees.paths,
                counterfactual_candidate_entries=costs.candidate_paths,
                charged_memo_states=outcome['global_memo_states'], charged_work_steps=outcome['global_split_evaluations'],
                runtime_gate_audit=gates, storage=storage, elapsed_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--archive-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.archive_dir)))
