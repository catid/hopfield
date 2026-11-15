"""Independent replay of one scored deeper policy and all its certificates."""
import json
from audit_trees import TreeAudit
from audit_costs import GraphAudit, choose, exact_guard


def verify(arm, P, S, sources, rows, pred, boards, selector_scores, tree_scores, ledger, io, check):
    read, sha = io.read, io.sha
    canonical = lambda v: json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    score_ids = dict(selector_scores_path='selector-scores.json',tree_scores_path='tree-scores.json',
                     selector_scores_sha256=sha(P/'selector-scores.json'),tree_scores_sha256=sha(P/'tree-scores.json'))
    check('arm_selector_score_contents',read(P/'selector-scores.json')==list(selector_scores))
    check('arm_tree_score_contents',read(P/'tree-scores.json')==list(tree_scores))
    ids = dict(predictions_sha256=sha(P/'predictions-bits.npy'),**score_ids,
               selector_sha256=sha(S/'selector.py'),tree_compiler_sha256=sha(S/'compiler.py'),
               tree_wrapper_sha256=sha(S/'policy.py'))
    check('arm_namespace_identity',ledger.take('arm_namespace_bound')['data'] == dict(
        arm=arm,relative_folder=arm,**score_ids,initial_tree_cache_entries=0,
        initial_counterfactual_cache_entries=0,selector_process_fresh=True))
    check('complete_arm_row_count_and_order', len(rows)==len(sources) and
          [r['case'] for r in rows]==list(range(len(sources))) and all(r['arm']==arm for r in rows))
    groups = read(P/'groups.json')
    trees = TreeAudit(P,S,groups,pred,tree_scores,ids['predictions_sha256'],check,read,sha,canonical,identities=score_ids)
    cache = {}
    def select(candidates,queried):
        key=tuple(candidates),tuple(queried)
        if key not in cache:cache[key]=choose(pred,selector_scores,candidates,queried)
        return cache[key]
    def request_for(candidates,queried):
        return dict(bits_path='predictions-bits.npy',scores_path='selector-scores.json',
                    board_count=len(boards),candidates=candidates,queried=queried)
    costs = GraphAudit(P,read(P/'counterfactual-groups.json'),pred,select,trees.get,
                      ids,check,read,sha,canonical,io.logical_cost_groups)
    for source, stored in zip(sources, rows):
        case, truth = source['case'], source['candidate']
        candidates, queried = list(source['initial_candidates']), []
        maps, steps, routes, guards = [], [], [], []
        handoff = handoff_candidates = handoff_depth = terminal = None
        check('deeper_initial_full_source_binding', ledger.take('arm_source_bound')['data'] == dict(
              case=case, arm=arm, candidate=truth, source_record_sha256=source['source_record_sha256'],
              initial_candidates=candidates, inherited_queries=0))
        while len(candidates) > 1:
            depth, row, proof, decision = 8-len(maps), None, None, None
            if len(candidates) > 16:
                data = dict(arm=arm, case=case, MAP_calls=len(maps), requested_depth=depth, candidates=candidates,
                            queried=queried, reason='candidate_count_eligibility_limit', limit=16,
                            not_an_infeasibility_certificate=True)
                event = ledger.take('handoff_ineligible_candidate_count')
            else:
                event = ledger.take('online_compiled_tree_bound')
                row, proof = trees.get(candidates, event, depth)
                group = row['group']
                data = dict(arm=arm, case=case, MAP_calls=len(maps), requested_depth=depth, candidates=candidates,
                            queried=queried, group=group, feasible=proof['feasible'], proof_sha256=row['proof_sha256'],
                            group_result_sha256=sha(P/f'group{group}/result.json'), cache_key=row['cache_key'])
            check('deeper_routing_receipt_and_input', event['data'] == data)
            routes.append(dict(data, bound_sequence=event['sequence'], bound_sha256=event['sha256']))
            if proof is not None and proof['feasible']:
                event = ledger.take('posterior_cost_guard_bound')
                baseline_cost = costs.verify(event['data']['baseline_cost_group'], candidates, queried, 4-len(maps), event)
                decision = exact_guard(proof, baseline_cost)
                data = dict(arm=arm, case=case, MAP_calls=len(maps), requested_depth=depth, tree_group=group,
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
                    candidates, step = ledger.ask(case, truth, arm, 'tree', candidates, queried, request, selection)
                    steps.append(step)
                    queried = queried+[index]
                    key = node['positive_state'] if step['label'] else node['negative_state']
                break
            if len(maps) == 4:
                break
            request, selection = request_for(candidates, queried), select(candidates, queried)
            if selection['query_index'] is None:
                event = ledger.take('no_informative_query')
                check('deeper_no_information', event['data'] == dict(case=case, arm=arm, request=request, selection=selection))
                terminal = dict(sequence=event['sequence'], sha256=event['sha256'])
                break
            candidates, step = ledger.ask(case, truth, arm, 'map', candidates, queried, request, selection)
            maps.append(step)
            queried = queried+[step['query_index']]
        result = dict(arm=arm, case=case, candidate=truth, source_record_sha256=source['source_record_sha256'],
                      initial_candidates=source['initial_candidates'], routing=routes, cost_guards=guards,
                      map_steps=maps, tree_steps=steps, handoff_MAP_calls=handoff,
                      handoff_candidates=handoff_candidates, handoff_depth=handoff_depth,
                      terminal_no_information=terminal, remaining_candidates=candidates,
                      exact_true_singleton=candidates == [truth], map_calls=len(maps),
                      tree_calls=len(steps), membership_calls=len(queried),
                  independently_paid_cross_arm_reacquisitions=sum(bool(s['previous_same_case_query_sequences']) for s in maps+steps))
        if arm in ['B','E']:
            from audit_fallback import verify as verify_fallback
            result=verify_fallback(result,pred,ledger,check)
        check('complete_deeper_case_replay', result == stored == ledger.take('arm_case_result')['data'] and
              len(maps) <= 4 and len(queried) <= 8)
    check('all_declared_arm_cases_complete', ledger.take('all_arm_executions_complete')['data'] == dict(
          arm=arm, cases=len(sources), paid_calls=ledger.calls[arm], arm_results_sha256=sha(P/'arm-results.json'),
          groups_sha256=sha(P/'groups.json'), counterfactual_registry_sha256=sha(P/'counterfactual-groups.json')))
    ns, nt = trees.finish()
    nc, np_ = costs.finish()
    return dict(arm=arm,tree_memo_states=ns,compiler_work_steps=nt,counterfactual_memo_states=nc,
                counterfactual_MAP_partitions=np_,distinct_compiled_inputs=len(groups),
                huffman_merge_operations=sum(r['huffman_merge_operations'] for r in groups),
                canonical_computation_reuse_hits=trees.reuse_hits,deeper_tree_candidate_paths=trees.paths,
                counterfactual_candidate_entries=costs.candidate_paths)
