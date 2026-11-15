"""Fresh paired policy execution; every actual label is separately paid."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT, SRC, OUT, frozen, read, save, sha, now
from compiler import Budget
from ledger import Ledger
from ledger_gate import run as ledger_gate
from synthetic_gate import run as synthetic_gate
from restriction import gate as restriction_gate
from pruning_gate import run as pruning_gate
from cache_gate import run as cache_gate
from cost_gate import run as cost_gate
from baseline_gate import run as baseline_gate
from wrapper_gate import run as wrapper_gate
from source_gate import run as source_gate
from prepare import run as prepare
from prior import draw_truth
from policy import OnlineCompiler
from counterfactual import BaselineCost, guard
from baseline import OriginalBaseline


def cpu():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime


def bank_hash(sample):
    return hashlib.sha256(b''.join(sorted(map(bytes, sample)))).hexdigest()


def main():
    start, wall = cpu(), time.perf_counter()
    p = frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'study-result.json').exists()
    budget = Budget(p['budget']['max_memo_states_total'], p['budget']['max_split_evaluations_total'])
    worker = compiler = cost_model = baseline = ledger = None
    sources, deeper, originals, results = [], [], [], []
    calls = 0
    arm_calls = dict(deeper=0, baseline=0)
    statuses = [dict(case=i, source='pending', deeper='pending', baseline='pending') for i in range(4096)]
    def expired(*_):
        raise TimeoutError('H3bx 600 CPU / 900 wall gate; whole study incomplete')
    signal.signal(signal.SIGUSR1, expired)
    try:
        ordinary, sourcepred, sourcepop, boards, bits, scores, history = prepare(p)
        sg = source_gate(OUT/'source-gates', p['historical_protocol_sha256'])
        assert sg['prior_rows'] == p['truths']
        assert sg['seed_audit']['historical_protocol_sha256'] == p['historical_protocol_sha256']
        worker = subprocess.Popen([sys.executable, str(SRC/'selector.py')], cwd=OUT,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        def select(request):
            worker.stdin.write(json.dumps(request)+'\n')
            worker.stdin.flush()
            line = worker.stdout.readline()
            assert line, 'Persistent selector exited'
            return json.loads(line)
        ledger_gate(p, select)
        toy = synthetic_gate(budget)
        save(OUT/'restriction-gate.json', dict(utc=now(), **restriction_gate(toy['fixtures'])))
        save(OUT/'pruning-gate.json', pruning_gate(OUT/'pruning-gate', budget))
        save(OUT/'cache-gate.json', cache_gate(OUT/'cache-gate', budget))
        save(OUT/'cost-gate.json', cost_gate(OUT/'cost-gate', budget))
        save(OUT/'baseline-gate.json', baseline_gate(OUT/'baseline-gate', budget))
        save(OUT/'wrapper-gate.json', wrapper_gate(OUT/'wrapper-gate', budget))
        gate_states, gate_work = budget.states, budget.splits
        prep = dict(utc=now(), protocol_sha256=sha(SRC/'protocol.json'),
                    runtime_gate_memo_states=gate_states, runtime_gate_work_steps=gate_work,
                    preflight_sha256=sha(SRC/'preflight.json'),
                    source_gate=sg['seed_audit'], source_prior_gate_sha256=sg['prior_gate_sha256'],
                    all_runtime_gates_before_prospective_draws=True)
        for name in ['population-parity', 'ledger-gate', 'synthetic-gate', 'restriction-gate',
                     'pruning-gate', 'cache-gate', 'cost-gate', 'baseline-gate', 'wrapper-gate']:
            prep[name+'_sha256'] = sha(OUT/(name+'.json'))
        save(OUT/'preparation.json', prep)
        ledger = Ledger(OUT/'query-ledger.jsonl')
        ledger.append('study_start', dict(protocol_sha256=sha(SRC/'protocol.json'),
                      preparation_sha256=sha(OUT/'preparation.json')))
        predictions = np.unpackbits(bits, axis=1, count=len(boards), bitorder='little').astype(bool)
        lookup = {bytes(row): i for i, row in enumerate(ordinary)}
        fresh, truth_counts, global_boards = {}, {}, set()
        for case in range(4096):
            truth_seed = p['seed_bases']['truth']+case
            source_seed = p['seed_bases']['source']+case
            truth, ticket = draw_truth(p['truths'], truth_seed)
            population = sourcepop[truth]
            bank = truth_counts.get(truth, 0)
            truth_counts[truth] = bank+1
            indices = np.random.default_rng(source_seed).choice(len(population), 8, replace=False)
            sample = population[indices]
            initial = np.flatnonzero(sourcepred[:, [lookup[bytes(row)] for row in sample]].all(1)).tolist()
            digest = bank_hash(sample)
            assert truth in initial and all(scores[c] is not None for c in initial)
            data = dict(case=case, candidate=truth, bank=bank, truth_seed=truth_seed,
                        truth_ticket=ticket, seed=source_seed, source_population=len(population),
                        indices=indices.tolist(), columns_zero_based=sample.tolist(),
                        initial_candidates=initial, unordered_sample_sha256=digest,
                        prior_same_study_cases=list(fresh.get(digest, [])),
                        historical_exact8_matches=history.get(digest, []))
            event = ledger.append('source_bank', data)
            sources.append(dict(data, source_sequence=event['sequence'], source_record_sha256=event['sha256']))
            fresh.setdefault(digest, []).append(case)
            global_boards.update(map(bytes, sample))
            statuses[case]['source'] = 'complete'
        save(OUT/'source-banks.json', sources)
        ledger.append('all_source_banks_drawn', dict(cases=4096, new_truth_draws=4096,
                      new_source_draws=4096, source_positive_union_appearances=32768,
                      positive_arm_appearances=65536, distinct_N12_boards=len(global_boards),
                      source_banks_sha256=sha(OUT/'source-banks.json')))
        call_history = {}
        def ask(case, truth, arm, phase, step, candidates, queried, request, selection):
            nonlocal calls
            index = selection['query_index']
            assert index is not None and index not in queried and step < 8
            assert calls < 65536 and arm_calls[arm] < 32768
            previous = list(call_history.get((case, index), []))
            query = ledger.append('query', dict(case=case, arm=arm, phase=phase, step=step,
                                  request=request, selection=selection,
                                  board_zero_based=boards[index].tolist(),
                                  previous_same_case_query_sequences=previous))
            label = bool((bits[truth, index//8] >> (index % 8)) & 1)
            calls += 1
            arm_calls[arm] += 1
            remaining = [c for c in candidates if selection['candidate_predictions'][str(c)] == label]
            answer = ledger.append('answer', dict(case=case, arm=arm, phase=phase, step=step,
                                   query_sha256=query['sha256'], label=label,
                                   candidates_before=candidates, candidates_after=remaining))
            assert truth in remaining
            call_history.setdefault((case, index), []).append(query['sequence'])
            return remaining, dict(step=step, query_index=index, label=label,
                                   candidates_before=candidates, candidates_after=remaining,
                                   query_sequence=query['sequence'], query_sha256=query['sha256'],
                                   answer_sequence=answer['sequence'], answer_sha256=answer['sha256'],
                                   previous_same_case_query_sequences=previous)
        def request_for(candidates, queried):
            return dict(bits_path='predictions-bits.npy', scores_path='score-values8.json',
                        board_count=len(boards), candidates=candidates, queried=queried)
        compiler = OnlineCompiler(predictions, scores, sha(OUT/'predictions-bits.npy'),
                                  budget, p['budget']['max_integer_bits'])
        identities = dict(predictions_sha256=sha(OUT/'predictions-bits.npy'),
                          scores_sha256=sha(OUT/'score-values8.json'),
                          selector_sha256=sha(SRC/'selector.py'),
                          tree_compiler_sha256=sha(SRC/'compiler.py'),
                          tree_wrapper_sha256=sha(SRC/'policy.py'))
        cost_model = BaselineCost(compiler, select, predictions, budget, identities)
        for source in sources:
            case, truth = source['case'], source['candidate']
            candidates, queried = list(source['initial_candidates']), []
            maps, trees, routes, guards = [], [], [], []
            handoff = handoff_candidates = handoff_depth = terminal = None
            ledger.append('arm_source_bound', dict(case=case, arm='deeper', candidate=truth,
                          source_record_sha256=source['source_record_sha256'],
                          initial_candidates=candidates, inherited_queries=0))
            while len(candidates) > 1:
                depth, row, proof, decision = 8-len(maps), None, None, None
                if len(candidates) > 16:
                    data = dict(case=case, MAP_calls=len(maps), requested_depth=depth,
                                candidates=candidates, queried=queried,
                                reason='candidate_count_eligibility_limit', limit=16,
                                not_an_infeasibility_certificate=True)
                    event = ledger.append('handoff_ineligible_candidate_count', data)
                    routes.append(dict(data, bound_sequence=event['sequence'], bound_sha256=event['sha256']))
                else:
                    row, proof = compiler.get(candidates, depth)
                    group = row['group']
                    assert proof['candidates'] == candidates and proof['depth_limit'] == depth
                    assert sha(OUT/f'group{group}/proof.json') == row['proof_sha256']
                    data = dict(case=case, MAP_calls=len(maps), requested_depth=depth,
                                candidates=candidates, queried=queried, group=group,
                                feasible=proof['feasible'], proof_sha256=row['proof_sha256'],
                                group_result_sha256=sha(OUT/f'group{group}/result.json'), cache_key=row['cache_key'])
                    event = ledger.append('online_compiled_tree_bound', data)
                    routes.append(dict(data, bound_sequence=event['sequence'], bound_sha256=event['sha256']))
                if proof is not None and proof['feasible']:
                    counterfactual = cost_model.evaluate(candidates, queried, 4-len(maps))
                    decision = guard(proof, counterfactual)
                    if decision['baseline_admissible']:
                        assert decision['cost_comparison_passes']
                    if len(maps) == 4:
                        assert decision['handoff'] and decision['tie']
                    data = dict(case=case, MAP_calls=len(maps), requested_depth=depth,
                                tree_group=group, baseline_cost_group=counterfactual['group'],
                                baseline_result_sha256=sha(OUT/f'cost{counterfactual["group"]}/result.json'),
                                baseline_cache_key=counterfactual['cache_key'], **decision)
                    event = ledger.append('posterior_cost_guard_bound', data)
                    guards.append(dict(data, sequence=event['sequence'], sha256=event['sha256']))
                if proof is not None and proof['feasible'] and decision['handoff']:
                    handoff, handoff_candidates, handoff_depth = len(maps), list(candidates), depth
                    key = proof['root_state']
                    while proof['states'][key]['reason'] != 'singleton_leaf':
                        node = proof['states'][key]
                        index = node['chosen_split']['board_index']
                        request = dict(group=group, proof_sha256=row['proof_sha256'], state=key,
                                       candidates=candidates, queried=queried)
                        selection = dict(query_index=index, candidate_predictions={
                                         str(c): bool(predictions[c, index]) for c in candidates})
                        candidates, step = ask(case, truth, 'deeper', 'tree', len(queried),
                                               candidates, queried, request, selection)
                        trees.append(step)
                        queried = queried+[index]
                        key = node['positive_state'] if step['label'] else node['negative_state']
                    break
                if len(maps) == 4:
                    break
                request = request_for(candidates, queried)
                selection = select(request)
                if selection['query_index'] is None:
                    event = ledger.append('no_informative_query', dict(case=case, arm='deeper',
                                          request=request, selection=selection))
                    terminal = dict(sequence=event['sequence'], sha256=event['sha256'])
                    break
                candidates, step = ask(case, truth, 'deeper', 'map', len(queried),
                                       candidates, queried, request, selection)
                maps.append(step)
                queried = queried+[step['query_index']]
            result = dict(case=case, candidate=truth, source_record_sha256=source['source_record_sha256'],
                          initial_candidates=source['initial_candidates'], routing=routes, cost_guards=guards,
                          map_steps=maps, tree_steps=trees, handoff_MAP_calls=handoff,
                          handoff_candidates=handoff_candidates, handoff_depth=handoff_depth,
                          terminal_no_information=terminal, remaining_candidates=candidates,
                          exact_true_singleton=candidates == [truth], map_calls=len(maps),
                          tree_calls=len(trees), membership_calls=len(queried))
            assert len(maps) <= 4 and len(trees) <= 8-len(maps) and len(queried) <= 8
            ledger.append('deeper_case_result', result)
            deeper.append(result)
            statuses[case]['deeper'] = 'complete'
            if case % 256 == 255:
                print('DEEPER', case+1, 'calls', calls, 'states', budget.states, 'work', budget.splits, flush=True)
        save(OUT/'deeper-results.json', deeper)
        save(OUT/'groups.json', compiler.rows)
        assert all(row is not None for row in cost_model.rows)
        save(OUT/'counterfactual-groups.json', [dict(group=row['group'], cache_key=row['cache_key'],
             input_sha256=row['input_sha256'], result_sha256=sha(OUT/f'cost{row["group"]}/result.json'))
             for row in cost_model.rows])
        ledger.append('all_deeper_executions_complete', dict(cases=4096, paid_calls=arm_calls['deeper'],
                      deeper_results_sha256=sha(OUT/'deeper-results.json'), groups_sha256=sha(OUT/'groups.json'),
                      counterfactual_registry_sha256=sha(OUT/'counterfactual-groups.json')))
        baseline = OriginalBaseline(OUT/'baseline', predictions, scores,
                                    sha(OUT/'predictions-bits.npy'), budget, p['budget']['max_integer_bits'])
        for source in sources:
            case, truth = source['case'], source['candidate']
            candidates, queried, maps, trees = list(source['initial_candidates']), [], [], []
            terminal = bound = None
            ledger.append('arm_source_bound', dict(case=case, arm='baseline', candidate=truth,
                          source_record_sha256=source['source_record_sha256'],
                          initial_candidates=candidates, inherited_queries=0))
            for _ in range(4):
                if len(candidates) == 1:
                    break
                request = request_for(candidates, queried)
                selection = select(request)
                if selection['query_index'] is None:
                    event = ledger.append('no_informative_query', dict(case=case, arm='baseline',
                                          request=request, selection=selection))
                    terminal = dict(sequence=event['sequence'], sha256=event['sha256'])
                    break
                candidates, step = ask(case, truth, 'baseline', 'map', len(queried),
                                       candidates, queried, request, selection)
                maps.append(step)
                queried = queried+[step['query_index']]
            checkpoint_candidates, checkpoint_queries = list(candidates), list(queried)
            if len(candidates) > 1:
                row, proof = baseline.get(candidates)
                group = row['group']
                assert sha(OUT/f'baseline/group{group}/posterior/proof.json') == row['proof_sha256']
                data = dict(case=case, group=group, candidates=candidates, queried=queried,
                            feasible=proof['feasible'], proof_sha256=row['proof_sha256'],
                            group_result_sha256=sha(OUT/f'baseline/group{group}/result.json'),
                            cache_key=row['cache_key'])
                event = ledger.append('original_baseline_tree_bound', data)
                bound = dict(data, sequence=event['sequence'], sha256=event['sha256'])
                if proof['feasible']:
                    key = proof['root_state']
                    while proof['states'][key]['reason'] != 'singleton_leaf':
                        node = proof['states'][key]
                        index = node['chosen_split']['board_index']
                        request = dict(group=group, proof_sha256=row['proof_sha256'], state=key,
                                       candidates=candidates, queried=queried)
                        selection = dict(query_index=index, candidate_predictions={
                                         str(c): bool(predictions[c, index]) for c in candidates})
                        candidates, step = ask(case, truth, 'baseline', 'tree', len(queried),
                                               candidates, queried, request, selection)
                        trees.append(step)
                        queried = queried+[index]
                        key = node['positive_state'] if step['label'] else node['negative_state']
            result = dict(case=case, candidate=truth, source_record_sha256=source['source_record_sha256'],
                          initial_candidates=source['initial_candidates'], map_steps=maps, tree_steps=trees,
                          checkpoint_candidates=checkpoint_candidates, checkpoint_queries=checkpoint_queries,
                          compiled_tree_bound=bound, terminal_no_information=terminal,
                          remaining_candidates=candidates, exact_true_singleton=candidates == [truth],
                          map_calls=len(maps), tree_calls=len(trees), membership_calls=len(queried),
                          independently_paid_cross_arm_reacquisitions=sum(bool(s['previous_same_case_query_sequences'])
                                                                          for s in maps+trees))
            assert len(maps) <= 4 and len(trees) <= 4 and len(queried) <= 8
            ledger.append('baseline_case_result', result)
            originals.append(result)
            statuses[case]['baseline'] = 'complete'
            if case % 256 == 255:
                print('BASELINE', case+1, 'calls', calls, 'states', budget.states, 'work', budget.splits, flush=True)
        worker.stdin.close()
        assert worker.wait() == 0
        worker = None
        save(OUT/'baseline-results.json', originals)
        save(OUT/'baseline-groups.json', baseline.rows)
        ledger.append('all_baseline_executions_complete', dict(cases=4096, paid_calls=arm_calls['baseline'],
                      baseline_results_sha256=sha(OUT/'baseline-results.json'),
                      baseline_groups_sha256=sha(OUT/'baseline-groups.json')))
        for new, old in zip(deeper, originals):
            assert new['case'] == old['case'] and new['candidate'] == old['candidate']
            assert new['exact_true_singleton'] == old['exact_true_singleton']
            for i, step in enumerate(new['map_steps']):
                assert step['query_index'] == old['map_steps'][i]['query_index']
                assert step['label'] == old['map_steps'][i]['label']
            pair = dict(case=new['case'], candidate=new['candidate'],
                        deeper_calls=new['membership_calls'], baseline_calls=old['membership_calls'],
                        deeper_exact=new['exact_true_singleton'], baseline_exact=old['exact_true_singleton'],
                        all_actual_calls=new['membership_calls']+old['membership_calls'],
                        independently_paid_cross_arm_reacquisitions=old['independently_paid_cross_arm_reacquisitions'])
            ledger.append('paired_case_result', pair)
            results.append(pair)
        D, B = arm_calls['deeper'], arm_calls['baseline']
        assert D == sum(r['deeper_calls'] for r in results)
        assert B == sum(r['baseline_calls'] for r in results) and calls == D+B
        decision = 'inconclusive_zero_comparator' if B == 0 else 'supported' if 100*D <= 98*B else 'falsified'
        outcome = dict(decision=decision, primary=dict(left=100*D, right=98*B,
                       passing_maximum=(98*B)//100 if B else None, threshold_percent=2),
                       cases=4096, deeper_policy_calls=D, baseline_policy_calls=B,
                       actual_membership_calls=calls,
                       MAP_calls={arm: sum(r['map_calls'] for r in rows)
                                  for arm, rows in [('deeper', deeper), ('baseline', originals)]},
                       tree_calls={arm: sum(r['tree_calls'] for r in rows)
                                   for arm, rows in [('deeper', deeper), ('baseline', originals)]},
                       exact_cases=sum(r['deeper_exact'] for r in results),
                       retained_failures=sum(not r['deeper_exact'] for r in results),
                       all4096_exact_flags_identical_correctness_invariant=True,
                       new_truth_draws=4096, new_source_draws=4096,
                       source_positive_union_appearances=32768, positive_arm_appearances=65536,
                       distinct_N12_boards=len(global_boards),
                       source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),
                       source_cases_matching_archived_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),
                       independently_paid_cross_arm_reacquisitions=sum(r['independently_paid_cross_arm_reacquisitions'] for r in results),
                       truth_frequencies=[dict(candidate=r['candidate'], eligible=r['eligible'],
                                          prior_integer_weight=r['prior_integer_weight'],
                                          cases=truth_counts.get(r['candidate'], 0)) for r in p['truths']],
                       handoff_MAP_call_histogram={str(i): sum(r['handoff_MAP_calls'] == i for r in deeper) for i in range(5)},
                       cases_without_tree_handoff=sum(r['handoff_MAP_calls'] is None for r in deeper),
                       distinct_compiled_inputs=len(compiler.rows), baseline_compiled_inputs=len(baseline.rows),
                       tree_memo_states=sum(r['memo_states'] for r in compiler.rows),
                       compiler_work_steps=sum(r['work_steps'] for r in compiler.rows),
                       huffman_merge_operations=sum(r['huffman_merge_operations'] for r in compiler.rows),
                       canonical_computation_reuse_hits=sum(r['canonical_computation_reuse'] is not None for r in compiler.rows),
                       baseline_memo_states=sum(r['memo_states'] for r in baseline.rows),
                       baseline_work_steps=sum(r['work_steps'] for r in baseline.rows),
                       counterfactual_memo_states=cost_model.memo_states,
                       counterfactual_MAP_partitions=cost_model.MAP_partitions,
                       runtime_gate_memo_states=gate_states, runtime_gate_work_steps=gate_work,
                       global_memo_states=budget.states, global_split_evaluations=budget.splits)
        assert budget.states == gate_states+outcome['tree_memo_states']+cost_model.memo_states+outcome['baseline_memo_states']
        assert budget.splits == gate_work+outcome['compiler_work_steps']+cost_model.MAP_partitions+outcome['baseline_work_steps']
        save(OUT/'case-results.json', results)
        save(OUT/'case-statuses.json', statuses)
        ledger.append('study_complete', outcome)
        ledger.close()
        seals = {name.replace('-', '_')+'_sha256': sha(OUT/(name+'.json')) for name in
                 ['case-results', 'case-statuses', 'source-banks', 'deeper-results', 'baseline-results',
                  'groups', 'baseline-groups', 'counterfactual-groups', 'preparation']}
        save(OUT/'study-result.json', dict(status='complete_verified', utc=now(), **outcome, **seals,
             ledger_sha256=sha(OUT/'query-ledger.jsonl'), ledger_records=ledger.sequence,
             ledger_last_record_sha256=ledger.previous, stage_cpu_seconds=cpu()-start,
             stage_wall_seconds=time.perf_counter()-wall,
             protocol_sha256=sha(SRC/'protocol.json'), source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:
            worker.terminate()
            worker.wait()
        if ledger is not None:
            ledger.close()
        save(OUT/'case-statuses.json', statuses)
        save(OUT/'study-result.json', dict(status='failed_or_incomplete', utc=now(), error=repr(exc),
             completed_sources=sources, completed_deeper_results=deeper,
             completed_baseline_results=originals, completed_pairs=results,
             completed_groups=compiler.rows if compiler else [],
             baseline_groups=baseline.rows if baseline else [],
             counterfactual_groups=cost_model.rows if cost_model else [],
             actual_membership_calls=calls, arm_calls=arm_calls,
             global_memo_states=budget.states, global_split_evaluations=budget.splits,
             partial_results_usable=False, cap_is_not_infeasibility=True,
             stage_cpu_seconds=cpu()-start, stage_wall_seconds=time.perf_counter()-wall,
             protocol_sha256=sha(SRC/'protocol.json')))
        raise


if __name__ == '__main__':
    main()
