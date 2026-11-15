"""One fixed deeper policy with an isolated score/selector/proof namespace."""
import json
import subprocess
import sys
from common import SRC, save, sha
import policy
import counterfactual as counterfactual_module
from policy import OnlineCompiler
from counterfactual import BaselineCost, guard


class ArmRunner:
    def __init__(self, arm, folder, predictions, scores, budget, max_bits, board_count):
        assert arm in ['H','U']
        self.arm, self.folder, self.predictions, self.scores = arm, folder, predictions, scores
        self.budget, self.max_bits, self.board_count = budget, max_bits, board_count
        self.results, self.compiler, self.cost_model, self.worker = [], None, None, None

    def select(self, request):
        self.worker.stdin.write(json.dumps(request)+'\n'); self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        assert line, 'Independent arm selector exited'
        return json.loads(line)

    def run(self, sources, ledger, ask, statuses):
        arm, folder, predictions, scores, budget = self.arm, self.folder, self.predictions, self.scores, self.budget
        previous_policy, previous_cost = policy.OUT, counterfactual_module.OUT
        assert not list(folder.glob('group*')) and not list(folder.glob('cost*'))
        self.worker = subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=folder,
                                       stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        policy.OUT, counterfactual_module.OUT = folder, folder
        try:
            select = self.select
            def request_for(candidates, queried):
                return dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',
                            board_count=self.board_count,candidates=candidates,queried=queried)
            compiler = self.compiler = OnlineCompiler(predictions,scores,sha(folder/'predictions-bits.npy'),
                                                     budget,self.max_bits)
            identities = dict(predictions_sha256=sha(folder/'predictions-bits.npy'),scores_sha256=sha(folder/'score-values8.json'),
                              selector_sha256=sha(SRC/'selector.py'),tree_compiler_sha256=sha(SRC/'compiler.py'),
                              tree_wrapper_sha256=sha(SRC/'policy.py'))
            cost_model = self.cost_model = BaselineCost(compiler,select,predictions,budget,identities)
            ledger.append('arm_namespace_bound',dict(arm=arm,relative_folder=arm,score_sha256=identities['scores_sha256'],
                          initial_tree_cache_entries=0,initial_counterfactual_cache_entries=0,selector_process_fresh=True))
            for source in sources:
                case, truth = source['case'], source['candidate']
                candidates, queried = list(source['initial_candidates']), []
                maps, trees, routes, guards = [], [], [], []
                handoff = handoff_candidates = handoff_depth = terminal = None
                ledger.append('arm_source_bound', dict(case=case, arm=arm, candidate=truth,
                              source_record_sha256=source['source_record_sha256'],
                              initial_candidates=candidates, inherited_queries=0))
                while len(candidates) > 1:
                    depth, row, proof, decision = 8-len(maps), None, None, None
                    if len(candidates) > 16:
                        data = dict(arm=arm, case=case, MAP_calls=len(maps), requested_depth=depth,
                                    candidates=candidates, queried=queried,
                                    reason='candidate_count_eligibility_limit', limit=16,
                                    not_an_infeasibility_certificate=True)
                        event = ledger.append('handoff_ineligible_candidate_count', data)
                        routes.append(dict(data, bound_sequence=event['sequence'], bound_sha256=event['sha256']))
                    else:
                        row, proof = compiler.get(candidates, depth)
                        group = row['group']
                        assert proof['candidates'] == candidates and proof['depth_limit'] == depth
                        assert sha(folder/f'group{group}/proof.json') == row['proof_sha256']
                        data = dict(arm=arm, case=case, MAP_calls=len(maps), requested_depth=depth,
                                    candidates=candidates, queried=queried, group=group,
                                    feasible=proof['feasible'], proof_sha256=row['proof_sha256'],
                                    group_result_sha256=sha(folder/f'group{group}/result.json'), cache_key=row['cache_key'])
                        event = ledger.append('online_compiled_tree_bound', data)
                        routes.append(dict(data, bound_sequence=event['sequence'], bound_sha256=event['sha256']))
                    if proof is not None and proof['feasible']:
                        counterfactual = cost_model.evaluate(candidates, queried, 4-len(maps))
                        decision = guard(proof, counterfactual)
                        if decision['baseline_admissible']:
                            assert decision['cost_comparison_passes']
                        if len(maps) == 4:
                            assert decision['handoff'] and decision['tie']
                        data = dict(arm=arm, case=case, MAP_calls=len(maps), requested_depth=depth,
                                    tree_group=group, baseline_cost_group=counterfactual['group'],
                                    baseline_result_sha256=sha(folder/f'cost{counterfactual["group"]}/result.json'),
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
                            candidates, step = ask(case, truth, arm, 'tree', len(queried),
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
                        event = ledger.append('no_informative_query', dict(case=case, arm=arm,
                                              request=request, selection=selection))
                        terminal = dict(sequence=event['sequence'], sha256=event['sha256'])
                        break
                    candidates, step = ask(case, truth, arm, 'map', len(queried),
                                           candidates, queried, request, selection)
                    maps.append(step)
                    queried = queried+[step['query_index']]
                result = dict(arm=arm, case=case, candidate=truth, source_record_sha256=source['source_record_sha256'],
                              initial_candidates=source['initial_candidates'], routing=routes, cost_guards=guards,
                              map_steps=maps, tree_steps=trees, handoff_MAP_calls=handoff,
                              handoff_candidates=handoff_candidates, handoff_depth=handoff_depth,
                              terminal_no_information=terminal, remaining_candidates=candidates,
                              exact_true_singleton=candidates == [truth], map_calls=len(maps),
                              tree_calls=len(trees), membership_calls=len(queried),
                              independently_paid_cross_arm_reacquisitions=sum(bool(s['previous_same_case_query_sequences']) for s in maps+trees))
                assert len(maps) <= 4 and len(trees) <= 8-len(maps) and len(queried) <= 8
                ledger.append('arm_case_result', result)
                self.results.append(result)
                statuses[case][arm] = 'complete'
                if case % 256 == 255:
                    print(arm, case+1, 'arm_calls', sum(r['membership_calls'] for r in self.results), 'states', budget.states, 'work', budget.splits, flush=True)
            save(folder/'arm-results.json', self.results)
            save(folder/'groups.json', compiler.rows)
            assert all(row is not None for row in cost_model.rows)
            save(folder/'counterfactual-groups.json', [dict(group=row['group'], cache_key=row['cache_key'],
                 input_sha256=row['input_sha256'], result_sha256=sha(folder/f'cost{row["group"]}/result.json'))
                 for row in cost_model.rows])
            ledger.append('all_arm_executions_complete', dict(arm=arm, cases=len(sources), paid_calls=sum(r['membership_calls'] for r in self.results),
                          arm_results_sha256=sha(folder/'arm-results.json'), groups_sha256=sha(folder/'groups.json'),
                          counterfactual_registry_sha256=sha(folder/'counterfactual-groups.json')))
            self.worker.stdin.close()
            assert self.worker.wait() == 0
            self.worker = None
            return self.results
        finally:
            if self.worker is not None:
                self.worker.terminate(); self.worker.wait(); self.worker = None
            policy.OUT, counterfactual_module.OUT = previous_policy, previous_cost

    def snapshot(self):
        return dict(completed_results=self.results,completed_groups=self.compiler.rows if self.compiler else [],
                    counterfactual_groups=self.cost_model.rows if self.cost_model else [])
