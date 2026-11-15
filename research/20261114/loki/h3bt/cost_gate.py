"""Shared-prefix baseline simulation must equal independent per-candidate rollout."""
import hashlib
import time
import numpy as np
from common import SRC,save,sha,now
from ledger import canonical
from selector import choose
import policy
import counterfactual


def run(root,budget,keep_file_manifest=True):
    root.mkdir(parents=True,exist_ok=True);old_policy_out=policy.OUT;old_cost_out=counterfactual.OUT;before_states=budget.states;before_splits=budget.splits;start=time.perf_counter();rows=[];all_selector_calls=0;all_naive_calls=0
    fixtures=[dict(name='tie_two',signatures=[0,1,2,3],scores=['1','1'],one_MAP_expected_tree_cost=2,one_MAP_expected_baseline_cost=2),dict(name='accept_four',signatures=[0,1,2,4,8,3,12,15],scores=['9','10','10','10'],one_MAP_expected_tree_cost=74,one_MAP_expected_baseline_cost=82),dict(name='reject_six',signatures=[0,1,2,4,8,16,32,3,63],scores=['1','100','100','100','100','100'],one_MAP_expected_tree_cost=215,one_MAP_expected_baseline_cost=119)]
    try:
        for fixture in fixtures:
            folder=root/fixture['name'];folder.mkdir();policy.OUT=folder;counterfactual.OUT=folder;n=len(fixture['scores']);pred=np.array([[bool(s&(1<<c)) for s in fixture['signatures']] for c in range(n)]);packed=np.packbits(pred,axis=1,bitorder='little');calls=[0];naive_calls=0
            def select(request):calls[0]+=1;return choose(request,packed,fixture['scores'])
            identities=dict(predictions_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),scores_sha256=hashlib.sha256(canonical(fixture['scores'])).hexdigest(),selector_sha256=sha(SRC/'selector.py'),tree_compiler_sha256=sha(SRC/'compiler.py'))
            compiler=policy.OnlineCompiler(pred,fixture['scores'],identities['predictions_sha256'],budget,100000);cost=counterfactual.BaselineCost(compiler,select,pred,budget,identities);_,tree=compiler.get(list(range(n)));assert tree['feasible'];runs=[]
            for remaining in [0,1,2,4]:
                baseline=cost.evaluate(list(range(n)),[],remaining);decision=counterfactual.guard(tree,baseline);states=budget.states;splits=budget.splits;selector_count=calls[0];again=cost.evaluate(list(range(n)),[],remaining);assert again==baseline and budget.states==states and budget.splits==splits and calls[0]==selector_count
                naive={}
                for true_candidate in range(n):
                    candidates=list(range(n));queried=[];used=0
                    for _ in range(remaining):
                        if len(candidates)==1:break
                        request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=len(fixture['signatures']),candidates=candidates,queried=queried);selection=choose(request,packed,fixture['scores']);naive_calls+=1;index=selection['query_index'];assert index is not None;label=bool(pred[true_candidate,index]);candidates=[c for c in candidates if bool(pred[c,index])==label];queried.append(index);used+=1
                    if len(candidates)>1:
                        _,terminal=compiler.get(candidates);assert terminal['feasible'];used+=terminal['leaf_depths'][str(true_candidate)]
                    naive[str(true_candidate)]=used
                assert naive==baseline['remaining_calls_by_candidate'] and budget.states==states and budget.splits==splits
                if remaining==0:assert decision['handoff'] and decision['tie']
                if remaining==1:assert int(decision['tree_weighted_cost'])==fixture['one_MAP_expected_tree_cost'] and int(decision['baseline_weighted_cost'])==fixture['one_MAP_expected_baseline_cost']
                runs.append(dict(remaining_MAP=remaining,baseline_group=baseline['group'],root_result_sha256=sha(folder/f'cost{baseline["group"]}/result.json'),calls_by_candidate=naive,guard=decision,repeat_cache_zero_work=True))
            assert calls[0]<=naive_calls;all_selector_calls+=calls[0];all_naive_calls+=naive_calls;rows.append(dict(**fixture,runs=runs,tree_groups=len(compiler.rows),baseline_groups=len(cost.rows),recursive_selector_calls=calls[0],naive_selector_calls=naive_calls,all_branches_identical=True))
        assert all_selector_calls<all_naive_calls
    finally:policy.OUT=old_policy_out;counterfactual.OUT=old_cost_out
    return dict(utc=now(),fixtures=rows,all_candidate_rollouts_match=True,accept_reject_and_tie_cases_verified=True,remaining_MAP_values=[0,1,2,4],recursive_selector_calls=all_selector_calls,naive_selector_calls=all_naive_calls,compiler_and_baseline_memo_states=budget.states-before_states,compiler_and_baseline_split_evaluations=budget.splits-before_splits,global_states_after_gate=budget.states,global_split_evaluations_after_gate=budget.splits,actual_oracle_calls=0,artifact_sha256={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob('*')) if path.is_file()} if keep_file_manifest else None,elapsed_seconds=time.perf_counter()-start)
