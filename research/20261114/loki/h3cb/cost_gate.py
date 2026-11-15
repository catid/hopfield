"""Synthetic variable-depth cache, all-branch admissibility and cost parity."""
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
    fixtures=[dict(name='tie_two',signatures=[0,1,2,3],scores=['1','1']),dict(name='accept_four',signatures=[0,1,2,4,8,3,12,15],scores=['9','10','10','10']),dict(name='reject_six',signatures=[0,1,2,4,8,16,32,3,63],scores=['1','100','100','100','100','100']),dict(name='deeper_weighted_six',signatures=list(range(64)),scores=['1','2','4','8','16','32'])]
    for remaining in [1,2,4]:
        n=remaining+6;full=(1<<n)-1;fixtures.append(dict(name=f'inadmissible_after{remaining}MAP',signatures=[0]+[1<<c for c in range(n)]+[(1<<(remaining+1))-1,full],scores=['1']*remaining+['100']*6,remaining_only=remaining))
    try:
        for fixture in fixtures:
            folder=root/fixture['name'];folder.mkdir();policy.OUT=folder;counterfactual.OUT=folder;n=len(fixture['scores']);pred=np.array([[bool(s&(1<<c)) for s in fixture['signatures']] for c in range(n)]);packed=np.packbits(pred,axis=1,bitorder='little');calls=[0];naive_calls=0
            def select(request):calls[0]+=1;return choose(request,packed,fixture['scores'])
            identities=dict(predictions_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),scores_sha256=hashlib.sha256(canonical(fixture['scores'])).hexdigest(),selector_sha256=sha(SRC/'selector.py'),tree_compiler_sha256=sha(SRC/'compiler.py'),tree_wrapper_sha256=sha(SRC/'policy.py'))
            compiler=policy.OnlineCompiler(pred,fixture['scores'],identities['predictions_sha256'],budget,100000);cost=counterfactual.BaselineCost(compiler,select,pred,budget,identities);runs=[]
            for remaining in ([fixture['remaining_only']] if 'remaining_only' in fixture else [0,1,2,4]):
                depth=4+remaining;_,tree=compiler.get(list(range(n)),depth);assert tree['feasible'];baseline=cost.evaluate(list(range(n)),[],remaining);decision=counterfactual.guard(tree,baseline);assert not decision['baseline_admissible'] or decision['cost_comparison_passes'];states=budget.states;splits=budget.splits;selector_count=calls[0];again=cost.evaluate(list(range(n)),[],remaining);repeat_tree=compiler.get(list(range(n)),depth)[1];assert again==baseline and repeat_tree==tree and budget.states==states and budget.splits==splits and calls[0]==selector_count;naive={};exact={}
                for truth in range(n):
                    candidates=list(range(n));queried=[];used=0
                    for _ in range(remaining):
                        if len(candidates)==1:break
                        request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=len(fixture['signatures']),candidates=candidates,queried=queried);selection=choose(request,packed,fixture['scores']);naive_calls+=1;index=selection['query_index'];assert index is not None;label=bool(pred[truth,index]);candidates=[c for c in candidates if bool(pred[c,index])==label];queried.append(index);used+=1
                    if len(candidates)>1:
                        _,terminal=compiler.get(candidates,4);exact[str(truth)]=terminal['feasible'];used+=terminal['leaf_depths'][str(truth)] if terminal['feasible'] else 0
                    else:exact[str(truth)]=True
                    naive[str(truth)]=used
                assert naive==baseline['remaining_calls_by_candidate'] and exact==baseline['exact_identification_by_candidate'] and budget.states==states and budget.splits==splits
                if remaining==0:assert decision['handoff'] and decision['tie'] and decision['baseline_admissible']
                if 'remaining_only' in fixture:assert not decision['baseline_admissible'] and not decision['handoff'] and decision['reason']=='baseline_inadmissible' and sum(not value for value in exact.values())==6
                runs.append(dict(remaining_MAP=remaining,requested_depth=depth,baseline_group=baseline['group'],root_result_sha256=sha(folder/f'cost{baseline["group"]}/result.json'),calls_by_candidate=naive,exact_by_candidate=exact,guard=decision,repeat_cache_zero_work=True))
            if fixture['name']=='deeper_weighted_six':
                original=compiler.cache[(tuple(range(n)),4,compiler.identity_key)][1];deep=compiler.cache[(tuple(range(n)),8,compiler.identity_key)][1];assert int(deep['optimal_cost'])<int(original['optimal_cost'])
            assert all((tuple(row['candidates']),row['depth'],compiler.identity_key) in compiler.cache for row in compiler.rows);assert len(compiler.cache)==len(compiler.rows);all_selector_calls+=calls[0];all_naive_calls+=naive_calls;rows.append(dict(**fixture,runs=runs,tree_groups=len(compiler.rows),baseline_groups=len(cost.rows),recursive_selector_calls=calls[0],naive_selector_calls=naive_calls,all_branches_identical=True,cache_key_includes_requested_depth=True))
    finally:policy.OUT=old_policy_out;counterfactual.OUT=old_cost_out
    assert all_selector_calls<all_naive_calls
    return dict(utc=now(),fixtures=rows,all_candidate_cost_and_exact_rollouts_match=True,inadmissible_branches_all_retained=True,deeper_tree_improvement_fixture_verified=True,recursive_selector_calls=all_selector_calls,naive_selector_calls=all_naive_calls,compiler_and_baseline_memo_states=budget.states-before_states,compiler_and_baseline_split_evaluations=budget.splits-before_splits,global_states_after_gate=budget.states,global_split_evaluations_after_gate=budget.splits,actual_oracle_calls=0,artifact_sha256={str(path.relative_to(root)):sha(path) for path in sorted(root.rglob('*')) if path.is_file()} if keep_file_manifest else None,elapsed_seconds=time.perf_counter()-start)
