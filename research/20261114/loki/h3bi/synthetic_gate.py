"""Exact small fixtures and resource-cap distinction before real checkpoints."""
from compiler import Budget,ResourceCap,compile_tree
from common import OUT,save,now


def catalog(signatures):return [dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(signatures)]


def brute(mask,depth,signatures):
    if mask.bit_count()==1:return True
    if depth==0:return False
    for signature in signatures:
        positive=signature&mask;negative=mask^positive
        if positive and negative and brute(positive,depth-1,signatures) and brute(negative,depth-1,signatures):return True
    return False


def run(budget):
    binary=[sum(1<<cid for cid in range(16) if cid&(1<<bit)) for bit in range(4)]
    fixtures=[('two_models_lex_tie',2,[0,1,2,3],1,True),('four_singleton_only_depth2',4,[0,1,2,4,8,15],2,False),('four_singleton_only_depth3',4,[0,1,2,4,8,15],3,True),('duplicate_predictions',2,[0,3],4,False),('sixteen_binary_depth4',16,[0]+binary+[65535],4,True),('sixteen_binary_depth3',16,[0]+binary+[65535],3,False),('three_models_depth1',3,[0,1,2,4,7],1,False),('three_models_depth2',3,[0,1,2,4,7],2,True)];rows=[]
    for name,n,signatures,depth,expected in fixtures:
        result=compile_tree(list(range(n)),catalog(signatures),depth,budget);assert result['feasible']==expected==brute((1<<n)-1,depth,signatures)
        if name=='two_models_lex_tie':assert result['states'][result['root_state']]['chosen_split']['board_index']==1
        rows.append(dict(name=name,signatures=signatures,depth=depth,expected=expected,result=result))
    rejected={}
    for name,limited in [('state_cap',Budget(0,100)),('split_cap',Budget(100,0))]:
        try:compile_tree([0,1],catalog([1,2]),1,limited)
        except ResourceCap:rejected[name]=True
        else:raise AssertionError('Resource cap treated as feasibility result')
    # Charge deliberately failing cap-test work to the global accounting too.
    budget.state()
    result=dict(utc=now(),fixtures=rows,all_independent_brute_force_results_identical=True,resource_caps_raise_incomplete=rejected,global_states_after_gate=budget.states,global_split_attempts_after_gate=budget.splits);save(OUT/'synthetic-gate.json',result);return result
