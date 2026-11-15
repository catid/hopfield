"""Small exact brute force verifies expected-cost minimization before real sets."""
from compiler import Budget,ResourceCap,compile_tree
from weights import integer_weights
from common import OUT,save,now


def brute(mask,depth,signatures,weights):
    if mask.bit_count()==1:return 0
    if depth==0:return None
    mass=sum(w for i,w in enumerate(weights) if mask&(1<<i));costs=[]
    for signature in signatures:
        positive=mask&signature;negative=mask^positive
        if not positive or not negative:continue
        a=brute(positive,depth-1,signatures,weights);b=brute(negative,depth-1,signatures,weights)
        if a is not None and b is not None:costs.append(mass+a+b)
    return min(costs) if costs else None


def run(budget):
    fixtures=[('two_models_lex_tie',[1,1],[0,1,2,3],1,2),('three_uniform',[1,1,1],[0,1,2,4,7],2,5),('three_heavy_first',[10,1,1],[0,1,2,4,7],2,14),('four_singleton_only_shallow',[1,1,1,1],[0,1,2,4,8,15],2,None),('four_singleton_only_deep',[1,1,1,1],[0,1,2,4,8,15],3,9),('duplicate_predictions',[1,2],[0,3],4,None),('four_binary_uniform',[1,1,1,1],[0,5,10,3,12,15],2,8),('binary_depth_insufficient',[1,1,1,1],[0,5,10,3,12,15],1,None)];rows=[]
    for name,weights,signatures,depth,expected in fixtures:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(signatures)];proof=compile_tree(list(range(len(weights))),weights,catalog,depth,budget);cost=int(proof['optimal_cost']) if proof['feasible'] else None;assert cost==expected==brute((1<<len(weights))-1,depth,signatures,weights)
        if name in ['two_models_lex_tie','three_heavy_first']:assert proof['states'][proof['root_state']]['chosen_split']['board_index']==1
        rows.append(dict(name=name,weights=weights,signatures=signatures,depth=depth,expected_cost=str(expected) if expected is not None else None,result=proof))
    rejected={}
    for name,limited in [('state_cap',Budget(0,100)),('split_cap',Budget(100,0))]:
        try:compile_tree([0,1],[1,1],[dict(signature=1,lexfirst_board_index=0,board_count=1)],1,limited)
        except ResourceCap:rejected[name]=True
        else:raise AssertionError('Cap did not raise incomplete')
    budget.state()
    try:integer_weights([0,1],['5','7'],max_bits=4)
    except ResourceCap:rejected['weight_bit_cap']=True
    else:raise AssertionError('Weight cap did not raise incomplete')
    weights,receipt=integer_weights([0,1,2],['6','10','15']);assert weights==[5,3,2]
    result=dict(utc=now(),fixtures=rows,all_independent_brute_force_costs_identical=True,resource_caps_raise_incomplete=rejected,weight_conversion_fixture=receipt,global_states_after_gate=budget.states,global_split_attempts_after_gate=budget.splits);save(OUT/'synthetic-gate.json',result);return result
