"""Synthetic parity and measured work for redundant-depth normalization."""
import time
from compiler import compile_tree,Budget,ResourceCap
from reference_compiler import compile_tree as reference_tree
from verify_tree import weighted_certificate
from common import save,now


def paths(proof):
    if not proof['feasible']:return None
    result={}
    for i,candidate in enumerate(proof['candidates']):
        state=proof['root_state'];path=[]
        while proof['states'][state]['reason']!='singleton_leaf':
            node=proof['states'][state];path.append(node['chosen_split']['board_index']);state=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
        result[str(candidate)]=path
    return result


def run(folder,budget):
    folder.mkdir(parents=True,exist_ok=True);fixtures=[dict(name='two_ties',weights=[1,1],signatures=list(range(4))),dict(name='six_full_weighted',weights=[32,16,8,4,2,1],signatures=list(range(64))),dict(name='eight_full_weighted',weights=[128,64,32,16,8,4,2,1],signatures=list(range(256))),dict(name='eight_singletons',weights=[3,1,8,2,7,4,6,5],signatures=[0]+[1<<i for i in range(8)]+[255]),dict(name='duplicate_predictions',weights=[1,2,3,4],signatures=[0,3,12,15])];rows=[];total=dict(reference_states=0,normalized_states=0,reference_splits=0,normalized_splits=0,reference_CPU_seconds=0.,normalized_CPU_seconds=0.,reference_wall_seconds=0.,normalized_wall_seconds=0.);checks=[0]
    def check(name,value):assert value,name;checks[0]+=1
    for fixture in fixtures:
        n=len(fixture['weights']);catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])]
        for depth in range(1,9):
            item=dict(name=fixture['name'],requested_depth=depth,candidates=list(range(n)),weights=fixture['weights'],signatures=fixture['signatures']);proofs={}
            for kind,compiler in [('reference',reference_tree),('normalized',compile_tree)]:
                bs=budget.states;bt=budget.splits;cpu=time.process_time();wall=time.perf_counter();proof=compiler(list(range(n)),fixture['weights'],catalog,depth,budget);elapsed=time.perf_counter()-wall;used=time.process_time()-cpu;proofs[kind]=proof;item[kind]=dict(states=budget.states-bs,splits=budget.splits-bt,CPU_seconds=used,wall_seconds=elapsed,feasible=proof['feasible'],optimal_cost=proof['optimal_cost'],paths=paths(proof));total[kind+'_states']+=budget.states-bs;total[kind+'_splits']+=budget.splits-bt;total[kind+'_CPU_seconds']+=used;total[kind+'_wall_seconds']+=elapsed;save(folder/f'{fixture["name"]}-depth{depth}-{kind}.json',proof)
            check('exact_cost_feasibility_and_all_lex_paths',all(item['reference'][field]==item['normalized'][field] for field in ['feasible','optimal_cost','paths']));check('normalization_never_more_work',item['normalized']['states']<=item['reference']['states'] and item['normalized']['splits']<=item['reference']['splits']);weighted_certificate(proofs['normalized'],catalog,check,'normalized_');rows.append(item)
    caps={}
    for name,limited in [('state_cap',Budget(0,100)),('split_cap',Budget(100,0))]:
        try:compile_tree([0,1],[1,1],[dict(signature=1,lexfirst_board_index=0,board_count=1)],8,limited)
        except ResourceCap:caps[name]=True
        else:raise AssertionError('Normalized compiler cap failed')
    assert total['normalized_states']<total['reference_states'] and total['normalized_splits']<total['reference_splits'];result=dict(utc=now(),cases=rows,all40_synthetic_cost_feasibility_and_lex_paths_identical=True,all_normalized_certificates_independently_verified=True,checks=checks[0],caps_raise_incomplete=caps,**total,global_states_after_gate=budget.states,global_splits_after_gate=budget.splits);save(folder/'summary.json',result);return result
