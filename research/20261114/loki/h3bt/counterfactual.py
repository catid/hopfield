"""Exact baseline costs for every remaining model; no actual oracle labels."""
import hashlib
import time
from common import OUT,save,sha,now
from ledger import canonical


class BaselineCost:
    def __init__(self,compiler,select,predictions,budget,identities):
        self.compiler=compiler;self.select=select;self.predictions=predictions;self.budget=budget;self.identities=dict(identities);self.cache={};self.rows=[];self.memo_states=0;self.MAP_partitions=0

    def evaluate(self,candidates,queried,remaining_MAP):
        candidates=list(candidates);queried=sorted(set(queried));assert candidates==sorted(set(candidates)) and candidates and 0<=remaining_MAP<=4
        assert all(all(bool(self.predictions[c,index])==bool(self.predictions[candidates[0],index]) for c in candidates) for index in queried)
        inputs=dict(policy='fixed_remaining_MAP_then_posterior_depth4',candidates=candidates,remaining_MAP=remaining_MAP,queried=queried,depth=4,**self.identities);key=hashlib.sha256(canonical(inputs)).hexdigest()
        if key in self.cache:return self.cache[key]
        self.budget.state();self.memo_states+=1;group=len(self.rows);self.rows.append(None);folder=OUT/f'cost{group}';folder.mkdir();start=time.process_time();wall=time.perf_counter();children=[];selection=None;request=None;tree=None
        if len(candidates)==1:
            reason='singleton';costs={str(candidates[0]):0}
        elif remaining_MAP==0:
            reason='posterior_tree';receipt,proof=self.compiler.get(candidates);assert proof['feasible'],'Feasible ancestor must have feasible baseline descendants';costs={str(c):proof['leaf_depths'][str(c)] for c in candidates};tree=dict(group=receipt['group'],proof_sha256=receipt['proof_sha256'],group_result_sha256=sha(OUT/f'group{receipt["group"]}/result.json'))
        else:
            reason='MAP_partition';request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=self.predictions.shape[1],candidates=candidates,queried=queried);selection=self.select(request);index=selection['query_index'];assert index is not None and index not in queried,'Feasible ancestor must admit an informative MAP query'
            self.budget.split();self.MAP_partitions+=1;costs={}
            for label in [False,True]:
                subset=[c for c in candidates if bool(self.predictions[c,index])==label];assert subset and all(selection['candidate_predictions'][str(c)]==label for c in subset);child=self.evaluate(subset,queried+[index],remaining_MAP-1);children.append(dict(label=label,group=child['group'],result_sha256=sha(OUT/f'cost{child["group"]}/result.json'),cache_key=child['cache_key']))
                costs.update({str(c):1+child['remaining_calls_by_candidate'][str(c)] for c in subset})
        costs={str(c):costs[str(c)] for c in candidates};assert all(isinstance(v,int) and 0<=v<=remaining_MAP+4 for v in costs.values())
        save(folder/'input.json',inputs);row=dict(group=group,utc=now(),cache_key=key,input_sha256=sha(folder/'input.json'),candidates=candidates,remaining_MAP=remaining_MAP,queried=queried,reason=reason,request=request,selection=selection,children=children,tree=tree,remaining_calls_by_candidate=costs,all_candidate_branches_retained=True,actual_oracle_calls=0,CPU_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall);save(folder/'result.json',row);self.rows[group]=row;self.cache[key]=row;return row


def guard(tree_proof,baseline):
    candidates=tree_proof['candidates'];weights=list(map(int,tree_proof['weights']));assert tree_proof['feasible'] and baseline['candidates']==candidates
    tree_cost=int(tree_proof['optimal_cost']);baseline_cost=sum(w*baseline['remaining_calls_by_candidate'][str(c)] for c,w in zip(candidates,weights))
    return dict(tree_weighted_cost=str(tree_cost),baseline_weighted_cost=str(baseline_cost),weight_sum=tree_proof['weight_sum'],handoff=tree_cost<=baseline_cost,tie=tree_cost==baseline_cost,comparison='exact integer tree_cost <= baseline_cost',expected_nonincrease_if_handoff=tree_cost<=baseline_cost)
