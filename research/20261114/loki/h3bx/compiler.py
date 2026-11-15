"""Exact Huffman branch bounds with lexicographic-safe pruning."""
import heapq


class ResourceCap(RuntimeError):pass


class Budget:
    def __init__(self,max_states=2000000,max_splits=50000000):self.max_states=max_states;self.max_splits=max_splits;self.states=0;self.splits=0
    def state(self):
        if self.states>=self.max_states:raise ResourceCap('Global state cap; whole-study incomplete')
        self.states+=1
    def split(self):
        if self.splits>=self.max_splits:raise ResourceCap('Global split-attempt cap; whole-study incomplete')
        self.splits+=1


def compile_tree(candidates,weights,catalog,depth,budget):
    assert candidates==sorted(set(candidates)) and 1<=len(candidates)<=16 and len(weights)==len(candidates) and all(isinstance(w,int) and w>0 for w in weights)
    full=(1<<len(candidates))-1;states={};mass_cache={};bounds={}
    def mass(mask):
        if mask not in mass_cache:mass_cache[mask]=sum(w for i,w in enumerate(weights) if mask&(1<<i))
        return mass_cache[mask]
    def huffman(mask):
        heap=[w for i,w in enumerate(weights) if mask&(1<<i)];heapq.heapify(heap);merges=[];cost=0
        while len(heap)>1:
            budget.split();a=heapq.heappop(heap);b=heapq.heappop(heap);merged=a+b;cost+=merged;merges.append([str(a),str(b),str(merged)]);heapq.heappush(heap,merged)
        return cost,dict(kind='inline_recomputed',mask=mask,models=mask.bit_count(),lower_bound=str(cost),merges=merges)
    def solve(mask,left):
        left=min(left,mask.bit_count()-1);key=f'{mask}:{left}'
        if key in states:return int(states[key]['optimal_cost']) if states[key]['feasible'] else None
        budget.state();record=dict(mask=mask,depth_remaining=left,models=mask.bit_count(),weight_sum=str(mass(mask)),feasible=False,optimal_cost=None,reason=None,attempts=[]);states[key]=record
        if mask.bit_count()==1:record.update(feasible=True,optimal_cost='0',reason='singleton_leaf');return 0
        if mask.bit_count()>2**left:record['reason']='binary_information_bound';return None
        partitions={}
        for entry in catalog:
            positive=mask&entry['signature'];negative=mask^positive
            if not positive or not negative:continue
            canonical=min(positive,negative);proposal=dict(canonical_side=canonical,positive_mask=positive,negative_mask=negative,board_index=entry['lexfirst_board_index'])
            if canonical not in partitions or proposal['board_index']<partitions[canonical]['board_index']:partitions[canonical]=proposal
        options=list(partitions.values());lower_bounds={}
        for option in options:
            a=option['negative_mask'];b=option['positive_mask']
            if max(a.bit_count(),b.bit_count())<=2**(left-1):
                negative_cost,negative_bound=huffman(a);positive_cost,positive_bound=huffman(b);lower_bounds[option['board_index']]=(mass(mask)+negative_cost+positive_cost,negative_bound,positive_bound)
        options.sort(key=lambda option:(0,lower_bounds[option['board_index']][0],option['board_index']) if option['board_index'] in lower_bounds else (1,0,option['board_index']));record['canonical_partition_count']=len(options);best=None;chosen=None
        for option in options:
            budget.split();attempt=dict(option);a=option['negative_mask'];b=option['positive_mask']
            if max(a.bit_count(),b.bit_count())>2**(left-1):attempt['result']='child_binary_information_bound';record['attempts'].append(attempt);continue
            lower,negative_bound,positive_bound=lower_bounds[option['board_index']];attempt['ordering_bound']=dict(negative_mask=a,positive_mask=b,lower_bound=str(lower),negative_certificate=negative_bound,positive_certificate=positive_bound)
            if best is not None:
                attempt['bound_evaluation']=dict(negative_mask=a,positive_mask=b,lower_bound=str(lower),incumbent_cost=str(best),incumbent_board_index=chosen['board_index'])
                if lower>best or (lower==best and option['board_index']>chosen['board_index']):
                    attempt['result']='huffman_bound_dominated';record['attempts'].append(attempt);continue
            negative_key=f'{a}:{min(left-1,a.bit_count()-1)}';attempt['negative_state']=negative_key;negative=solve(a,left-1)
            if negative is None:attempt['result']='negative_child_infeasible';record['attempts'].append(attempt);continue
            positive_key=f'{b}:{min(left-1,b.bit_count()-1)}';attempt['positive_state']=positive_key;positive=solve(b,left-1)
            if positive is None:attempt['result']='positive_child_infeasible';record['attempts'].append(attempt);continue
            cost=mass(mask)+negative+positive;attempt.update(result='both_children_feasible',candidate_cost=str(cost));record['attempts'].append(attempt)
            if best is None or (cost,option['board_index'])<(best,chosen['board_index']):best=cost;chosen=dict(option);record.update(chosen_split=chosen,negative_state=negative_key,positive_state=positive_key)
        if best is None:record['reason']='all_nontrivial_partitions_exhausted';return None
        record.update(feasible=True,optimal_cost=str(best),reason='all_alternatives_certified');return best
    root=f'{full}:{min(depth,len(candidates)-1)}';cost=solve(full,depth);leaf_depths={}
    if cost is not None:
        for local,cid in enumerate(candidates):
            key=root;used=[]
            while states[key]['reason']!='singleton_leaf':
                node=states[key];split=node['chosen_split'];assert split['board_index'] not in used;used.append(split['board_index']);key=node['positive_state'] if split['positive_mask']&(1<<local) else node['negative_state']
            assert states[key]['mask']==1<<local and len(used)<=depth;leaf_depths[str(cid)]=len(used)
        assert cost==sum(w*leaf_depths[str(cid)] for cid,w in zip(candidates,weights))
    return dict(candidates=candidates,weights=[str(w) for w in weights],depth_limit=depth,depth_canonicalization='min(depth,subset_models-1)',pruning='huffman_bound_ordered_with_lexicographic_ties',huffman_bounds=bounds,trivial_bound_storage='all_inline_no_bound_memo',root_state=root,feasible=cost is not None,optimal_cost=str(cost) if cost is not None else None,weight_sum=str(sum(weights)),states=states,leaf_depths=leaf_depths,worst_leaf_depth=max(leaf_depths.values()) if leaf_depths else None)
