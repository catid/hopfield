"""Truth-blind bounded exact feasibility compiler over a complete signature catalog."""


class ResourceCap(RuntimeError):pass


class Budget:
    def __init__(self,max_states=250000,max_splits=10000000):self.max_states=max_states;self.max_splits=max_splits;self.states=0;self.splits=0
    def state(self):
        if self.states>=self.max_states:raise ResourceCap('Global memo-state cap; incomplete, not infeasible')
        self.states+=1
    def split(self):
        if self.splits>=self.max_splits:raise ResourceCap('Global canonical-split-attempt cap; incomplete, not infeasible')
        self.splits+=1


def compile_tree(candidates,catalog,depth,budget):
    """Only candidates, all board signatures, depth, and resource counters are inputs."""
    assert candidates==sorted(set(candidates)) and 1<=len(candidates)<=16
    states={};full=(1<<len(candidates))-1
    def solve(mask,left):
        key=f'{mask}:{left}'
        if key in states:return states[key]['feasible']
        budget.state();record=dict(mask=mask,depth_remaining=left,models=mask.bit_count(),feasible=None,reason=None,attempts=[]);states[key]=record
        if mask.bit_count()==1:record.update(feasible=True,reason='singleton_leaf');return True
        if mask.bit_count()>2**left:record.update(feasible=False,reason='binary_information_bound');return False
        partitions={}
        for entry in catalog:
            positive=mask&entry['signature'];negative=mask^positive
            if positive==0 or negative==0:continue
            canonical=min(positive,negative);proposal=dict(canonical_side=canonical,positive_mask=positive,negative_mask=negative,board_index=entry['lexfirst_board_index'])
            if canonical not in partitions or proposal['board_index']<partitions[canonical]['board_index']:partitions[canonical]=proposal
        options=sorted(partitions.values(),key=lambda row:row['board_index']);record['canonical_partition_count']=len(options)
        for option in options:
            budget.split();attempt=dict(option);a=option['negative_mask'];b=option['positive_mask']
            if max(a.bit_count(),b.bit_count())>2**(left-1):attempt['result']='child_binary_information_bound';record['attempts'].append(attempt);continue
            negative_key=f'{a}:{left-1}';attempt['negative_state']=negative_key
            if not solve(a,left-1):attempt['result']='negative_child_infeasible';record['attempts'].append(attempt);continue
            positive_key=f'{b}:{left-1}';attempt['positive_state']=positive_key
            if not solve(b,left-1):attempt['result']='positive_child_infeasible';record['attempts'].append(attempt);continue
            attempt['result']='both_children_feasible';record['attempts'].append(attempt);record.update(feasible=True,reason='feasible_split',chosen_split=option,negative_state=negative_key,positive_state=positive_key);return True
        record.update(feasible=False,reason='all_nontrivial_partitions_exhausted');return False
    root=f'{full}:{depth}';feasible=solve(full,depth);leaf_depths={}
    if feasible:
        for local,cid in enumerate(candidates):
            key=root;used=[]
            while states[key]['reason']!='singleton_leaf':
                node=states[key];split=node['chosen_split'];assert split['board_index'] not in used;used.append(split['board_index']);key=node['positive_state'] if split['positive_mask']&(1<<local) else node['negative_state']
            assert states[key]['mask']==1<<local and len(used)<=depth;leaf_depths[str(cid)]=len(used)
    return dict(candidates=candidates,depth_limit=depth,root_state=root,feasible=feasible,states=states,leaf_depths=leaf_depths,worst_leaf_depth=max(leaf_depths.values()) if leaf_depths else None)
