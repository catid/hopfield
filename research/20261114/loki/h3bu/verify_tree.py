"""Independent all-alternative certificate audit with redundant-depth normalization."""
def weighted_certificate(proof,catalog,check,prefix):
    candidates=proof['candidates'];weights=list(map(int,proof['weights']));full=(1<<len(candidates))-1;depth=proof['depth_limit'];states=proof['states'];root=f'{full}:{min(depth,len(candidates)-1)}';seen=set()
    check(prefix+'root',proof['root_state']==root and proof['depth_canonicalization']=='min(depth,subset_models-1)' and candidates==sorted(set(candidates)) and len(weights)==len(candidates) and all(w>0 for w in weights) and proof['weight_sum']==str(sum(weights)))
    def inspect(key):
        if key in seen:return int(states[key]['optimal_cost']) if states[key]['feasible'] else None
        seen.add(key);node=states[key];mask,d=map(int,key.split(':'));mass=sum(w for i,w in enumerate(weights) if mask&(1<<i));count=mask.bit_count();attempts=node['attempts'];check(prefix+key+'identity',node['mask']==mask and node['depth_remaining']==d and node['models']==count and node['weight_sum']==str(mass) and 0<=d<=count-1 and mask>0 and mask&full==mask)
        if count==1:check(prefix+key+'singleton',node['feasible'] and node['optimal_cost']=='0' and node['reason']=='singleton_leaf' and not attempts);return 0
        if count>2**d:check(prefix+key+'information',not node['feasible'] and node['optimal_cost'] is None and node['reason']=='binary_information_bound' and not attempts);return None
        partitions={}
        for row in catalog:
            pos=mask&row['signature'];neg=mask^pos
            if not pos or not neg:continue
            unordered=tuple(sorted((pos,neg)));proposal=(row['lexfirst_board_index'],pos,neg)
            if unordered not in partitions or proposal[0]<partitions[unordered][0]:partitions[unordered]=proposal
        options=[dict(canonical_side=min(pos,neg),positive_mask=pos,negative_mask=neg,board_index=index) for index,pos,neg in sorted(partitions.values())];check(prefix+key+'everyalternative',node['canonical_partition_count']==len(options)==len(attempts));possible=[]
        for i,(option,attempt) in enumerate(zip(options,attempts)):
            a=option['negative_mask'];b=option['positive_mask'];expected=dict(option)
            if max(a.bit_count(),b.bit_count())>2**(d-1):expected['result']='child_binary_information_bound'
            else:
                negative=f'{a}:{min(d-1,a.bit_count()-1)}';expected['negative_state']=negative;left=inspect(negative)
                if left is None:expected['result']='negative_child_infeasible'
                else:
                    positive=f'{b}:{min(d-1,b.bit_count()-1)}';expected['positive_state']=positive;right=inspect(positive)
                    if right is None:expected['result']='positive_child_infeasible'
                    else:
                        cost=mass+left+right;expected.update(result='both_children_feasible',candidate_cost=str(cost));possible.append((cost,option['board_index'],i))
            check(prefix+key+f'alternative{i}',attempt==expected)
        if not possible:check(prefix+key+'infeasibleexhaustion',not node['feasible'] and node['optimal_cost'] is None and node['reason']=='all_nontrivial_partitions_exhausted');return None
        cost,_,i=min(possible);check(prefix+key+'exactminimumandlexfirsttie',node['feasible'] and node['optimal_cost']==str(cost) and node['reason']=='all_alternatives_optimized' and node['chosen_split']==options[i] and node['negative_state']==attempts[i]['negative_state'] and node['positive_state']==attempts[i]['positive_state']);return cost
    cost=inspect(root);check(prefix+'allvisitedstates',seen==set(states) and proof['feasible']==(cost is not None) and proof['optimal_cost']==(str(cost) if cost is not None else None));depths={};queries=set()
    if cost is not None:
        for i,cid in enumerate(candidates):
            key=root;used=[]
            while states[key]['reason']!='singleton_leaf':
                node=states[key];index=node['chosen_split']['board_index'];check(prefix+f'branch{cid}norepeat{len(used)}',index not in used);used.append(index);queries.add(index);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
            check(prefix+f'branch{cid}correctleaf',states[key]['mask']==1<<i and len(used)<=depth);depths[str(cid)]=len(used)
        check(prefix+'weightedleafidentity',cost==sum(w*depths[str(cid)] for cid,w in zip(candidates,weights)))
    check(prefix+'allleafdepths',proof['leaf_depths']==depths and proof['worst_leaf_depth']==(max(depths.values()) if depths else None));return len(states),sum(len(n['attempts']) for n in states.values()),queries

