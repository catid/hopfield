"""Independent normalized optimality audit with exact bound and tie certificates."""
import bisect


def weighted_certificate(proof,catalog,check,prefix):
    candidates=proof['candidates'];weights=list(map(int,proof['weights']));full=(1<<len(candidates))-1;depth=proof['depth_limit'];states=proof['states'];root=f'{full}:{min(depth,len(candidates)-1)}';seen=set();bounds=proof['huffman_bounds'];bound_seen=set();bound_costs={};inline_merges=[0]
    check(prefix+'root',proof['root_state']==root and proof['depth_canonicalization']=='min(depth,subset_models-1)' and proof['pruning']=='huffman_bound_ordered_with_lexicographic_ties' and proof['trivial_bound_storage']=='all_inline_no_bound_memo' and not bounds and candidates==sorted(set(candidates)) and len(weights)==len(candidates) and all(w>0 for w in weights) and proof['weight_sum']==str(sum(weights)))
    def bound(mask):
        key=str(mask)
        if key in bound_seen:return bound_costs[key]
        check(prefix+'bound_mask',mask.bit_count()>2 and mask&full==mask);queue=sorted(w for i,w in enumerate(weights) if mask&(1<<i));merges=[];cost=0
        while len(queue)>1:
            a=queue.pop(0);b=queue.pop(0);value=a+b;cost+=value;merges.append([str(a),str(b),str(value)]);bisect.insort(queue,value)
        check(prefix+'independent_two_smallest_merge_certificate',bounds[key]==dict(mask=mask,models=mask.bit_count(),lower_bound=str(cost),merges=merges));bound_seen.add(key);bound_costs[key]=cost;return cost
    def certificate(mask):
        queue=sorted(w for i,w in enumerate(weights) if mask&(1<<i));cost=0;merges=[]
        while len(queue)>1:
            a=queue.pop(0);b=queue.pop(0);value=a+b;cost+=value;merges.append([str(a),str(b),str(value)]);bisect.insort(queue,value);inline_merges[0]+=1
        return cost,dict(kind='inline_recomputed',mask=mask,models=mask.bit_count(),lower_bound=str(cost),merges=merges)
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
        options=[dict(canonical_side=min(pos,neg),positive_mask=pos,negative_mask=neg,board_index=index) for index,pos,neg in partitions.values()];lower_bounds={}
        for option in options:
            if max(option['negative_mask'].bit_count(),option['positive_mask'].bit_count())<=2**(d-1):
                left,left_certificate=certificate(option['negative_mask']);right,right_certificate=certificate(option['positive_mask']);lower_bounds[option['board_index']]=(mass+left+right,left_certificate,right_certificate)
        options.sort(key=lambda option:(0,lower_bounds[option['board_index']][0],option['board_index']) if option['board_index'] in lower_bounds else (1,0,option['board_index']));check(prefix+key+'everyalternative',node['canonical_partition_count']==len(options)==len(attempts));possible=[]
        for i,(option,attempt) in enumerate(zip(options,attempts)):
            a=option['negative_mask'];b=option['positive_mask'];expected=dict(option);pruned=False
            if max(a.bit_count(),b.bit_count())>2**(d-1):expected['result']='child_binary_information_bound'
            else:
                lower,left_certificate,right_certificate=lower_bounds[option['board_index']];expected['ordering_bound']=dict(negative_mask=a,positive_mask=b,lower_bound=str(lower),negative_certificate=left_certificate,positive_certificate=right_certificate)
                if possible:
                    incumbent,board,_=min(possible);expected['bound_evaluation']=dict(negative_mask=a,positive_mask=b,lower_bound=str(lower),incumbent_cost=str(incumbent),incumbent_board_index=board);pruned=lower>incumbent or (lower==incumbent and option['board_index']>board)
                if pruned:expected['result']='huffman_bound_dominated'
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
        cost,_,i=min(possible);check(prefix+key+'certifiedminimumandlexfirsttie',node['feasible'] and node['optimal_cost']==str(cost) and node['reason']=='all_alternatives_certified' and node['chosen_split']==options[i] and node['negative_state']==attempts[i]['negative_state'] and node['positive_state']==attempts[i]['positive_state']);return cost
    cost=inspect(root);check(prefix+'all_tree_and_bound_states_referenced',seen==set(states) and bound_seen==set(bounds) and proof['feasible']==(cost is not None) and proof['optimal_cost']==(str(cost) if cost is not None else None));depths={};queries=set()
    if cost is not None:
        for i,cid in enumerate(candidates):
            key=root;used=[]
            while states[key]['reason']!='singleton_leaf':
                node=states[key];index=node['chosen_split']['board_index'];check(prefix+'nonrepeating_path',index not in used);used.append(index);queries.add(index);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
            check(prefix+'correct_leaf',states[key]['mask']==1<<i and len(used)<=depth);depths[str(cid)]=len(used)
        check(prefix+'weighted_leaf_cost',cost==sum(w*depths[str(cid)] for cid,w in zip(candidates,weights)))
    check(prefix+'all_leaf_depths',proof['leaf_depths']==depths and proof['worst_leaf_depth']==(max(depths.values()) if depths else None));return len(states)+len(bounds),sum(len(node['attempts']) for node in states.values())+sum(len(item['merges']) for item in bounds.values())+inline_merges[0],queries
