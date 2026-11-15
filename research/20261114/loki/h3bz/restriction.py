"""Construct a depth-nonincreasing identification tree for any nonempty subset."""


def restrict_tree(proof,candidates):
    assert proof['feasible'] and candidates and set(candidates)<=set(proof['candidates'])
    mask=sum(1<<proof['candidates'].index(cid) for cid in candidates)
    def visit(state,active):
        if active.bit_count()==1:return dict(candidate=proof['candidates'][active.bit_length()-1])
        node=proof['states'][state];split=node['chosen_split'];positive=active&split['positive_mask'];negative=active^positive
        if not positive:return visit(node['negative_state'],negative)
        if not negative:return visit(node['positive_state'],positive)
        return dict(board_index=split['board_index'],negative=visit(node['negative_state'],negative),positive=visit(node['positive_state'],positive))
    return visit(proof['root_state'],mask)


def gate(fixtures):
    checks=0;subsets=0
    for fixture in fixtures:
        proof=fixture['result']
        if not proof['feasible']:continue
        candidates=proof['candidates'];signatures=fixture['signatures']
        for mask in range(1,1<<len(candidates)):
            chosen=[cid for i,cid in enumerate(candidates) if mask&(1<<i)];tree=restrict_tree(proof,chosen);subsets+=1
            for cid in chosen:
                node=tree;path=[];local=candidates.index(cid)
                while 'candidate' not in node:
                    index=node['board_index'];assert index not in path;path.append(index);node=node['positive'] if signatures[index]&(1<<local) else node['negative']
                assert node['candidate']==cid and len(path)<=proof['leaf_depths'][str(cid)]<=proof['depth_limit'];checks+=1
    assert subsets>0
    return dict(all_nonempty_toy_subsets_verified=subsets,all_subset_candidate_paths_verified=checks,depth_never_increases=True)
