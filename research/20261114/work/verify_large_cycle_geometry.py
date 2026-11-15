"""Independent degree/chain geometry with an explicit size argument."""
import numpy as np
from verify_graph_block_source import connected
from verify_cycle_cut_geometry import cycle_rank


def raw_chains(tokens,board,limit=512):
    h,w=board.shape;selected=set(map(tuple,np.argwhere(board)))
    def adjacent(x):
        r,c=x;return [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
    eligible={x for x in selected if tokens[x]==1 and 1<=sum(y in selected for y in adjacent(x))<=2};unseen=set(eligible);kept=[];oversize=single=0
    while unseen:
        root=min(unseen);unseen.remove(root);part={root};queue=[root]
        while queue:
            for v in adjacent(queue.pop()):
                if v in unseen:unseen.remove(v);part.add(v);queue.append(v)
        cells=tuple(sorted(int(r*w+c) for r,c in part))
        if len(cells)==1:single+=1
        elif len(cells)>limit:oversize+=1
        else:kept.append(cells)
    return sorted(kept),{'oversize_components':oversize,'singletons':single,'eligible_vertices':len(eligible)}


def geometry(tokens,board,limit=512):
    raw,stats=raw_chains(tokens,board,limit);admitted=[]
    for cells in raw:
        changed=board.copy();changed.flat[list(cells)]=False
        if connected(tokens,changed):admitted.append(cells)
    stats.update(unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(admitted),reachability_checks=len(raw));before=cycle_rank(board);kept=[];ranks=[]
    for cells in admitted:
        changed=board.copy();changed.flat[list(cells)]=False;after=cycle_rank(changed)
        if after<before:kept.append(cells);ranks.append({'cells':list(cells),'before':before,'after':after})
    stats.update(chain_veto_candidate_count=len(admitted),cycle_rank=before,not_cycle_reducing=len(admitted)-len(kept),cycle_candidates=ranks);return kept,stats
