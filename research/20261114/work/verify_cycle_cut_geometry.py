"""Independent chain admission and cycle rank via component edge counting."""
import numpy as np
from verify_graph_block_source import geometry as chain_geometry,connected


def cycle_rank(board):
    h,w=board.shape;vertices=set(map(tuple,np.argwhere(board)));unseen=set(vertices);rank=0
    def neighbors(x):
        r,c=x;return [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
    while unseen:
        first=min(unseen);unseen.remove(first);seen={first};stack=[first];degree=0
        while stack:
            x=stack.pop()
            for y in neighbors(x):
                if y in vertices:
                    degree+=1
                    if y in unseen:unseen.remove(y);seen.add(y);stack.append(y)
        rank+=degree//2-len(seen)+1
    return rank


def geometry(tokens,board):
    raw,stats=chain_geometry(tokens,board);admitted=[]
    for cells in raw:
        changed=board.copy();changed.flat[list(cells)]=False
        if connected(tokens,changed):admitted.append(cells)
    stats.update(unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(admitted),reachability_checks=len(raw));before=cycle_rank(board);kept=[];ranks=[]
    for cells in admitted:
        changed=board.copy();changed.flat[list(cells)]=False;after=cycle_rank(changed)
        if after<before:kept.append(cells);ranks.append({'cells':list(cells),'before':before,'after':after})
    stats.update(chain_veto_candidate_count=len(admitted),cycle_rank=before,not_cycle_reducing=len(admitted)-len(kept),cycle_candidates=ranks);return kept,stats
