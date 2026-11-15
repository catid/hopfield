"""H2bg explicit chain-size limit; immutable source bridge traversal reused."""
import numpy as np
from bridge_cycle_geometry import selected_graph


def raw_candidates(tokens,board,max_cells):
    """Maximal chains: selected free global-degree1/2 vertices, four-neighbor components."""
    assert board.dtype==bool and np.array_equal(board[tokens!=1],(tokens>=2)[tokens!=1]);h,w=board.shape;degree=np.zeros(board.shape,np.int8);degree[1:]+=board[:-1];degree[:-1]+=board[1:];degree[:,1:]+=board[:,:-1];degree[:,:-1]+=board[:,1:]
    eligible=board&(tokens==1)&(degree>=1)&(degree<=2);remaining=set(map(int,np.flatnonzero(eligible)));result=[];oversize=singletons=0
    while remaining:
        start=min(remaining);remaining.remove(start);stack=[start];part=[]
        while stack:
            u=stack.pop();part.append(u);r,c=divmod(u,w)
            for v in ([u-w] if r else [])+([u+w] if r+1<h else [])+([u-1] if c else [])+([u+1] if c+1<w else []):
                if v in remaining:remaining.remove(v);stack.append(v)
        part=tuple(sorted(part))
        if len(part)==1:singletons+=1
        elif len(part)>max_cells:oversize+=1
        else:result.append(part)
    return sorted(result),{'oversize_components':oversize,'singletons':singletons,'eligible_vertices':int(eligible.sum())}


def candidates(tokens,board,max_cells=256):
    raw,stats=raw_candidates(tokens,board,max_cells);adj,bridges,critical,connected,rank=selected_graph(tokens,board);admitted=[];kept=[];ranks=[]
    for cells in raw:
        edges={(min(u,v),max(u,v)) for u in cells for v in adj[u]}
        if not connected or not edges.isdisjoint(critical):continue
        admitted.append(cells)
        if any(edge not in bridges for edge in edges):
            kept.append(cells);ranks.append({'cells':list(cells),'before':rank,'after':rank-1})
    stats.update(unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(admitted),reachability_checks=len(raw),chain_veto_candidate_count=len(admitted),cycle_rank=rank,not_cycle_reducing=len(admitted)-len(kept),cycle_candidates=ranks)
    return kept,stats
