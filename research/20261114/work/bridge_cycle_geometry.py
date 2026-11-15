"""Exact lexical256 geometry using one selected-graph bridge traversal.

The frozen production raw chains consist only of selected free degree-1/2
vertices. A candidate is a path or a cycle. Deleting a path can disconnect the
task endpoints only if one of its edges is a bridge separating those endpoints.
It removes one independent cycle precisely when its edges are not bridges;
deleting a complete isolated cycle also removes one independent cycle.

Returned metadata preserves logical predicate counts. It does not claim that
the legacy number of separate graph traversals was physically executed.
"""
import numpy as np
import graph_block_moves as RAW


def selected_graph(tokens,board):
    h,w=board.shape;b=board.ravel();vertices=list(map(int,np.flatnonzero(b)));adj={}
    for u in vertices:
        r,c=divmod(u,w)
        adj[u]=[v for v in (([u-w] if r else [])+([u+w] if r+1<h else [])+([u-1] if c else [])+([u+1] if c+1<w else [])) if b[v]]
    entered={};low={};left={};parent={};component={};bridges={};clock=0;components=0
    for root in vertices:
        if root in entered:continue
        entered[root]=low[root]=clock;clock+=1;parent[root]=-1;component[root]=components;components+=1;stack=[(root,iter(adj[root]))]
        while stack:
            u,it=stack[-1]
            try:v=next(it)
            except StopIteration:
                stack.pop();left[u]=clock;par=parent[u]
                if par!=-1:
                    low[par]=min(low[par],low[u])
                    if low[u]>entered[par]:bridges[(min(u,par),max(u,par))]=u
                continue
            if v==parent[u]:continue
            if v in entered:low[u]=min(low[u],entered[v]);continue
            parent[v]=u;entered[v]=low[v]=clock;clock+=1;component[v]=component[u];stack.append((v,iter(adj[v])))
    start=int(np.flatnonzero(tokens==2)[0]);goal=int(np.flatnonzero(tokens==3)[0]);connected=start in component and goal in component and component[start]==component[goal];critical=set()
    if connected:
        for edge,child in bridges.items():
            contains_start=entered[child]<=entered[start]<left[child];contains_goal=entered[child]<=entered[goal]<left[child]
            if contains_start!=contains_goal:critical.add(edge)
    rank=sum(map(len,adj.values()))//2-len(vertices)+components
    return adj,bridges,critical,connected,rank


def candidates(tokens,board):
    raw,stats=RAW.candidates(tokens,board);adj,bridges,critical,connected,rank=selected_graph(tokens,board);admitted=[];kept=[];ranks=[]
    for cells in raw:
        edges={(min(u,v),max(u,v)) for u in cells for v in adj[u]}
        if not connected or not edges.isdisjoint(critical):continue
        admitted.append(cells)
        if any(edge not in bridges for edge in edges):
            kept.append(cells);ranks.append({'cells':list(cells),'before':rank,'after':rank-1})
    stats.update(unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(admitted),reachability_checks=len(raw),chain_veto_candidate_count=len(admitted),cycle_rank=rank,not_cycle_reducing=len(admitted)-len(kept),cycle_candidates=ranks)
    return kept,stats
