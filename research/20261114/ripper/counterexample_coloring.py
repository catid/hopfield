"""Bounded deterministic exact search for a proper nine-color counterexample.

Color names are canonicalized with the forced-equal pair assigned color zero.
An interruption returns unknown; only an explicit coloring is evidence.
"""
import time


def search(support,missing_pairs,cpu_seconds=2.,node_limit=100000):
    pairs=[(i,j)for i in range(81)for j in range(i+1,81)]
    adjacency=[set()for _ in range(81)]
    for (i,j),keep in zip(pairs,support):
        if keep:adjacency[i].add(j);adjacency[j].add(i)
    started=time.process_time();nodes=0;attempted=0
    class Budget(Exception):pass
    def dfs(colors,max_color):
        nonlocal nodes
        nodes+=1
        if nodes>node_limit or (nodes%128==0 and time.process_time()-started>=cpu_seconds):raise Budget
        candidates=[]
        for vertex in range(81):
            if colors[vertex]>=0:continue
            forbidden={colors[n]for n in adjacency[vertex]if colors[n]>=0}
            if len(forbidden)>=9:return None
            candidates.append((len(forbidden),len(adjacency[vertex]),-vertex,vertex,forbidden))
        if not candidates:return colors.copy()
        _,_,_,vertex,forbidden=max(candidates,key=lambda x:x[:3])
        for color in range(min(max_color+1,8)+1):
            if color in forbidden:continue
            colors[vertex]=color;answer=dfs(colors,max(max_color,color))
            if answer is not None:return answer
        colors[vertex]=-1;return None
    try:
        for a,b in missing_pairs:
            assert b not in adjacency[a];attempted+=1
            if time.process_time()-started>=cpu_seconds:raise Budget
            colors=[-1]*81;colors[a]=colors[b]=0;answer=dfs(colors,0)
            if answer is not None:
                assert all(answer[i]!=answer[j]for (i,j),keep in zip(pairs,support)if keep)
                return dict(status='explicit_counterexample',board=answer,forced_equal_pair=[a,b],nodes=nodes,attempted_pairs=attempted,cpu_seconds=time.process_time()-started)
        return dict(status='no_witness_in_complete_search',nodes=nodes,attempted_pairs=attempted,cpu_seconds=time.process_time()-started)
    except Budget:
        return dict(status='unresolved_budget',nodes=nodes,attempted_pairs=attempted,cpu_seconds=time.process_time()-started)
