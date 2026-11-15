"""Exact veto reachability via bridges, preserving the full raw chain family."""
import graph_block_moves as RAW
from bridge_cycle_geometry import selected_graph


def candidates(tokens,board):
    raw,stats=RAW.candidates(tokens,board);adj,bridges,critical,connected,rank=selected_graph(tokens,board);kept=[]
    if connected:
        for cells in raw:
            edges={(min(u,v),max(u,v)) for u in cells for v in adj[u]}
            if edges.isdisjoint(critical):kept.append(cells)
    return kept,{**stats,'unfiltered_candidate_count':len(raw),'connectivity_vetoed_components':len(raw)-len(kept),'reachability_checks':len(raw)}
