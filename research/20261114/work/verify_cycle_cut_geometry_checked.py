"""Check raw-chain metadata before comparison of agreed cycle-cut schema."""
from verify_cycle_cut_geometry import cycle_rank
from verify_cycle_cut_geometry import geometry as old_geometry
from verify_graph_block_source import geometry as chain_geometry


def geometry(tokens,board):
    raw,raw_stats=chain_geometry(tokens,board);assert raw_stats['candidate_count']==len(raw)
    cells,stats=old_geometry(tokens,board);assert stats['candidate_count']==len(raw);stats.pop('candidate_count');return cells,stats
