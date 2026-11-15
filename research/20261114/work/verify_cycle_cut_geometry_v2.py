"""H2ar schema-normalized independent H2aq cycle geometry; no policy change."""
from verify_cycle_cut_geometry import cycle_rank
from verify_cycle_cut_geometry import geometry as old_geometry


def geometry(tokens,board):
    cells,stats=old_geometry(tokens,board);stats.pop('candidate_count');return cells,stats
