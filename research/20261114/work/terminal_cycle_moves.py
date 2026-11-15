"""H2ba terminal-only certification and unchanged sixteen-round lexical continuation."""
import numpy as np
import lexical_cycle_cut_moves as LEX


def terminal_checks(tokens,boards,eligible):
    terminal=np.zeros(len(tokens),bool);records=[]
    for i in np.flatnonzero(eligible):
        cells,stats=LEX.candidates(tokens[i],boards[i]);terminal[i]=not cells;records.append({'local_input':int(i),'candidate_count':len(cells),'candidates':[list(c) for c in cells],'geometry':stats})
    return terminal,records


def certify_final(tokens,raw):
    """No board/search change: inspect the state reached after the last allowed move."""
    board=raw['prediction'].detach().cpu().numpy().astype(bool);eligible=raw['block_cap']&raw['finite']&~raw['native_cap']&(raw['minimum_flip_margin']>=-1e-8);terminal,records=terminal_checks(tokens,board,eligible);raw['pre_terminal_stopped']=raw['stopped'].copy();raw['pre_terminal_block_cap']=raw['block_cap'].copy();raw['stopped'][terminal]=True;raw['block_cap'][terminal]=False;raw['terminal_certified']=terminal;raw['terminal_check_records']=records;return raw


def solve(cache,initial,watchdog=None):
    raw=LEX.solve(cache,initial,watchdog);return certify_final(cache.t.cpu().numpy(),raw)
