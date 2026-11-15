"""H2bg explicit bounded continuation; unchanged lexical/native solve body."""
import numpy as np
import study as S
import binary_coordinate as C
from lexical_cycle_cut_moves import scores
from continuation_geometry import candidates


def solve(cache,initial,watchdog=None,max_rounds=16,max_cells=256):
    torch,_,_=S.imports();p=initial.clone();n=len(p);done=np.zeros(n,bool);native_failure=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);native_sweeps=np.zeros(n,np.int64);trace=[];states=[];cpu_seconds=score_seconds=descent_seconds=0.;import time
    for round_index in range(max_rounds):
        before=time.perf_counter();boards=p.cpu().numpy().astype(bool);states.append(boards.copy());tokens=cache.t.cpu().numpy();items=[];geometry=[]
        for i in range(n):
            if done[i] or native_failure[i] or nonfinite[i]:continue
            blocks,extra=candidates(tokens[i],boards[i],max_cells);geometry.append({'local_input':i,'candidate_count':len(blocks),**extra});items.extend((i,cells) for cells in blocks)
            if not blocks:done[i]=True
        cpu_seconds+=time.perf_counter()-before
        if not items:
            trace.append({'round':round_index,'geometry':geometry,'candidates':[],'chosen':[]});break
        if p.is_cuda:torch.cuda.synchronize()
        before=time.perf_counter();values=scores(cache,p,items).cpu().numpy()
        if p.is_cuda:torch.cuda.synchronize()
        score_seconds+=time.perf_counter()-before;chosen=[];records=[]
        for j,((i,cells),v) in enumerate(zip(items,values)):records.append({'local_input':i,'cells':list(cells),'delta':float(v)})
        for i in range(n):
            ids=[j for j,(owner,_) in enumerate(items) if owner==i]
            if not ids:continue
            if not np.isfinite(values[ids]).all():nonfinite[i]=True;continue
            j=min(ids,key=lambda j:items[j][1])
            chosen.append(j)
        trace.append({'round':round_index,'geometry':geometry,'candidates':records,'chosen':chosen})
        if not chosen:break
        chosen_inputs=[]
        for j in chosen:
            i,cells=items[j];ix=torch.tensor(cells,device=p.device,dtype=torch.int64);assert bool(cache.free[i].flatten()[ix].all()) and bool((p[i].flatten()[ix]==1).all());p[i].flatten()[ix]=0;moves[i]+=1;chosen_inputs.append(i)
        if p.is_cuda:torch.cuda.synchronize()
        before=time.perf_counter();out=C.solve(cache,initial=p)
        if p.is_cuda:torch.cuda.synchronize()
        descent_seconds+=time.perf_counter()-before;p=out['prediction'];used=out['sweeps'].cpu().numpy();stopped=out['stopped'].cpu().numpy();finite=out['finite'].cpu().numpy()
        trace[-1]['native_results']=[{'local_input':i,'sweeps':int(used[i]),'stopped':bool(stopped[i]),'finite':bool(finite[i])} for i in chosen_inputs]
        for i in chosen_inputs:native_sweeps[i]+=used[i];native_failure[i]|=not stopped[i];nonfinite[i]|=not finite[i]
        if watchdog:watchdog()
    margin=cache.delta(p).masked_fill(~cache.free,float('inf')).flatten(1).amin(1);finite=torch.isfinite(p).flatten(1).all(1)&torch.isfinite(margin);block_cap=~done&~native_failure&~nonfinite
    assert torch.equal(p[~cache.free],cache.initial()[~cache.free])
    return {'prediction':p,'stopped':done&~native_failure&~nonfinite&finite.cpu().numpy(),'finite':finite.cpu().numpy()&~nonfinite,'block_moves':moves,'native_sweeps':native_sweeps,'native_cap':native_failure,'block_cap':block_cap,'minimum_flip_margin':margin.cpu().numpy(),'trace':trace,'states_before':np.stack(states),'cpu_graph_seconds':cpu_seconds,'score_seconds':score_seconds,'descent_seconds':descent_seconds}


def certify_final(tokens,raw,max_cells=256):
    board=raw['prediction'].detach().cpu().numpy().astype(bool);eligible=raw['block_cap']&raw['finite']&~raw['native_cap']&(raw['minimum_flip_margin']>=-1e-8);terminal=np.zeros(len(tokens),bool);records=[]
    for i in np.flatnonzero(eligible):
        cells,stats=candidates(tokens[i],board[i],max_cells);terminal[i]=not cells;records.append({'local_input':int(i),'candidate_count':len(cells),'candidates':[list(c) for c in cells],'geometry':stats})
    raw['pre_terminal_stopped']=raw['stopped'].copy();raw['pre_terminal_block_cap']=raw['block_cap'].copy();raw['stopped'][terminal]=True;raw['block_cap'][terminal]=False;raw['terminal_certified']=terminal;raw['terminal_check_records']=records;return raw
