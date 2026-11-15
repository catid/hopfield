"""H2bf unchanged negative-energy veto solver with exact bridge geometry."""
import numpy as np
import study as S
import binary_coordinate as C
from connected_block_moves import scores,simple_path
from bridge_veto_geometry import candidates
MAX_ROUNDS=16
THRESHOLD=-1e-8


def solve(cache,initial,watchdog=None):
    torch,_,_=S.imports();p=initial.clone();n=len(p);done=np.zeros(n,bool);native_failure=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);native_sweeps=np.zeros(n,np.int64);trace=[];states=[];cpu_seconds=score_seconds=descent_seconds=0.;import time
    for round_index in range(MAX_ROUNDS):
        before=time.perf_counter();boards=p.cpu().numpy().astype(bool);states.append(boards.copy());tokens=cache.t.cpu().numpy();items=[];geometry=[]
        for i in range(n):
            if done[i] or native_failure[i] or nonfinite[i]:continue
            blocks,extra=candidates(tokens[i],boards[i]);geometry.append({'local_input':i,'candidate_count':len(blocks),**extra});items.extend((i,cells) for cells in blocks)
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
            j=min(ids,key=lambda j:(float(values[j]),items[j][1]))
            if values[j]<THRESHOLD:chosen.append(j)
            else:done[i]=True
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
