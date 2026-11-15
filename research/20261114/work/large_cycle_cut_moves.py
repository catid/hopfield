"""H2aw same cycle-cut solver with only the chain size limit expanded to512."""
import numpy as np
import study as S
import binary_coordinate as C
from connected_block_moves import reachable,simple_path
from cycle_cut_moves import cycle_rank
MAX_CELLS=512
MAX_ROUNDS=16
THRESHOLD=-1e-8


def chain_candidates(tokens,board):
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
        elif len(part)>MAX_CELLS:oversize+=1
        else:result.append(part)
    return sorted(result),{'oversize_components':oversize,'singletons':singletons,'eligible_vertices':int(eligible.sum())}


def candidates(tokens,board):
    raw,stats=chain_candidates(tokens,board);admitted=[cells for cells in raw if reachable(tokens,board,cells)];stats.update(unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(admitted),reachability_checks=len(raw));before=cycle_rank(board);kept=[];ranks=[]
    for cells in admitted:
        changed=board.copy();changed.flat[list(cells)]=False;after=cycle_rank(changed)
        if after<before:kept.append(cells);ranks.append({'cells':list(cells),'before':before,'after':after})
    return kept,{**stats,'chain_veto_candidate_count':len(admitted),'cycle_rank':before,'not_cycle_reducing':len(admitted)-len(kept),'cycle_candidates':ranks}


def scores(cache,p,items):
    """Delta=sum single flips + each removed internal quadratic edge once."""
    torch,_,_=S.imports();h,w=p.shape[-2:];owners=torch.tensor([item[0] for item in items],device=p.device,dtype=torch.int64);mask=np.zeros((len(items),h*w),bool)
    for j,(_,cells) in enumerate(items):mask[j,list(cells)]=True
    m=torch.tensor(mask.reshape(len(items),h,w),device=p.device);single=cache.delta(p);value=(single[owners]*m).sum((1,2))
    for (dr,dc),weight in cache.weights.items():
        l,r=C.slices(h,w,dr,dc);value+=.5*(weight[owners,l[0],l[1]]*m[:,l[0],l[1]]*m[:,r[0],r[1]]).sum((1,2))
    return value


def solve(cache,initial,watchdog=None,selector="lexical"):
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
            j=min(ids,key=(lambda j:items[j][1]) if selector=="lexical" else (lambda j:(float(values[j]),items[j][1])))
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
