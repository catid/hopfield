"""H2ao pendant components at articulation cuts, with exact old energy scoring."""
import numpy as np
import study as S
import binary_coordinate as C
from connected_block_moves import reachable,simple_path

MAX_CELLS=256
MAX_ROUNDS=16
THRESHOLD=-1e-8


def pendant_components(tokens,board):
    """Deterministic iterative Tarjan, complete articulation-component enumeration."""
    assert board.dtype==bool and np.array_equal(board[tokens!=1],(tokens>=2)[tokens!=1]);h,w=board.shape;vertices=list(map(int,np.flatnonzero(board)));selected=set(vertices);adj={}
    for u in vertices:
        r,c=divmod(u,w);adj[u]=sorted(v for v in ([u-w] if r else [])+([u+w] if r+1<h else [])+([u-1] if c else [])+([u+1] if c+1<w else []) if v in selected)
    discovery={};low={};parent={};end={};children={u:[] for u in vertices};separated={u:[] for u in vertices};order=[];components=[];component_of={}
    for root in vertices:
        if root in discovery:continue
        start=len(order);discovery[root]=low[root]=len(order);order.append(root);parent[root]=None;stack=[(root,0)]
        while stack:
            u,index=stack[-1]
            if index<len(adj[u]):
                v=adj[u][index];stack[-1]=(u,index+1)
                if v==parent[u]:continue
                if v not in discovery:
                    parent[v]=u;children[u].append(v);discovery[v]=low[v]=len(order);order.append(v);stack.append((v,0))
                else:low[u]=min(low[u],discovery[v])
            else:
                stack.pop();end[u]=len(order);par=parent[u]
                if par is not None:
                    low[par]=min(low[par],low[u])
                    if low[u]>=discovery[par]:separated[par].append(u)
        comp=tuple(order[start:]);cid=len(components);components.append(comp)
        for u in comp:component_of[u]=cid
    endpoints=set(map(int,np.flatnonzero(tokens>=2)));articulations=[];raw=[];witness={};oversize=singletons=protected=0
    for u in vertices:
        is_cut=len(children[u])>1 if parent[u] is None else bool(separated[u])
        if not is_cut:continue
        articulations.append(u);parts=[tuple(order[discovery[v]:end[v]]) for v in separated[u]];covered={u}
        for part in parts:covered.update(part)
        remainder=tuple(v for v in components[component_of[u]] if v not in covered)
        if remainder:parts.append(remainder)
        for part in parts:
            cells=tuple(sorted(part))
            if any(v in endpoints for v in cells):protected+=1;continue
            if len(cells)==1:singletons+=1;continue
            if len(cells)>MAX_CELLS:oversize+=1;continue
            assert len(cells)>=2 and all(tokens.flat[v]==1 for v in cells);raw.append(cells);witness.setdefault(cells,[]).append(u)
    unique=sorted(witness);return unique,{'articulation_vertices':articulations,'raw_component_candidates':len(raw),'unique_component_candidates':len(unique),'duplicate_component_candidates':len(raw)-len(unique),'oversize_components':oversize,'singletons':singletons,'endpoint_components_excluded':protected,'witnesses':[{'cells':list(c),'articulations':sorted(witness[c])} for c in unique]}


def candidates(tokens,board):
    raw,stats=pendant_components(tokens,board);keep=[c for c in raw if reachable(tokens,board,c)]
    return keep,{**stats,'connectivity_vetoed_components':len(raw)-len(keep),'reachability_checks':len(raw)}


def scores(cache,p,items):
    """Delta=sum single flips + each removed internal quadratic edge once."""
    torch,_,_=S.imports();h,w=p.shape[-2:];owners=torch.tensor([item[0] for item in items],device=p.device,dtype=torch.int64);mask=np.zeros((len(items),h*w),bool)
    for j,(_,cells) in enumerate(items):mask[j,list(cells)]=True
    m=torch.tensor(mask.reshape(len(items),h,w),device=p.device);single=cache.delta(p);value=(single[owners]*m).sum((1,2))
    for (dr,dc),weight in cache.weights.items():
        l,r=C.slices(h,w,dr,dc);value+=.5*(weight[owners,l[0],l[1]]*m[:,l[0],l[1]]*m[:,r[0],r[1]]).sum((1,2))
    return value


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
