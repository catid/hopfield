"""H2an four-arm inference: one binary prefix, two explicitly symbolic policies."""
import time
import numpy as np
import study as S
import binary_coordinate as C
import graph_block_moves as OLD
import connected_block_moves as VETO
import larger_relaxation as L

ARMS=('pl','old_binary','old_blocks','connectivity_veto')
MODEL_INDEX=(0,1,1,1)


def solve(models,doubles,t,a,rotation=0,watchdog=None):
    torch,_,_=S.imports();n=len(t);out={};prefix={};blocks={};caches={};unique=np.zeros(4);setup=np.zeros(1)
    def ensure(v):
        if v in out:return
        torch.cuda.synchronize();before=time.perf_counter()
        if v==0:
            z=L.solve(models[0],t,a,torch.full_like(a[:,0],3.),8192,0.);out[0]={'prediction':z[0],'iterations':z[1],'stopped':z[2],'finite':z[3],'final_logits':z[4]}
        else:
            st=time.perf_counter();cache=C.BinaryQuadratic(doubles[1],t,a.double());torch.cuda.synchronize();setup[0]=time.perf_counter()-st;z=C.solve(cache);prefix[1]=z;caches[1]=cache;out[1]={'prediction':z['prediction'],'iterations':z['sweeps'],'stopped':z['stopped'],'finite':z['finite']}
        torch.cuda.synchronize();unique[v]=time.perf_counter()-before
        if watchdog:watchdog()
    order=list(range(4));rotation%=4;order=order[rotation:]+order[:rotation]
    for v in order:
        if v<2:ensure(v);continue
        ensure(1);base=prefix[1];torch.cuda.synchronize();before=time.perf_counter();good=(base['stopped']&base['finite']).cpu().numpy();simple=np.zeros(n,bool)
        indices=np.flatnonzero(good);prediction=base['prediction'].clone();stopped=np.zeros(n,bool);finite=base['finite'].cpu().numpy().copy();moves=np.zeros(n,np.int64);post_sweeps=np.zeros(n,np.int64);native_cap=(~base['stopped']&base['finite']).cpu().numpy();block_cap=np.zeros(n,bool);raw=None
        if len(indices):
            if len(indices)==n:cache=caches[1];initial=base['prediction']
            else:ix=torch.tensor(indices,device=t.device);cache=C.BinaryQuadratic(doubles[1],t[ix],a[ix].double());initial=base['prediction'][ix]
            raw=(OLD if v==2 else VETO).solve(cache,initial,watchdog);prediction[indices]=raw['prediction'];stopped[indices]=raw['stopped'];finite[indices]=raw['finite'];moves[indices]=raw['block_moves'];post_sweeps[indices]=raw['native_sweeps'];native_cap[indices]=raw['native_cap'];block_cap[indices]=raw['block_cap']
        torch.cuda.synchronize();unique[v]=time.perf_counter()-before;out[v]={'prediction':prediction,'iterations':base['sweeps']+torch.tensor(post_sweeps,device=t.device),'stopped':torch.tensor(stopped,device=t.device),'finite':torch.tensor(finite,device=t.device)};blocks[v]={'valid_prefix_indices':indices,'initial_simple_path':simple,'block_moves':moves,'post_block_sweeps':post_sweeps,'native_cap':native_cap,'block_cap':block_cap,'raw':raw}
        if watchdog:watchdog()
    charged=unique.copy();charged[2:]+=unique[1]
    return {'outputs':out,'prefixes':prefix,'blocks':blocks,'timing':{'rotated_arm_order':[ARMS[v] for v in order],'unique_operation_seconds':unique.tolist(),'charged_answer_seconds':charged.tolist(),'binary_cache_setup_seconds_included':setup.tolist(),'block_cpu_graph_seconds':[blocks[v]['raw']['cpu_graph_seconds'] if blocks[v]['raw'] else 0. for v in [2,3]],'block_gpu_scoring_seconds':[blocks[v]['raw']['score_seconds'] if blocks[v]['raw'] else 0. for v in [2,3]],'block_native_descent_seconds':[blocks[v]['raw']['descent_seconds'] if blocks[v]['raw'] else 0. for v in [2,3]],'charging':'Each symbolicpolicy charged full reused binaryprefix plus full incrementalcall/selectiontime. Uniqueoperation totals count actual work once. Hidden shared, diagnostics/archive separate.'}}
