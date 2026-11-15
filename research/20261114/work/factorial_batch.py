"""H2aj shared five-arm inference; block arms reuse exact binary prefixes."""
import time
import numpy as np
import study as S
import binary_coordinate as C
import graph_block_moves as B
import larger_relaxation as L

ARMS=('pl','old_binary','new_binary','old_blocks','new_blocks')
MODEL_INDEX=(0,1,2,1,2)


def solve(models,doubles,t,a,rotation=0,watchdog=None):
    torch,_,_=S.imports();n=len(t);out={};prefix={};caches={};blocks={};unique=np.zeros(5);setup=np.zeros(2)
    def ensure(v):
        if v in out:return
        torch.cuda.synchronize();before=time.perf_counter()
        if v==0:
            value=L.solve(models[0],t,a,torch.full_like(a[:,0],3.),8192,0.);out[0]={'prediction':value[0],'iterations':value[1],'stopped':value[2],'finite':value[3],'final_logits':value[4]}
        else:
            start=time.perf_counter();cache=C.BinaryQuadratic(doubles[v],t,a.double());torch.cuda.synchronize();setup[v-1]=time.perf_counter()-start;value=C.solve(cache);prefix[v]=value;caches[v]=cache;out[v]={'prediction':value['prediction'],'iterations':value['sweeps'],'stopped':value['stopped'],'finite':value['finite']}
        torch.cuda.synchronize();unique[v]=time.perf_counter()-before
        if watchdog:watchdog()
    order=list(range(5));rotation%=5;order=order[rotation:]+order[:rotation]
    for v in order:
        if v<3:ensure(v);continue
        source=v-2;ensure(source);base=prefix[source];valid=(base['stopped']&base['finite']).cpu().numpy();indices=np.flatnonzero(valid);torch.cuda.synchronize();before=time.perf_counter();prediction=base['prediction'].clone();stopped=np.zeros(n,bool);finite=base['finite'].cpu().numpy().copy();moves=np.zeros(n,np.int64);post_sweeps=np.zeros(n,np.int64);native_cap=(~base['stopped']&base['finite']).cpu().numpy();block_cap=np.zeros(n,bool);raw=None
        if len(indices):
            if len(indices)==n:cache=caches[source];initial=base['prediction']
            else:
                ix=torch.tensor(indices,device=t.device);cache=C.BinaryQuadratic(doubles[source],t[ix],a[ix].double());initial=base['prediction'][ix]
            raw=B.solve(cache,initial,watchdog);prediction[indices]=raw['prediction'];stopped[indices]=raw['stopped'];finite[indices]=raw['finite'];moves[indices]=raw['block_moves'];post_sweeps[indices]=raw['native_sweeps'];native_cap[indices]=raw['native_cap'];block_cap[indices]=raw['block_cap']
        torch.cuda.synchronize();unique[v]=time.perf_counter()-before;out[v]={'prediction':prediction,'iterations':base['sweeps']+torch.tensor(post_sweeps,device=t.device),'stopped':torch.tensor(stopped,device=t.device),'finite':torch.tensor(finite,device=t.device)};blocks[v]={'valid_prefix_indices':indices,'block_moves':moves,'post_block_sweeps':post_sweeps,'native_cap':native_cap,'block_cap':block_cap,'raw':raw}
        if watchdog:watchdog()
    charged=unique.copy();charged[3]+=unique[1];charged[4]+=unique[2]
    return {'outputs':out,'prefixes':prefix,'blocks':blocks,'timing':{'rotated_arm_order':[ARMS[v] for v in order],'unique_operation_seconds':unique.tolist(),'charged_answer_seconds':charged.tolist(),'binary_cache_setup_seconds_included':setup.tolist(),'block_cpu_graph_seconds':[blocks[v]['raw']['cpu_graph_seconds'] if blocks[v]['raw'] else 0. for v in [3,4]],'block_gpu_scoring_seconds':[blocks[v]['raw']['score_seconds'] if blocks[v]['raw'] else 0. for v in [3,4]],'block_native_descent_seconds':[blocks[v]['raw']['descent_seconds'] if blocks[v]['raw'] else 0. for v in [3,4]],'charging':'Block arms charged exact reused prefix cost plus measured full incremental block call. Unique operation times represent actual execution; charged totals intentionally include reused prefix twice across arms. Hidden once shared; archive/diagnostic separate.'}}
