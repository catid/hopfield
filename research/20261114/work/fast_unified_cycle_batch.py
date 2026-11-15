"""H2be six unchanged policies with H2bd exact bridge geometry."""
import time
import numpy as np
import study as S
import binary_coordinate as C
import connected_block_moves as VETO
import bridge_lexical_moves as LEX
import bridge_lexical_moves as BOUND
import bridge_terminal_moves as TERM
import larger_relaxation as L

ARMS=('pl','old_binary','connectivity_veto','original_lexical16','unified16','unified48')
MODEL_INDEX=(0,1,1,1,1,1)


def solve(models,doubles,t,a,rotation=0,watchdog=None):
    torch,_,_=S.imports();n=len(t);out={};prefix={};blocks={};caches={};unique=np.zeros(6);setup=np.zeros(1);tokens=t.cpu().numpy()
    def ensure(v):
        if v in out:return
        if v>=2:ensure(v-1)
        torch.cuda.synchronize();before=time.perf_counter()
        if v==0:
            z=L.solve(models[0],t,a,torch.full_like(a[:,0],3.),8192,0.);out[0]={'prediction':z[0],'iterations':z[1],'stopped':z[2],'finite':z[3],'final_logits':z[4]}
        elif v==1:
            st=time.perf_counter();cache=C.BinaryQuadratic(doubles[1],t,a.double());torch.cuda.synchronize();setup[0]=time.perf_counter()-st;z=C.solve(cache);prefix[1]=z;caches[1]=cache;out[1]={'prediction':z['prediction'],'iterations':z['sweeps'],'stopped':z['stopped'],'finite':z['finite']}
        else:
            parent=v-1;base=out[parent];prediction=base['prediction'].clone();stopped=base['stopped'].cpu().numpy().copy();finite=base['finite'].cpu().numpy().copy();moves=np.zeros(n,np.int64);post_sweeps=np.zeros(n,np.int64);native_cap=(~base['stopped']&base['finite']).cpu().numpy() if v==2 else blocks[parent]['native_cap'].copy();block_cap=np.zeros(n,bool) if v==2 else blocks[parent]['block_cap'].copy();simple=np.zeros(n,bool);raw=None;initial_terminal=np.zeros(n,bool);terminal_records=[]
            if v<=3:
                if v==3:simple=np.array([VETO.simple_path(tok,b) for tok,b in zip(tokens,prediction.cpu().numpy().astype(bool))])
                eligible=stopped&finite&~simple
            else:
                margins=caches[1].delta(prediction).masked_fill(~caches[1].free,float('inf')).flatten(1).amin(1).cpu().numpy();stable=finite&~native_cap&(margins>=-1e-8)
                if v==4:
                    initial_terminal,terminal_records=TERM.terminal_checks(tokens,prediction.cpu().numpy().astype(bool),block_cap&stable);stopped[initial_terminal]=True;block_cap[initial_terminal]=False
                    # Ordinary states have already spent their first 16 rounds in arm3.
                    # Veto-capped states spent zero cycle rounds and are now admitted.
                    eligible=blocks[2]['block_cap']&block_cap&stable
                else:eligible=block_cap&stable
            indices=np.flatnonzero(eligible)
            if len(indices):
                if len(indices)==n:cache=caches[1];initial=prediction
                else:ix=torch.tensor(indices,device=t.device);cache=C.BinaryQuadratic(doubles[1],t[ix],a[ix].double());initial=prediction[ix]
                raw=VETO.solve(cache,initial,watchdog) if v==2 else LEX.solve(cache,initial,watchdog) if v==3 else BOUND.solve(cache,initial,watchdog,max_rounds=16 if v==4 else 32)
                if v>=4:raw=TERM.certify_final(tokens[indices],raw)
                prediction[indices]=raw['prediction'];stopped[indices]=raw['stopped'];finite[indices]=raw['finite'];moves[indices]=raw['block_moves'];post_sweeps[indices]=raw['native_sweeps'];native_cap[indices]=raw['native_cap'];block_cap[indices]=raw['block_cap']
            out[v]={'prediction':prediction,'iterations':base['iterations']+torch.tensor(post_sweeps,device=t.device),'stopped':torch.tensor(stopped,device=t.device),'finite':torch.tensor(finite,device=t.device)};blocks[v]={'parent_arm':parent,'valid_prefix_indices':indices,'initial_simple_path':simple,'initial_terminal':initial_terminal,'initial_terminal_records':terminal_records,'block_moves':moves,'post_block_sweeps':post_sweeps,'native_cap':native_cap,'block_cap':block_cap,'raw':raw}
        torch.cuda.synchronize();unique[v]=time.perf_counter()-before
        if watchdog:watchdog()
    order=list(range(6));rotation%=6;order=order[rotation:]+order[:rotation]
    for v in order:ensure(v)
    charged=unique.copy()
    for v in range(2,6):charged[v]=sum(unique[1:v+1])
    return {'outputs':out,'prefixes':prefix,'blocks':blocks,'timing':{'rotated_arm_order':[ARMS[v] for v in order],'unique_operation_seconds':unique.tolist(),'charged_answer_seconds':charged.tolist(),'binary_cache_setup_seconds_included':setup.tolist(),'block_cpu_graph_seconds':[blocks[v]['raw']['cpu_graph_seconds'] if blocks[v]['raw'] else 0. for v in [2,3,4,5]],'block_gpu_scoring_seconds':[blocks[v]['raw']['score_seconds'] if blocks[v]['raw'] else 0. for v in [2,3,4,5]],'block_native_descent_seconds':[blocks[v]['raw']['descent_seconds'] if blocks[v]['raw'] else 0. for v in [2,3,4,5]],'charging':'Every arm charged its complete binary/veto/cycle/terminal dependency chain, executed once. Unified16 includes original16 on stopped veto states and up to16 on native-stable veto caps; unified48 adds at most32. Hidden/diagnostics/archive separate.'}}
