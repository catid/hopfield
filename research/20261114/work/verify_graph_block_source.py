"""Independent public H2ai graph, all-candidate literal energy and trace audit."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_hard_negative_feasibility import valid_shortest
from audit_leaf_selection import components


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(t,b):
    h,w=t.shape;degree=np.zeros((h,w),np.int8)
    for r,c in np.argwhere(b):
        for rr,cc in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
            if 0<=rr<h and 0<=cc<w:degree[rr,cc]+=1
    eligible=b&(t==1)&(degree>=1)&(degree<=2);parts=components(eligible);moves=sorted(tuple(sorted(r*w+c for r,c in part)) for part in parts if 2<=len(part)<=256)
    return moves,{'candidate_count':len(moves),'oversize_components':sum(len(part)>256 for part in parts),'singletons':sum(len(part)==1 for part in parts),'eligible_vertices':int(eligible.sum())}


def connected(t,b):
    start=tuple(np.argwhere(t==2)[0]);goal=tuple(np.argwhere(t==3)[0]);seen={start};queue=[start];h,w=t.shape
    while queue:
        r,c=queue.pop()
        for rr,cc in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
            if 0<=rr<h and 0<=cc<w and b[rr,cc] and (rr,cc) not in seen:seen.add((rr,cc));queue.append((rr,cc))
    return goal in seen


def verify(directory,training,baseline):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());assert sha(training/'training-arrays.npz')==p['training_manifest']['arrays_sha256'];rows=[];candidate_count=accepted_count=0;max_delta_error=max_final_margin_error=max_energy_error=0.
    with np.load(training/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    assert all(valid_shortest(t,y) for t,y in zip(tokens,labels))
    for seed in range(3):
        receipt=json.loads((directory/f'seed{seed}/result.json').read_text());assert receipt['status']=='complete' and receipt['protocol_sha256']==sha(directory/'protocol.json');assert sha(directory/f'seed{seed}/outputs.npz')==receipt['outputs_sha256'];assert sha(baseline/f'mined-seed{seed}.npz')==p['source_outputs'][str(seed)]['sha256']
        with np.load(baseline/f'mined-seed{seed}.npz') as z:initial=z['prediction']
        with np.load(directory/f'seed{seed}/outputs.npz') as z:final={k:z[k] for k in z.files}
        cp=baseline/f'current-cast-seed{seed}.npy';assert sha(cp)==p['coefficient_arrays'][str(seed)]['sha256'];theta=np.load(cp).astype(LD)
        with np.load(baseline/f'initial-seed{seed}.npz') as z:mu=z['mu_offsets'];ao=z['aoffs']
        ap=training/f'train-readout-seed{seed}.npz';assert sha(ap)==p['readouts'][str(seed)]['readout_sha256']
        with np.load(ap) as z:attention=z['couplings'].astype(LD)
        for batch in receipt['batches']:
            lo=batch['offset'];n=batch['n'];folder=directory/f'seed{seed}';path=folder/f'batch{lo//64:02d}.npz';tracepath=folder/f'batch{lo//64:02d}-trace.json';assert sha(path)==batch['states_sha256'] and sha(tracepath)==batch['trace_sha256']
            with np.load(path) as z:saved={k:z[k] for k in z.files}
            trace=json.loads(tracepath.read_text());assert trace['protocol_sha256']==sha(directory/'protocol.json');history=saved['states_before'];assert len(trace['trace'])==len(history)<=16 and np.array_equal(history[0],initial[lo:lo+n]);t=tokens[lo:lo+n];a=attention[lo:lo+n];done=np.zeros(n,bool);native_failure=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);sweeps=np.zeros(n,np.int64)
            for ri,record in enumerate(trace['trace']):
                before=history[ri];assert np.array_equal(before[t!=1],(t>=2)[t!=1]);expected=[];geo=[]
                for i in range(n):
                    if done[i] or native_failure[i] or nonfinite[i]:continue
                    blocks,stats=geometry(t[i],before[i]);geo.append({'local_input':i,**stats});expected.extend((i,cells) for cells in blocks)
                    if not blocks:done[i]=True
                assert geo==record['geometry'] and expected==[(r['local_input'],tuple(r['cells'])) for r in record['candidates']]
                values=np.array([r['delta'] for r in record['candidates']],np.float64);assert np.isfinite(values).all();candidate_count+=len(values)
                for start in range(0,len(expected),64):
                    part=expected[start:start+64];owners=np.array([i for i,_ in part]);old=before[owners];changed=old.copy()
                    for j,(_,cells) in enumerate(part):changed[j].flat[list(cells)]=False
                    actual=energy(t[owners],changed,a[owners],theta,mu,ao)-energy(t[owners],old,a[owners],theta,mu,ao);error=float(abs(actual-values[start:start+len(part)]).max());assert error<1e-9;max_delta_error=max(max_delta_error,error)
                chosen=[]
                for i in range(n):
                    ids=[j for j,(owner,_) in enumerate(expected) if owner==i]
                    if not ids:continue
                    j=min(ids,key=lambda j:(float(values[j]),expected[j][1]))
                    if values[j]<-1e-8:chosen.append(j)
                    else:done[i]=True
                assert chosen==record['chosen'];accepted_count+=len(chosen);after_block=before.copy();chosen_inputs=[]
                for j in chosen:
                    i,cells=expected[j];assert all(t[i].flat[f]==1 and before[i].flat[f] for f in cells);after_block[i].flat[list(cells)]=False;moves[i]+=1;chosen_inputs.append(i)
                after=history[ri+1] if ri+1<len(history) else saved['prediction'];assert np.array_equal(after[t!=1],(t>=2)[t!=1])
                if chosen:
                    results=record['native_results'];assert [r['local_input'] for r in results]==chosen_inputs
                    change=energy(t,after,a,theta,mu,ao)-energy(t,after_block,a,theta,mu,ao);assert np.all(change<=1e-8)
                    for item in results:
                        i=item['local_input'];assert 1<=item['sweeps']<=512;sweeps[i]+=item['sweeps'];native_failure[i]|=not item['stopped'];nonfinite[i]|=not item['finite']
                        if item['stopped']:assert flip(t[i:i+1],after[i:i+1],a[i:i+1],theta,mu,ao)[t[i:i+1]==1].min()>=-1e-8
                        else:assert item['sweeps']==512
                    for i in range(n):
                        if i not in chosen_inputs:assert np.array_equal(after[i],before[i])
                else:assert np.array_equal(after,before) and ri==len(history)-1
            assert np.array_equal(saved['block_moves'],moves) and np.array_equal(saved['native_sweeps'],sweeps) and np.array_equal(saved['native_cap'],native_failure) and np.array_equal(saved['block_cap'],~done&~native_failure&~nonfinite)
            assert np.array_equal(saved['stopped'],done&~native_failure&~nonfinite) and saved['finite'].all()
            delta=flip(t,saved['prediction'],a,theta,mu,ao);margin=np.where(t==1,delta,np.inf).reshape(n,-1).min(1);error=float(abs(margin-saved['upstream_minimum_flip_margin']).max());assert error<1e-10;max_final_margin_error=max(max_final_margin_error,error)
            entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));e=energy(t,saved['prediction'],a,theta,mu,ao)+entropy;error=float(abs(e-saved['energy']).max());assert error<1e-7;max_energy_error=max(max_energy_error,error)
            for key,value in final.items():assert np.array_equal(saved[key],value[lo:lo+n]),key
        valid=np.array([valid_shortest(t,b) for t,b in zip(tokens,final['prediction'])]);old_valid=np.array([valid_shortest(t,b) for t,b in zip(tokens,initial)]);success=valid&final['stopped']&final['finite']&(final['upstream_minimum_flip_margin']>=-1e-6);assert int((~old_valid).sum())==p['baseline_wrong_by_model'][seed]
        rows.append({'seed':seed,'baseline_wrong':int((~old_valid).sum()),'new_wrong':int((~success).sum()),'paired_gains':int((success&~old_valid).sum()),'paired_losses':int((~success&old_valid).sum()),'wrong_indices':np.flatnonzero(~success).tolist(),'accepted_block_moves':int(final['block_moves'].sum()),'maximum_block_moves':int(final['block_moves'].max()),'native_cap_failures':int(final['native_cap'].sum()),'block_cap_failures':int(final['block_cap'].sum()),'nonfinite':int((~final['finite']).sum()),'endpoint_disconnections':sum(not connected(t,b) for t,b in zip(tokens,final['prediction'])),'timings':{k:receipt[k] for k in ['cpu_graph_seconds','score_seconds','descent_seconds','seconds']}})
    old=sum(r['baseline_wrong'] for r in rows);new=sum(r['new_wrong'] for r in rows)
    return {'status':'passed','hypothesis':'H2ai','decision':'descriptive_target_met' if new<=290 else 'descriptive_target_not_met','rows':rows,'baseline_wrong':old,'new_wrong':new,'wrong_reduction_percent':100*(old-new)/old,'paired_gains':sum(r['paired_gains'] for r in rows),'paired_losses':sum(r['paired_losses'] for r in rows),'source_model_input_interventions':3000,'reused_source_prefixes':3000,'new_independent_test_inputs':0,'candidate_moves_literal_energy_checked':candidate_count,'accepted_moves_checked':accepted_count,'maximum_literal_delta_error':max_delta_error,'maximum_final_margin_error':max_final_margin_error,'maximum_final_energy_error':max_energy_error,'seconds':time.perf_counter()-started,'protocol_sha256':sha(directory/'protocol.json'),'verifier_sha256':sha(Path(__file__)),'scope':'All frozen source cases retained. Independently reconstruct every graph candidate, clamp and deterministic choice; every candidate delta checked against literal before/after full original energy, all native transitions nonincreasing, every final state checked. Labels used only for graph evaluation. Descriptive fixed-source result, not independent generalization.'}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('training',type=Path);a.add_argument('baseline',type=Path);v=a.parse_args();print(json.dumps(verify(v.directory,v.training,v.baseline),indent=2))
