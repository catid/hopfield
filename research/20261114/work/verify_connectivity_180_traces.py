"""Independent NumPy all-state/all-candidate H2am audit, no Torch/decoder."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_graph_block_source import geometry,connected


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def job_check(args):
    root,directory,job,p=args;started=time.perf_counter();run=directory/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert receipt['protocol_sha256']==sha(directory/'protocol.json');assert sha(run/'outputs.npz')==receipt['outputs_sha256'];manifest=json.loads((directory/'datasets.json').read_text());ds=manifest['datasets'][job['shard']];dp=root/ds['path'];assert sha(dp)==ds['sha256']
    with np.load(dp) as z:t=z['tokens']
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in z.files}
    model=[]
    for arm in p['arms'][:2]:
        cp=directory/f'model-seed{job["seed"]}-{arm}.npz'
        with np.load(cp) as z:model.append((z['theta'].astype(LD),z['mu_offsets'],z['aoffs']))
    a=out['couplings'].astype(LD);max_margin=max_energy=max_delta=0.;candidate_count=accepted_count=vetoed=native_disconnections=0;final_count=0;disconnected=[]
    for v,k in enumerate([0,1,1,1]):
        theta,mu,ao=model[k];board=out['predictions'][v];delta=flip(t,board,a,theta,mu,ao);margin=np.where(t==1,delta,np.inf).reshape(256,-1).min(1);err=float(abs(margin-out['minimum_flip_margin64'][v]).max());assert err<1e-9;max_margin=max(max_margin,err);entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));e=energy(t,board,a,theta,mu,ao)+entropy;err=float(abs(e-out['energy64'][v]).max());assert err<1e-6;max_energy=max(max_energy,err);final_count+=256;disconnected.append(sum(not connected(tok,b) for tok,b in zip(t,board)))
    for item in receipt['trace_archives']:
        v=p['arms'].index(item['arm']);lo=item['offset'];path=run/item['path'];assert sha(path)==item['sha256'];theta,mu,ao=model[1]
        with np.load(path) as z:ix=z['valid_prefix_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())
        tok=t[lo:lo+32][ix];aa=a[lo:lo+32][ix];n=len(ix);final=out['predictions'][v,lo:lo+32][ix];done=np.zeros(n,bool);native_failure=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);sweeps=np.zeros(n,np.int64)
        assert trace['protocol_sha256']==sha(directory/'protocol.json') and len(history)==len(trace['trace'])<=16
        if n:assert np.array_equal(history[0],out['predictions'][1,lo:lo+32][ix])
        for ri,record in enumerate(trace['trace']):
            before=history[ri];assert np.array_equal(before[tok!=1],(tok>=2)[tok!=1]);expected=[];geo=[]
            for i in range(n):
                if done[i] or native_failure[i] or nonfinite[i]:continue
                blocks,stats=geometry(tok[i],before[i])
                if v==3:
                    raw=blocks;blocks=[]
                    for cells in raw:
                        changed=before[i].copy();changed.flat[list(cells)]=False
                        if connected(tok[i],changed):blocks.append(cells)
                    vetoed+=len(raw)-len(blocks);stats.update(candidate_count=len(blocks),unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(blocks),reachability_checks=len(raw))
                geo.append({'local_input':i,**stats});expected.extend((i,cells) for cells in blocks)
                if not blocks:done[i]=True
            assert geo==record['geometry'] and expected==[(r['local_input'],tuple(r['cells'])) for r in record['candidates']];values=np.array([r['delta'] for r in record['candidates']],np.float64);assert np.isfinite(values).all();candidate_count+=len(values)
            for start in range(0,len(expected),32):
                part=expected[start:start+32];owners=np.array([i for i,_ in part]);old=before[owners];changed=old.copy()
                for j,(_,cells) in enumerate(part):changed[j].flat[list(cells)]=False
                literal=energy(tok[owners],changed,aa[owners],theta,mu,ao)-energy(tok[owners],old,aa[owners],theta,mu,ao);err=float(abs(literal-values[start:start+len(part)]).max());assert err<1e-8;max_delta=max(max_delta,err)
            chosen=[]
            for i in range(n):
                ids=[j for j,(owner,_) in enumerate(expected) if owner==i]
                if not ids:continue
                j=min(ids,key=lambda j:(float(values[j]),expected[j][1]))
                if values[j]<-1e-8:chosen.append(j)
                else:done[i]=True
            assert chosen==record['chosen'];accepted_count+=len(chosen);after_block=before.copy();chosen_inputs=[]
            for j in chosen:i,cells=expected[j];assert all(tok[i].flat[f]==1 and before[i].flat[f] for f in cells);after_block[i].flat[list(cells)]=False;moves[i]+=1;chosen_inputs.append(i)
            after=history[ri+1] if ri+1<len(history) else final;assert np.array_equal(after[tok!=1],(tok>=2)[tok!=1])
            if chosen:
                assert [r['local_input'] for r in record['native_results']]==chosen_inputs;change=energy(tok,after,aa,theta,mu,ao)-energy(tok,after_block,aa,theta,mu,ao);assert np.all(change<=1e-8)
                for native in record['native_results']:
                    i=native['local_input'];native_disconnections+=int(v==3 and not connected(tok[i],after[i]));assert 1<=native['sweeps']<=512;sweeps[i]+=native['sweeps'];native_failure[i]|=not native['stopped'];nonfinite[i]|=not native['finite']
                    if native['stopped']:assert flip(tok[i:i+1],after[i:i+1],aa[i:i+1],theta,mu,ao)[tok[i:i+1]==1].min()>=-1e-8
                    else:assert native['sweeps']==512
                for i in range(n):
                    if i not in chosen_inputs:assert np.array_equal(after[i],before[i])
            else:assert np.array_equal(after,before) and ri==len(history)-1
        assert np.array_equal(out['block_moves'][v-2,lo:lo+32][ix],moves) and np.array_equal(out['post_block_sweeps'][v-2,lo:lo+32][ix],sweeps);assert np.array_equal(out['block_native_cap'][v-2,lo:lo+32][ix],native_failure);assert np.array_equal(out['block_cap'][v-2,lo:lo+32][ix],~done&~native_failure&~nonfinite);assert np.array_equal(out['stopped'][v,lo:lo+32][ix],done&~native_failure&~nonfinite);assert out['finite'][v,lo:lo+32][ix].all()
    return {'job':job['id'],'status':'passed','all_final_states':final_count,'candidate_literal_checks':candidate_count,'accepted_moves':accepted_count,'max_literal_error':max_delta,'max_final_margin_error':max_margin,'max_final_energy_error':max_energy,'disconnected_by_arm':disconnected,'vetoed_candidates':vetoed,'veto_native_descent_disconnections':native_disconnections,'seconds':time.perf_counter()-started}


def main(directory,root,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());ex=json.loads((directory/'model-extraction.json').read_text());assert ex['protocol_sha256']==sha(directory/'protocol.json')
    for r in ex['records']:assert sha(directory/r['path'])==r['sha256'] and r['checkpoint']==p['models'][str(r['seed'])][r['arm']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=[]
        for row in pool.map(job_check,[(root,directory,j,p) for j in p['jobs']]):rows.append(row);print(json.dumps({'event':'connectivity180_trace_audited',**row}),flush=True)
    result={'status':'passed','protocol_sha256':sha(directory/'protocol.json'),'rows':rows,'all_final_states':sum(r['all_final_states'] for r in rows),'candidate_literal_checks':sum(r['candidate_literal_checks'] for r in rows),'accepted_moves':sum(r['accepted_moves'] for r in rows),'maximum_literal_error':max(r['max_literal_error'] for r in rows),'maximum_final_margin_error':max(r['max_final_margin_error'] for r in rows),'maximum_final_energy_error':max(r['max_final_energy_error'] for r in rows),'vetoed_candidates':sum(r['vetoed_candidates'] for r in rows),'veto_native_descent_disconnections':sum(r['veto_native_descent_disconnections'] for r in rows),'processes':workers,'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'scope':'All49152final boards, every candidate and choice, clamps, prefixreuse, native transitions/flags, original literal longdouble energies/margins. Independent NumPy, noTorch/decoder/solver. No new independent inputs.'};(directory/'independent-trace-audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('root',type=Path);a.add_argument('--workers',type=int,default=16);v=a.parse_args();main(v.directory,v.root,v.workers)
