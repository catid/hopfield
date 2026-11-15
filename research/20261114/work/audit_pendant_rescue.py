"""Independent all-case H2ap intervention/state/geometry/cost audit; NumPy only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_pendant_geometry import geometry
from verify_graph_block_source import connected
from audit_connectivity_screen import path_valid


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def one(args):
    root,directory,parent,p,job=args;started=time.perf_counter();source=p['source_outputs'][job['id']];assert sha(root/source['path'])==source['sha256'];ds=p['datasets'][job['shard']];assert sha(root/ds['path'])==ds['sha256'];run=directory/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert r['protocol_sha256']==sha(directory/'protocol.json') and r['source_outputs_sha256']==source['sha256'] and sha(run/'outputs.npz')==r['outputs_sha256']
    with np.load(root/ds['path']) as z:tokens=z['tokens'];distance=z['shortest_edges']
    with np.load(root/source['path']) as z:old={k:z[k] for k in z.files}
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in z.files}
    mp=p['model_arrays'][str(job['seed'])];assert sha(root/mp['path'])==mp['sha256']
    with np.load(root/mp['path']) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    a=old['couplings'].astype(LD);initial=old['predictions'][3];simple=np.array([path_valid(t,b) for t,b in zip(tokens,initial)]);eligible=~simple&old['stopped'][3]&old['finite'][3];assert np.array_equal(simple,out['initial_simple_path']);assert r['initial_simple_paths_skipped']==int(simple.sum()) and r['new_search_inputs']==int(eligible.sum());assert np.array_equal(out['prediction'][tokens!=1],(tokens>=2)[tokens!=1])
    for newkey,oldkey in [('prediction','predictions'),('stopped','stopped'),('finite','finite'),('minimum_flip_margin','minimum_flip_margin64'),('energy','energy64')]:assert np.array_equal(out[newkey][~eligible],old[oldkey][3,~eligible]),newkey
    assert np.array_equal(out['iterations'],old['iterations'][3]+out['native_sweeps']);assert (out['block_moves'][~eligible]==0).all() and (out['native_sweeps'][~eligible]==0).all() and not out['block_cap'][~eligible].any()
    margin=np.where(tokens==1,flip(tokens,out['prediction'],a,theta,mu,ao),np.inf).reshape(256,-1).min(1);max_margin=float(abs(margin-out['minimum_flip_margin']).max());assert max_margin<1e-9;entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(tokens==1).sum((1,2));max_energy=float(abs(energy(tokens,out['prediction'],a,theta,mu,ao)+entropy-out['energy']).max());assert max_energy<1e-6;candidate_count=accepted_count=rejected_count=native_disconnections=0;max_delta=0.;total_states=0;move_cases=[]
    for batch in r['batches']:
        lo=batch['offset'];fp=run/batch['path'];assert sha(fp)==batch['sha256']
        with np.load(fp) as z:
            ix=z['intervention_indices'];history=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());trace=payload['trace']
            for k in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap','minimum_flip_margin','energy','initial_simple_path']:assert np.array_equal(z[k],out[k][lo:lo+64]),k
        assert payload['protocol_sha256']==sha(directory/'protocol.json') and payload['offset']==lo;assert np.array_equal(ix,np.flatnonzero(eligible[lo:lo+64]));assert len(history)==len(trace)<=16;tok=tokens[lo:lo+64][ix];aa=a[lo:lo+64][ix];n=len(ix);done=np.zeros(n,bool);native_fail=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);sweeps=np.zeros(n,np.int64);final=out['prediction'][lo:lo+64][ix]
        if n:assert np.array_equal(history[0],initial[lo:lo+64][ix])
        else:assert len(trace)==0
        for ri,record in enumerate(trace):
            before=history[ri];expected=[];geos=[];total_states+=n;assert np.array_equal(before[tok!=1],(tok>=2)[tok!=1])
            for i in range(n):
                if done[i] or native_fail[i] or nonfinite[i]:continue
                raw,stats=geometry(tok[i],before[i]);kept=[]
                for cells in raw:
                    changed=before[i].copy();changed.flat[list(cells)]=False
                    if connected(tok[i],changed):kept.append(cells)
                rejected_count+=len(raw)-len(kept);stats.update(connectivity_vetoed_components=len(raw)-len(kept),reachability_checks=len(raw));geos.append({'local_input':i,'candidate_count':len(kept),**stats});expected.extend((i,cells) for cells in kept)
                if not kept:done[i]=True
            assert geos==record['geometry'];assert expected==[(c['local_input'],tuple(c['cells'])) for c in record['candidates']];values=np.array([c['delta'] for c in record['candidates']]);candidate_count+=len(values);assert np.isfinite(values).all()
            for lo2 in range(0,len(expected),32):
                part=expected[lo2:lo2+32];owners=np.array([i for i,_ in part]);b=before[owners];z=b.copy()
                for j,(_,cells) in enumerate(part):z[j].flat[list(cells)]=False
                actual=energy(tok[owners],z,aa[owners],theta,mu,ao)-energy(tok[owners],b,aa[owners],theta,mu,ao);err=float(abs(actual-values[lo2:lo2+len(part)]).max());assert err<1e-8;max_delta=max(max_delta,err)
            chosen=[]
            for i in range(n):
                ids=[j for j,(owner,_) in enumerate(expected) if owner==i]
                if not ids:continue
                j=min(ids,key=lambda j:(float(values[j]),expected[j][1]))
                if values[j]<-1e-8:chosen.append(j)
                else:done[i]=True
            assert chosen==record['chosen'];accepted_count+=len(chosen);after_block=before.copy();chosen_inputs=[]
            for j in chosen:
                i,cells=expected[j];assert all(tok[i].flat[f]==1 and before[i].flat[f] for f in cells);after_block[i].flat[list(cells)]=False;assert connected(tok[i],after_block[i]);moves[i]+=1;chosen_inputs.append(i)
            after=history[ri+1] if ri+1<len(history) else final;assert np.array_equal(after[tok!=1],(tok>=2)[tok!=1])
            if chosen:
                assert [x['local_input'] for x in record['native_results']]==chosen_inputs;change=energy(tok,after,aa,theta,mu,ao)-energy(tok,after_block,aa,theta,mu,ao);assert np.all(change<=1e-8)
                for native in record['native_results']:
                    i=native['local_input'];native_disconnections+=not connected(tok[i],after[i]);sweeps[i]+=native['sweeps'];native_fail[i]|=not native['stopped'];nonfinite[i]|=not native['finite'];assert 1<=native['sweeps']<=512
                    if native['stopped']:assert flip(tok[i:i+1],after[i:i+1],aa[i:i+1],theta,mu,ao)[tok[i:i+1]==1].min()>=-1e-8
                    else:assert native['sweeps']==512
                for i in range(n):
                    if i not in chosen_inputs:assert np.array_equal(after[i],before[i])
            else:assert np.array_equal(after,before) and ri==len(history)-1
        assert np.array_equal(moves,out['block_moves'][lo:lo+64][ix]) and np.array_equal(sweeps,out['native_sweeps'][lo:lo+64][ix]);assert np.array_equal(native_fail,out['native_cap'][lo:lo+64][ix]);assert np.array_equal(~done&~native_fail&~nonfinite,out['block_cap'][lo:lo+64][ix]);assert np.array_equal(done&~native_fail&~nonfinite,out['stopped'][lo:lo+64][ix]);assert np.array_equal(~nonfinite,out['finite'][lo:lo+64][ix]);move_cases.extend((lo+ix[moves>0]).tolist())
    success=[];disconnected=[]
    for v in range(5):
        if v<4:boards=old['predictions'][v];stopped=old['stopped'][v];finite=old['finite'][v];marg=old['minimum_flip_margin64'][v]
        else:boards=out['prediction'];stopped=out['stopped'];finite=out['finite'];marg=out['minimum_flip_margin']
        shortest=np.array([path_valid(t,b) and b.sum()==d+1 for t,b,d in zip(tokens,boards,distance)]);success.append(shortest&stopped&finite&(marg>=-1e-6)&old['hidden_good']);disconnected.append(sum(not connected(t,b) for t,b in zip(tokens,boards)))
    assert np.all(simple[success[3]]);assert np.array_equal(initial[success[3]],out['prediction'][success[3]])
    assert r['charged_answer_seconds']==source['archived_answer_prefix_seconds']+r['new_incremental_answer_seconds'];assert r['new_incremental_answer_seconds']==r['selector_seconds']+r['search_seconds'];assert r['archived_shared_hidden_seconds']==source['archived_shared_hidden_seconds']
    return {'job':job,'success':[s.astype(int).tolist() for s in success],'initial_simple_paths':int(simple.sum()),'new_search_inputs':int(eligible.sum()),'all_final_states':256,'intermediate_states':total_states,'disconnected_by_arm':disconnected,'candidate_literal_checks':candidate_count,'vetoed_candidates':rejected_count,'accepted_moves':accepted_count,'inputs_with_moves':move_cases,'native_descent_disconnections':int(native_disconnections),'maximum_literal_error':max_delta,'maximum_final_margin_error':max_margin,'maximum_final_energy_error':max_energy,'native_caps':int(out['native_cap'].sum()),'block_caps':int(out['block_cap'].sum()),'nonfinite':int((~out['finite']).sum()),'charged_answer_seconds':r['charged_answer_seconds'],'new_incremental_answer_seconds':r['new_incremental_answer_seconds'],'archived_answer_prefix_seconds':r['archived_answer_prefix_seconds'],'archived_shared_hidden_seconds':r['archived_shared_hidden_seconds'],'seconds':time.perf_counter()-started}


def main(directory,parent,root,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text())
    for path,h in p['source_sha256'].items():assert sha(root/path)==h
    for name in ['numerical-gate.json','old-chain-fixed-point-control.json']:
        r=json.loads((directory/name).read_text());assert r['status']=='passed' and r['protocol_sha256']==sha(directory/'protocol.json')
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one,[(root,directory,parent,p,j) for j in p['jobs']]):rows.append(row);print(json.dumps({'event':'pendant_intervention_audited','job':row['job']['id']}),flush=True)
    arms=['pl','old_binary','old_blocks','connectivity_veto','pendant_rescue'];byjob={r['job']['id']:r for r in rows};joint={arm:np.concatenate([np.stack([byjob[f'shard{s:02d}-s{seed}']['success'][v] for seed in range(3)]).all(0) for s in range(16)]) for v,arm in enumerate(arms)};assert joint['connectivity_veto'].sum()==4074;gain=int((joint['pendant_rescue']&~joint['connectivity_veto']).sum());loss=int((~joint['pendant_rescue']&joint['connectivity_veto']).sum());perinput=[{'input_index':i,**{k:int(v[i]) for k,v in joint.items()}} for i in range(4096)]
    with (directory/'per-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(perinput[0]));w.writeheader();w.writerows(perinput)
    result={'status':'passed','protocol_sha256':sha(directory/'protocol.json'),'targets':{'joint_rescues':gain,'initial_joint_failures':22,'minimum11rescues':'met' if gain>=11 else 'not_met','joint_regressions':loss,'zero_regressions':'met' if loss==0 else 'not_met'},'joint_success':{k:int(v.sum()) for k,v in joint.items()},'model_success':{arm:sum(sum(r['success'][i]) for r in rows) for i,arm in enumerate(arms)},'all_final_states':sum(r['all_final_states'] for r in rows),'new_search_inputs':sum(r['new_search_inputs'] for r in rows),'initial_simple_paths':sum(r['initial_simple_paths'] for r in rows),'candidate_literal_checks':sum(r['candidate_literal_checks'] for r in rows),'accepted_moves':sum(r['accepted_moves'] for r in rows),'vetoed_candidates':sum(r['vetoed_candidates'] for r in rows),'native_descent_disconnections':sum(r['native_descent_disconnections'] for r in rows),'maximum_literal_error':max(r['maximum_literal_error'] for r in rows),'maximum_final_margin_error':max(r['maximum_final_margin_error'] for r in rows),'maximum_final_energy_error':max(r['maximum_final_energy_error'] for r in rows),'native_caps':sum(r['native_caps'] for r in rows),'block_caps':sum(r['block_caps'] for r in rows),'nonfinite':sum(r['nonfinite'] for r in rows),'charged_answer_seconds':sum(r['charged_answer_seconds'] for r in rows),'new_incremental_answer_seconds':sum(r['new_incremental_answer_seconds'] for r in rows),'archived_answer_prefix_seconds':sum(r['archived_answer_prefix_seconds'] for r in rows),'archived_shared_hidden_seconds':sum(r['archived_shared_hidden_seconds'] for r in rows),'rows':rows,'workers':workers,'seconds':time.perf_counter()-started,'script_sha256':sha(Path(__file__)),'scope':'All4096previouslyseeninputs/all3models. PureNumPy independent outcomes, simplepathidentity, completebrutecutgeometry/witnesses/candidates/choices, originalenergydeletions,margins,flags/native transitions/costs. Zero freshinputs; no generalization/confidence claim. Identity oncorrectsimplepaths proof separate from measured rescues.'};assert result['seconds']<300;(directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('parent',type=Path);a.add_argument('root',type=Path);a.add_argument('--workers',type=int,default=16);v=a.parse_args();main(v.directory,v.parent,v.root,v.workers)
