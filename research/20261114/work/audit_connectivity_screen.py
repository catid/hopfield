"""Independent all-case H2ak archived mapping, reachability and energy audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_graph_block_source import geometry,connected


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def path_valid(t,b):
    selected=set(map(tuple,np.argwhere(b)));start=tuple(np.argwhere(t==2)[0]);goal=tuple(np.argwhere(t==3)[0])
    if start not in selected or goal not in selected or any(t[x]==0 for x in selected):return False
    def adj(x):return [(x[0]-1,x[1]),(x[0],x[1]+1),(x[0]+1,x[1]),(x[0],x[1]-1)]
    if any(sum(y in selected for y in adj(x))!=(1 if x in [start,goal] else 2) for x in selected):return False
    seen={start};queue=[start]
    while queue:
        for y in adj(queue.pop()):
            if y in selected and y not in seen:seen.add(y);queue.append(y)
    return seen==selected


def one(args):
    root,directory,parent,p,job=args;started=time.perf_counter();source=parent/'runs'/job['id']/'outputs.npz';assert sha(source)==p['parent_results'][job['id']]['outputs_sha256'];ds=p['datasets'][job['shard']];assert sha(root/ds['path'])==ds['sha256']
    with np.load(root/ds['path']) as z:tokens=z['tokens'];distance=z['shortest_edges'];indices=z['indices']
    with np.load(source) as z:old={k:z[k] for k in z.files}
    sr=json.loads((directory/'skip'/job['id']/'result.json').read_text());vp=directory/'veto'/job['id'];vr=json.loads((vp/'result.json').read_text());assert sha(directory/'skip'/job['id']/'outputs.npz')==sr['outputs_sha256'] and sha(vp/'outputs.npz')==vr['outputs_sha256'];assert vr['protocol_sha256']==sha(directory/'protocol.json')
    with np.load(directory/'skip'/job['id']/'outputs.npz') as z:skip={k:z[k] for k in z.files}
    with np.load(vp/'outputs.npz') as z:veto={k:z[k] for k in z.files}
    simple=np.array([path_valid(t,b) for t,b in zip(tokens,old['predictions'][1])]);selected=np.where(simple,1,3);assert np.array_equal(simple,skip['initial_simple_path']) and np.array_equal(selected,skip['selected_parent_arm'])
    for k in ['predictions','iterations','stopped','finite','minimum_flip_margin64','energy64']:assert np.array_equal(skip[k],old[k][selected,np.arange(256)]),k
    with np.load(parent/f'model-seed{job["seed"]}-old_binary.npz') as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    a=old['couplings'].astype(LD);margin=np.where(tokens==1,flip(tokens,veto['prediction'],a,theta,mu,ao),np.inf).reshape(256,-1).min(1);max_margin=float(abs(margin-veto['minimum_flip_margin']).max());assert max_margin<1e-9;entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(tokens==1).sum((1,2));max_energy=float(abs(energy(tokens,veto['prediction'],a,theta,mu,ao)+entropy-veto['energy']).max());assert max_energy<1e-6;candidate_count=accepted_count=rejected_count=native_disconnections=0;max_delta=0.
    for batch in vr['batches']:
        lo=batch['offset'];fp=vp/batch['path'];assert sha(fp)==batch['sha256']
        with np.load(fp) as z:ix=z['valid_prefix_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())['trace']
        assert np.array_equal(ix,np.flatnonzero(old['stopped'][1,lo:lo+64]&old['finite'][1,lo:lo+64]));tok=tokens[lo:lo+64][ix];aa=a[lo:lo+64][ix];n=len(ix);done=np.zeros(n,bool);native_fail=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);sweeps=np.zeros(n,np.int64);final=veto['prediction'][lo:lo+64][ix]
        if n:assert np.array_equal(history[0],old['predictions'][1,lo:lo+64][ix])
        for ri,r in enumerate(trace):
            before=history[ri];expected=[];geos=[];assert np.array_equal(before[tok!=1],(tok>=2)[tok!=1])
            for i in range(n):
                if done[i] or native_fail[i] or nonfinite[i]:continue
                raw,stats=geometry(tok[i],before[i]);kept=[]
                for cells in raw:
                    b=before[i].copy();b.flat[list(cells)]=False
                    if connected(tok[i],b):kept.append(cells)
                rejected_count+=len(raw)-len(kept);stats.update(candidate_count=len(kept),unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(kept),reachability_checks=len(raw));geos.append({'local_input':i,**stats});expected.extend((i,cells) for cells in kept)
                if not kept:done[i]=True
            assert geos==r['geometry'];assert expected==[(c['local_input'],tuple(c['cells'])) for c in r['candidates']];values=np.array([c['delta'] for c in r['candidates']]);candidate_count+=len(values)
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
            assert chosen==r['chosen'];accepted_count+=len(chosen);after_block=before.copy();chosen_inputs=[]
            for j in chosen:i,cells=expected[j];after_block[i].flat[list(cells)]=False;assert connected(tok[i],after_block[i]);moves[i]+=1;chosen_inputs.append(i)
            after=history[ri+1] if ri+1<len(history) else final;assert np.array_equal(after[tok!=1],(tok>=2)[tok!=1])
            if chosen:
                assert [x['local_input'] for x in r['native_results']]==chosen_inputs;change=energy(tok,after,aa,theta,mu,ao)-energy(tok,after_block,aa,theta,mu,ao);assert np.all(change<=1e-8)
                for native in r['native_results']:
                    i=native['local_input'];native_disconnections+=not connected(tok[i],after[i]);sweeps[i]+=native['sweeps'];native_fail[i]|=not native['stopped'];nonfinite[i]|=not native['finite'];assert 1<=native['sweeps']<=512
                    if native['stopped']:assert flip(tok[i:i+1],after[i:i+1],aa[i:i+1],theta,mu,ao)[tok[i:i+1]==1].min()>=-1e-8
                for i in range(n):
                    if i not in chosen_inputs:assert np.array_equal(after[i],before[i])
            else:assert np.array_equal(after,before)
        assert np.array_equal(moves,veto['block_moves'][lo:lo+64][ix]) and np.array_equal(sweeps,veto['native_sweeps'][lo:lo+64][ix]);assert np.array_equal(native_fail,veto['native_cap'][lo:lo+64][ix]);assert np.array_equal(~done&~native_fail&~nonfinite,veto['block_cap'][lo:lo+64][ix]);assert np.array_equal(done&~native_fail&~nonfinite,veto['stopped'][lo:lo+64][ix])
    outputs=[(old['predictions'][0],old['stopped'][0],old['finite'][0],old['minimum_flip_margin64'][0]),(old['predictions'][1],old['stopped'][1],old['finite'][1],old['minimum_flip_margin64'][1]),(old['predictions'][3],old['stopped'][3],old['finite'][3],old['minimum_flip_margin64'][3]),(skip['predictions'],skip['stopped'],skip['finite'],skip['minimum_flip_margin64']),(veto['prediction'],veto['stopped'],veto['finite'],veto['minimum_flip_margin'])];success=[];disconnected=[]
    for boards,stopped,finite,marg in outputs:
        shortest=np.array([path_valid(t,b) and b.sum()==d+1 for t,b,d in zip(tokens,boards,distance)]);success.append(shortest&stopped&finite&(marg>=-1e-6)&old['hidden_good']);disconnected.append(sum(not connected(t,b) for t,b in zip(tokens,boards)))
    return {'job':job,'success':[s.astype(int).tolist() for s in success],'initial_simple_paths':int(simple.sum()),'initial_disconnected':disconnected[1],'disconnected_by_arm':disconnected,'candidate_literal_checks':candidate_count,'vetoed_candidates':rejected_count,'accepted_moves':accepted_count,'native_descent_disconnections':native_disconnections,'maximum_literal_error':max_delta,'maximum_final_margin_error':max_margin,'maximum_final_energy_error':max_energy,'veto_binary_regressions':int((success[1]&~success[4]).sum()),'skip_binary_regressions':int((success[1]&~success[3]).sum()),'native_caps':int(veto['native_cap'].sum()),'block_caps':int(veto['block_cap'].sum()),'nonfinite':int((~veto['finite']).sum()),'seconds':time.perf_counter()-started}


def main(directory,parent,root,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one,[(root,directory,parent,p,j) for j in p['jobs']]):rows.append(row);print(json.dumps({'event':'connectivity_screen_audited','job':row['job']['id']}),flush=True)
    arms=['pl','old_binary','old_blocks','skip_simple_path','connectivity_veto'];byjob={r['job']['id']:r for r in rows};joint={arm:np.concatenate([np.stack([byjob[f'shard{s:02d}-s{seed}']['success'][v] for seed in range(3)]).all(0) for s in range(16)]) for v,arm in enumerate(arms)};original_gains=joint['old_blocks']&~joint['old_binary'];assert original_gains.sum()==63;targets={}
    for name in arms[-2:]:
        gain=int((joint[name]&~joint['old_binary']).sum());loss=int((~joint[name]&joint['old_binary']).sum());retained=int((joint[name]&original_gains).sum());targets[name]={'joint_gains':gain,'joint_losses':loss,'zero_regressions_target':'met' if loss==0 else 'not_met','retained_original_gain_inputs':retained,'original_gain_inputs':63,'retention_fraction':retained/63,'retention90percent_target':'met' if retained>=57 else 'not_met'}
    perinput=[{'input_index':i,**{k:int(v[i]) for k,v in joint.items()}} for i in range(4096)]
    with (directory/'per-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(perinput[0]));w.writeheader();w.writerows(perinput)
    summary={'status':'passed','protocol_sha256':sha(directory/'protocol.json'),'targets':targets,'joint_success':{k:int(v.sum()) for k,v in joint.items()},'model_success':{arm:sum(sum(r['success'][i]) for r in rows) for i,arm in enumerate(arms)},'candidate_literal_checks':sum(r['candidate_literal_checks'] for r in rows),'vetoed_candidates':sum(r['vetoed_candidates'] for r in rows),'accepted_moves':sum(r['accepted_moves'] for r in rows),'native_descent_disconnections':sum(r['native_descent_disconnections'] for r in rows),'initial_disconnected':sum(r['initial_disconnected'] for r in rows),'maximum_literal_error':max(r['maximum_literal_error'] for r in rows),'maximum_final_margin_error':max(r['maximum_final_margin_error'] for r in rows),'maximum_final_energy_error':max(r['maximum_final_energy_error'] for r in rows),'rows':rows,'workers':workers,'seconds':time.perf_counter()-started,'script_sha256':sha(Path(__file__)),'scope':'All4096alreadyseeninputs/all3models. Independent archivedcontrolmapping, graphoutcomes and fullveto trace/state audit. Descriptive policy-specific targets; nofreshgeneralization/confidence claim. Skip zero-loss theorem fromidentity oncorrectsimplepaths is separate frommeasured gainretention.'};(directory/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('parent',type=Path);a.add_argument('root',type=Path);a.add_argument('--workers',type=int,default=16);v=a.parse_args();main(v.directory,v.parent,v.root,v.workers)
