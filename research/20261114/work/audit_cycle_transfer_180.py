"""Independent NumPy all-state/all-candidate H2av audit, no Torch/decoder."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_graph_block_source import geometry,connected
from verify_cycle_cut_geometry_checked import geometry as cycle_geometry,cycle_rank
from audit_connectivity_screen import path_valid
import audit as A
import csv


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def job_check(args):
    root,directory,job,p=args;started=time.perf_counter();run=directory/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert receipt['protocol_sha256']==sha(directory/'protocol.json');assert sha(run/'outputs.npz')==receipt['outputs_sha256'];manifest=json.loads((directory/'datasets.json').read_text());ds=manifest['datasets'][job['shard']];dp=root/ds['path'];assert sha(dp)==ds['sha256']
    with np.load(dp) as z:t=z['tokens'];distance=z['shortest_edges']
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in z.files}
    model=[]
    for arm in p['arms'][:2]:
        cp=directory/f'model-seed{job["seed"]}-{arm}.npz'
        with np.load(cp) as z:model.append((z['theta'].astype(LD),z['mu_offsets'],z['aoffs']))
    a=out['couplings'].astype(LD);max_margin=max_energy=max_delta=0.;candidate_count=accepted_count=vetoed=native_disconnections=0;positive_by_arm=[0,0,0];moves_by_arm=[0,0,0];restored_vertices=[0,0,0];restored_cycles=[0,0,0];final_count=0;disconnected=[]
    for v,k in enumerate([0,1,1,1,1]):
        theta,mu,ao=model[k];board=out['predictions'][v];delta=flip(t,board,a,theta,mu,ao);margin=np.where(t==1,delta,np.inf).reshape(256,-1).min(1);err=float(abs(margin-out['minimum_flip_margin64'][v]).max());assert err<1e-9;max_margin=max(max_margin,err);entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));e=energy(t,board,a,theta,mu,ao)+entropy;err=float(abs(e-out['energy64'][v]).max());assert err<1e-6;max_energy=max(max_energy,err);final_count+=256;disconnected.append(sum(not connected(tok,b) for tok,b in zip(t,board)))
    for item in receipt['trace_archives']:
        v=p['arms'].index(item['arm']);lo=item['offset'];path=run/item['path'];assert sha(path)==item['sha256'];theta,mu,ao=model[1]
        with np.load(path) as z:ix=z['valid_prefix_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())
        tok=t[lo:lo+32][ix];aa=a[lo:lo+32][ix];n=len(ix);final=out['predictions'][v,lo:lo+32][ix];done=np.zeros(n,bool);native_failure=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);sweeps=np.zeros(n,np.int64)
        assert trace['protocol_sha256']==sha(directory/'protocol.json') and len(history)==len(trace['trace'])<=16
        parent=1 if v==2 else 2;simple=np.array([path_valid(tt,bb) for tt,bb in zip(t[lo:lo+32],out['predictions'][parent,lo:lo+32])]) if v>=3 else np.zeros(32,bool);assert np.array_equal(simple,out['initial_simple_path'][v-2,lo:lo+32]);eligible=out['stopped'][parent,lo:lo+32]&out['finite'][parent,lo:lo+32]&~simple;assert np.array_equal(ix,np.flatnonzero(eligible))
        for key in ['predictions','stopped','finite']:assert np.array_equal(out[key][v,lo:lo+32][~eligible],out[key][parent,lo:lo+32][~eligible])
        if n:assert np.array_equal(history[0],out['predictions'][parent,lo:lo+32][ix])
        for ri,record in enumerate(trace['trace']):
            before=history[ri];assert np.array_equal(before[tok!=1],(tok>=2)[tok!=1]);expected=[];geo=[]
            for i in range(n):
                if done[i] or native_failure[i] or nonfinite[i]:continue
                if v==2:
                    raw,stats=geometry(tok[i],before[i]);blocks=[]
                    for cells in raw:
                        changed=before[i].copy();changed.flat[list(cells)]=False
                        if connected(tok[i],changed):blocks.append(cells)
                    stats.update(candidate_count=len(blocks),unfiltered_candidate_count=len(raw),connectivity_vetoed_components=len(raw)-len(blocks),reachability_checks=len(raw))
                else:blocks,stats=cycle_geometry(tok[i],before[i]);stats['candidate_count']=len(blocks)
                vetoed+=stats['connectivity_vetoed_components'];geo.append({'local_input':i,**stats});expected.extend((i,cells) for cells in blocks)
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
                j=min(ids,key=(lambda j:expected[j][1]) if v==4 else (lambda j:(float(values[j]),expected[j][1])))
                if v>=3 or values[j]<-1e-8:chosen.append(j)
                else:done[i]=True
            assert chosen==record['chosen'];accepted_count+=len(chosen);after_block=before.copy();chosen_inputs=[]
            for j in chosen:
                i,cells=expected[j];assert all(tok[i].flat[f]==1 and before[i].flat[f] for f in cells);after_block[i].flat[list(cells)]=False;assert connected(tok[i],after_block[i]);moves[i]+=1;chosen_inputs.append(i);moves_by_arm[v-2]+=1;positive_by_arm[v-2]+=int(values[j]>1e-8)
                if v>=3:assert cycle_rank(after_block[i])<cycle_rank(before[i])
            after=history[ri+1] if ri+1<len(history) else final;assert np.array_equal(after[tok!=1],(tok>=2)[tok!=1])
            if chosen:
                assert [r['local_input'] for r in record['native_results']]==chosen_inputs;change=energy(tok,after,aa,theta,mu,ao)-energy(tok,after_block,aa,theta,mu,ao);assert np.all(change<=1e-8)
                for j in chosen:
                    i,cells=expected[j];restored_vertices[v-2]+=int(after[i].flat[list(cells)].sum());restored_cycles[v-2]+=int(cycle_rank(after[i])>cycle_rank(after_block[i]))
                for native in record['native_results']:
                    i=native['local_input'];native_disconnections+=int(not connected(tok[i],after[i]));assert 1<=native['sweeps']<=512;sweeps[i]+=native['sweeps'];native_failure[i]|=not native['stopped'];nonfinite[i]|=not native['finite']
                    if native['stopped']:assert flip(tok[i:i+1],after[i:i+1],aa[i:i+1],theta,mu,ao)[tok[i:i+1]==1].min()>=-1e-8
                    else:assert native['sweeps']==512
                for i in range(n):
                    if i not in chosen_inputs:assert np.array_equal(after[i],before[i])
            else:assert np.array_equal(after,before) and ri==len(history)-1
        assert np.array_equal(out['block_moves'][v-2,lo:lo+32][ix],moves) and np.array_equal(out['post_block_sweeps'][v-2,lo:lo+32][ix],sweeps);assert np.array_equal(out['block_native_cap'][v-2,lo:lo+32][ix],native_failure);assert np.array_equal(out['block_cap'][v-2,lo:lo+32][ix],~done&~native_failure&~nonfinite);assert np.array_equal(out['stopped'][v,lo:lo+32][ix],done&~native_failure&~nonfinite);assert out['finite'][v,lo:lo+32][ix].all()
    assert out['predictions'].shape==(5,256,180,180);logits=out['native_final_logits'];assert np.array_equal(out['predictions'][:1],logits>0) and np.array_equal(out['finite'][:1],np.isfinite(logits).all((2,3)));assert np.array_equal(out['stopped'][:1],out['finite'][:1]&(out['native_minimum_flip_margin32']>=-1e-6));assert np.all(out['iterations'][:1]%16==0) and np.all((out['iterations'][:1]>=16)&(out['iterations'][:1]<=8192))
    assert np.all((out['iterations'][1]>=1)&(out['iterations'][1]<=512));assert np.all(out['binary_cache_margin'][out['stopped'][1:2]]>=-1e-8);assert abs(out['binary_cache_margin']-out['minimum_flip_margin64'][1:2]).max()<1e-9
    for v in [2,3,4]:
        parent=1 if v==2 else 2;assert np.array_equal(out['iterations'][v],out['iterations'][parent]+out['post_block_sweeps'][v-2]);assert not np.any(out['stopped'][v]&(out['block_native_cap'][v-2]|out['block_cap'][v-2]))
    unique=np.sum([x['unique_operation_seconds'] for x in receipt['batch_timings']],0);charged=unique.copy();charged[2:]+=unique[1];charged[3:]+=unique[2];assert np.allclose(charged,receipt['answer_seconds_by_arm'],rtol=0,atol=1e-12)
    success=[];outcomes=[];cache={}
    for v,arm in enumerate(p['arms']):
        valid=[]
        for i,board in enumerate(out['predictions'][v]):
            key=(i,np.packbits(board).tobytes())
            if key not in cache:cache[key]=path_valid(t[i],board)
            valid.append(cache[key])
        valid=np.array(valid);shortest=valid&(out['predictions'][v].sum((1,2))==distance+1);certified=out['stopped'][v]&out['finite'][v]&(out['minimum_flip_margin64'][v]>=-1e-6)&np.isfinite(out['energy64'][v]);success.append(shortest&certified&out['hidden_good']);outcomes.append({'job':job['id'],'seed':job['seed'],'shard':job['shard'],'arm':arm,'n':256,'solved':int(success[-1].sum()),'valid_paths':int(valid.sum()),'wrong_certified':int((certified&~shortest).sum()),'timeouts':int((~out['stopped'][v]&out['finite'][v]).sum()),'nonfinite':int((~out['finite'][v]).sum()),'hidden_failures':int((~out['hidden_good']).sum()),'answer_seconds':float(charged[v]),'unique_operation_seconds':float(unique[v]),'block_moves':int(out['block_moves'][v-2].sum()) if v>=2 else 0,'native_caps':int(out['block_native_cap'][v-2].sum()) if v>=2 else 0,'block_caps':int(out['block_cap'][v-2].sum()) if v>=2 else 0})
    return {'job':job,'success':[v.astype(int).tolist() for v in success],'outcomes':outcomes,'positive_cost_moves_by_policy':positive_by_arm,'accepted_moves_by_policy':moves_by_arm,'native_restored_vertices_by_policy':restored_vertices,'native_restored_cycles_by_policy':restored_cycles,'hidden_seconds':receipt['hidden_seconds_shared'],'diagnostic_seconds':receipt['diagnostic_seconds'],'archive_seconds':receipt['archive_seconds'],'status':'passed','all_final_states':final_count,'candidate_literal_checks':candidate_count,'accepted_moves':accepted_count,'max_literal_error':max_delta,'max_final_margin_error':max_margin,'max_final_energy_error':max_energy,'disconnected_by_arm':disconnected,'vetoed_candidates':vetoed,'veto_native_descent_disconnections':native_disconnections,'seconds':time.perf_counter()-started}


def main(directory,root,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());ex=json.loads((directory/'model-extraction.json').read_text());assert ex['protocol_sha256']==sha(directory/'protocol.json');inputs=json.loads((directory/'independent-input-audit.json').read_text());assert inputs['status']=='passed' and inputs['inputs_checked']==4096 and inputs['protocol_sha256']==sha(directory/'protocol.json')
    for path,h in p['source_sha256'].items():assert sha(root/path)==h
    for r in ex['records']:assert sha(directory/r['path'])==r['sha256'] and r['checkpoint']==p['models'][str(r['seed'])][r['arm']]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(job_check,[(root,directory,j,p) for j in p['jobs']]):rows.append(row);print(json.dumps({'event':'cycle_transfer_job_audited','job':row['job']['id']}),flush=True)
    byjob={r['job']['id']:r for r in rows};joint={arm:np.concatenate([np.stack([byjob[f'shard{s:02d}-s{seed}']['success'][v] for seed in range(3)]).all(0) for s in range(16)]) for v,arm in enumerate(p['arms'])}
    def paired(x,y):
        g=int((x&~y).sum());l=int((~x&y).sum());gl,gu=A.bounds(g,4096,alpha=.025);ll,lu=A.bounds(l,4096,alpha=.025);return {'gains':g,'losses':l,'observed_net_gain_pp':100*(g-l)/4096,'separately_one_sided95_net_lower_pp':100*(gl-lu),'separately_one_sided95_net_upper_pp':100*(gu-ll)}
    endpoints={}
    for name,d in p['primary_endpoints'].items():
        value=paired(joint[d['candidate']],joint[d['control']]);value.update(d);threshold=d['minimum_net_gain_pp'];value['decision']='supported' if value['separately_one_sided95_net_lower_pp']>=threshold else 'falsified' if value['separately_one_sided95_net_upper_pp']<threshold else 'inconclusive';endpoints[name]=value
    cases=[{'input_index':i,**{arm:int(x[i]) for arm,x in joint.items()}} for i in range(4096)];outcomes=[x for r in rows for x in r['outcomes']]
    for filename,items in [('per-input.csv',cases),('runs.csv',outcomes)]:
        with (directory/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    workers_state=[json.loads((directory/f'eval-worker{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='complete' and r['wall_seconds']<=300 for r in workers_state)
    result={'status':'passed','hypothesis':'H2av','protocol_sha256':sha(directory/'protocol.json'),'primary_endpoints':endpoints,'joint_success':{arm:int(x.sum()) for arm,x in joint.items()},'model_success':{arm:sum(x['solved'] for x in outcomes if x['arm']==arm) for arm in p['arms']},'energy_vs_lexical_descriptive':paired(joint['energy_cycle_cut'],joint['lexical_cycle_cut']),'all_final_states':sum(r['all_final_states'] for r in rows),'candidate_literal_checks':sum(r['candidate_literal_checks'] for r in rows),'accepted_moves':sum(r['accepted_moves'] for r in rows),'maximum_literal_error':max(r['max_literal_error'] for r in rows),'maximum_final_margin_error':max(r['max_final_margin_error'] for r in rows),'maximum_final_energy_error':max(r['max_final_energy_error'] for r in rows),'vetoed_candidates':sum(r['vetoed_candidates'] for r in rows),'native_descent_disconnections':sum(r['veto_native_descent_disconnections'] for r in rows),'positive_cost_moves_by_policy':np.sum([r['positive_cost_moves_by_policy'] for r in rows],0).tolist(),'accepted_moves_by_policy':np.sum([r['accepted_moves_by_policy'] for r in rows],0).tolist(),'native_restored_vertices_by_policy':np.sum([r['native_restored_vertices_by_policy'] for r in rows],0).tolist(),'native_restored_cycles_by_policy':np.sum([r['native_restored_cycles_by_policy'] for r in rows],0).tolist(),'answer_seconds_by_arm':{arm:sum(x['answer_seconds'] for x in outcomes if x['arm']==arm) for arm in p['arms']},'rows':rows,'run_rows':outcomes,'worker_receipts':workers_state,'processes':workers,'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'scope':'All61440finalstates and every candidate/choice/transition/cyclechange; independent NumPy original energies, margins, graph outcomes, prefixidentity, costs, flags. All4096freshinputs included, modeltrials joined permaze. Primarypairednetendpoints separate; no simultaneous95claim.'};assert result['seconds']<300;(directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','run_rows','worker_receipts']}),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('root',type=Path);a.add_argument('--workers',type=int,default=16);v=a.parse_args();main(v.directory,v.root,v.workers)
