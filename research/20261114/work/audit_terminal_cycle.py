"""Independent complete H2ba terminal-only/added-search audit, original local factors."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_large_cycle_geometry import geometry,cycle_rank
from verify_graph_block_source import connected
from audit_connectivity_screen import path_valid
from local_factor_delta import deltas


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def terminal_records(tokens,boards,eligible):
    terminal=np.zeros(len(tokens),bool);records=[]
    for i in np.flatnonzero(eligible):
        cells,stats=geometry(tokens[i],boards[i],256);terminal[i]=not cells;records.append({'local_input':int(i),'candidate_count':len(cells),'candidates':[list(c) for c in cells],'geometry':stats})
    return terminal,records


def one(args):
    root,directory,parent,p,job=args;started=time.perf_counter();source=p['source_outputs'][job['id']];assert sha(root/source['path'])==source['sha256'];ds=p['datasets'][job['shard']];assert sha(root/ds['path'])==ds['sha256'];run=directory/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert r['protocol_sha256']==sha(directory/'protocol.json') and r['source_outputs_sha256']==source['sha256'] and sha(run/'outputs.npz')==r['outputs_sha256']
    with np.load(root/ds['path']) as z:tokens=z['tokens'];distance=z['shortest_edges']
    with np.load(root/source['path']) as z:old={k:z[k] for k in z.files}
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in z.files}
    with np.load(parent/f'model-seed{job["seed"]}-old_binary.npz') as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    a=old['couplings'].astype(LD);initial=old['predictions'][3];eligible=old['block_cap'][1]&~old['block_cap'][0]&old['finite'][3]&~old['block_native_cap'][1]&(old['minimum_flip_margin64'][3]>=-1e-8);assert np.array_equal(eligible,out['eligible']);terminal,initial_records=terminal_records(tokens,initial,eligible);assert np.array_equal(terminal,out['initial_terminal']);search=eligible&~terminal;assert r['eligible_lexical_caps']==int(eligible.sum()) and r['terminal_only_cases']==int(terminal.sum()) and r['new_search_inputs']==int(search.sum());assert np.array_equal(out['predictions'][0],initial)
    for k in ['iterations','finite','minimum_flip_margin64','energy64']:assert np.array_equal(out[k][0],old[k][3]),k
    expected=old['stopped'][3].copy();expected[terminal]=True;assert np.array_equal(out['stopped'][0],expected);expected=old['block_cap'][1].copy();expected[terminal]=False;assert np.array_equal(out['block_cap'][0],expected);assert np.array_equal(out['native_cap'][0],old['block_native_cap'][1]);assert not out['new_block_moves'][0].any() and not out['new_native_sweeps'][0].any();assert np.array_equal(out['terminal_certified'][0],terminal)
    for key in ['predictions','iterations','stopped','finite','block_cap','native_cap','minimum_flip_margin64','energy64','new_block_moves','new_native_sweeps','terminal_certified']:assert np.array_equal(out[key][1,~search],out[key][0,~search]),key
    assert np.array_equal(out['iterations'][1],old['iterations'][3]+out['new_native_sweeps'][1]);max_margin=max_energy=0.
    for v in range(2):
        assert np.array_equal(out['predictions'][v][tokens!=1],(tokens>=2)[tokens!=1]);margin=np.where(tokens==1,flip(tokens,out['predictions'][v],a,theta,mu,ao),np.inf).reshape(256,-1).min(1);err=float(abs(margin-out['minimum_flip_margin64'][v]).max());assert err<1e-9;max_margin=max(max_margin,err);entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(tokens==1).sum((1,2));err=float(abs(energy(tokens,out['predictions'][v],a,theta,mu,ao)+entropy-out['energy64'][v]).max());assert err<1e-6;max_energy=max(max_energy,err)
    candidate_count=moves_count=positive=restored=restored_cycles=disconnections=finalcert=0;max_delta=0.
    for batch in r['batches']:
        lo=batch['offset'];path=run/batch['path'];assert sha(path)==batch['sha256']
        with np.load(path) as z:
            ix=z['search_indices'];history=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());prestop=z['pre_terminal_stopped'];precap=z['pre_terminal_block_cap']
            for k in out:
                want=out[k][lo:lo+32] if k in ['eligible','initial_terminal'] else out[k][:,lo:lo+32];assert np.array_equal(z[k],want),k
        want,records=terminal_records(tokens[lo:lo+32],initial[lo:lo+32],eligible[lo:lo+32]);assert payload['initial_terminal_checks']==records;assert np.array_equal(ix,np.flatnonzero(search[lo:lo+32]));assert payload['protocol_sha256']==sha(directory/'protocol.json') and payload['offset']==lo;trace=payload['trace'];assert len(history)==len(trace)<=16;tok=tokens[lo:lo+32][ix];aa=a[lo:lo+32][ix];n=len(ix);done=np.zeros(n,bool);nativefail=np.zeros(n,bool);nonfinite=np.zeros(n,bool);moves=np.zeros(n,np.int64);sweeps=np.zeros(n,np.int64);final=out['predictions'][1,lo:lo+32][ix]
        if n:assert np.array_equal(history[0],initial[lo:lo+32][ix])
        else:assert len(trace)==0
        for ri,record in enumerate(trace):
            before=history[ri];expected=[];stats=[]
            for i in range(n):
                if done[i] or nativefail[i] or nonfinite[i]:continue
                cells,g=geometry(tok[i],before[i],256);stats.append({'local_input':i,'candidate_count':len(cells),**g});expected.extend((i,c) for c in cells)
                if not cells:done[i]=True
            assert stats==record['geometry'] and expected==[(c['local_input'],tuple(c['cells'])) for c in record['candidates']];values=np.array([c['delta'] for c in record['candidates']]);assert np.isfinite(values).all();candidate_count+=len(values)
            if expected:
                actual=deltas(tok,before,aa,theta,mu,ao,expected);err=float(abs(actual-values).max());assert err<1e-8;max_delta=max(max_delta,err)
            chosen=[]
            for i in range(n):
                ids=[j for j,(owner,_) in enumerate(expected) if owner==i]
                if ids:chosen.append(min(ids,key=lambda j:expected[j][1]))
            assert chosen==record['chosen'];moves_count+=len(chosen);after_block=before.copy();chosen_inputs=[]
            for j in chosen:
                i,cells=expected[j];after_block[i].flat[list(cells)]=False;assert connected(tok[i],after_block[i]) and cycle_rank(after_block[i])<cycle_rank(before[i]);moves[i]+=1;positive+=int(values[j]>1e-8);chosen_inputs.append(i)
            after=history[ri+1] if ri+1<len(history) else final
            if chosen:
                assert [x['local_input'] for x in record['native_results']]==chosen_inputs;assert np.all(energy(tok,after,aa,theta,mu,ao)-energy(tok,after_block,aa,theta,mu,ao)<=1e-8)
                for j in chosen:
                    i,cells=expected[j];restored+=int(after[i].flat[list(cells)].sum());restored_cycles+=int(cycle_rank(after[i])>cycle_rank(after_block[i]))
                for native in record['native_results']:
                    i=native['local_input'];disconnections+=not connected(tok[i],after[i]);sweeps[i]+=native['sweeps'];nativefail[i]|=not native['stopped'];nonfinite[i]|=not native['finite'];assert 1<=native['sweeps']<=512
                    if native['stopped']:assert flip(tok[i:i+1],after[i:i+1],aa[i:i+1],theta,mu,ao)[tok[i:i+1]==1].min()>=-1e-8
                    else:assert native['sweeps']==512
                for i in range(n):
                    if i not in chosen_inputs:assert np.array_equal(after[i],before[i])
            else:assert np.array_equal(after,before) and ri==len(history)-1
        assert np.array_equal(prestop,done&~nativefail&~nonfinite) and np.array_equal(precap,~done&~nativefail&~nonfinite);assert np.array_equal(moves,out['new_block_moves'][1,lo:lo+32][ix]) and np.array_equal(sweeps,out['new_native_sweeps'][1,lo:lo+32][ix]);assert np.array_equal(nativefail,out['native_cap'][1,lo:lo+32][ix]);assert out['finite'][1,lo:lo+32][ix].all()
        term,records=terminal_records(tok,final,precap&~nativefail&~nonfinite&(out['minimum_flip_margin64'][1,lo:lo+32][ix]>=-1e-8));assert records==payload['final_terminal_checks'];assert np.array_equal(term,out['terminal_certified'][1,lo:lo+32][ix]);assert np.array_equal(prestop|term,out['stopped'][1,lo:lo+32][ix]) and np.array_equal(precap&~term,out['block_cap'][1,lo:lo+32][ix]);finalcert+=int(term.sum())
    success=[]
    for v in range(3):
        boards=initial if v==0 else out['predictions'][v-1];stopped=old['stopped'][3] if v==0 else out['stopped'][v-1];finite=old['finite'][3] if v==0 else out['finite'][v-1];margin=old['minimum_flip_margin64'][3] if v==0 else out['minimum_flip_margin64'][v-1];shortest=np.array([path_valid(t,b) and b.sum()==d+1 for t,b,d in zip(tokens,boards,distance)]);success.append(shortest&stopped&finite&(margin>=-1e-6)&old['hidden_good'])
    expected=[source['archived_answer_prefix_seconds']+r['selector_seconds'],source['archived_answer_prefix_seconds']+r['selector_seconds']+r['search_seconds']];assert r['charged_answer_seconds']==expected
    return {'job':job,'success':[v.astype(int).tolist() for v in success],'all_new_final_states':512,'eligible_lexical_caps':int(eligible.sum()),'initial_terminal_cases':int(terminal.sum()),'new_search_inputs':int(search.sum()),'final_terminal_certifications':finalcert,'candidate_cost_checks':candidate_count,'accepted_moves':moves_count,'positive_moves':positive,'restored_vertices':restored,'restored_cycles':restored_cycles,'native_disconnections':int(disconnections),'maximum_cost_error':max_delta,'maximum_margin_error':max_margin,'maximum_energy_error':max_energy,'remaining_caps_by_new_arm':out['block_cap'].sum(1).tolist(),'native_caps_by_new_arm':out['native_cap'].sum(1).tolist(),'nonfinite_by_new_arm':(~out['finite']).sum(1).tolist(),'charged_answer_seconds':r['charged_answer_seconds'],'selector_seconds':r['selector_seconds'],'search_seconds':r['search_seconds'],'diagnostic_seconds':r['diagnostic_seconds'],'archive_seconds':r['archive_seconds'],'seconds':time.perf_counter()-started}


def main(directory,parent,audit_source,root,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());assert sha(audit_source/'audit.json')==p['source_audit_sha256'];prior=json.loads((audit_source/'audit.json').read_text());byold={r['job']['id']:r for r in prior['rows']}
    for path,h in p['source_sha256'].items():assert sha(root/path)==h
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,(root,directory,parent,p,j)) for j in p['jobs']]
        for f in as_completed(futures):r=f.result();rows.append(r);print(json.dumps({'event':'terminal_cycle_audited','job':r['job']['id'],'completed_jobs':len(rows)}),flush=True)
    rows.sort(key=lambda r:(r['job']['shard'],r['job']['seed']));byjob={r['job']['id']:r for r in rows}
    for row in rows:assert row['success'][0]==byold[row['job']['id']]['success'][3]
    arms=['inherited_lexical256','terminal_only','terminal_then_extra16'];joint={arm:np.concatenate([np.stack([byjob[f'shard{s:02d}-s{k}']['success'][v] for k in range(3)]).all(0) for s in range(16)]) for v,arm in enumerate(arms)};old=joint[arms[0]];term=joint[arms[1]];extra=joint[arms[2]];assert old.sum()==4024 and term.sum()==4034;gain=int((extra&~term).sum());loss=int((~extra&term).sum());cases=[{'input_index':i,**{k:int(v[i]) for k,v in joint.items()}} for i in range(4096)]
    with (directory/'per-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    receipts=[json.loads((directory/f'worker{gpu}.json').read_text()) for gpu in [0,1]];assert all(r['status']=='complete' and r['wall_seconds']<=300 for r in receipts);sums=['all_new_final_states','eligible_lexical_caps','initial_terminal_cases','new_search_inputs','final_terminal_certifications','candidate_cost_checks','accepted_moves','positive_moves','restored_vertices','restored_cycles','native_disconnections','selector_seconds','search_seconds','diagnostic_seconds','archive_seconds'];maxes=['maximum_cost_error','maximum_margin_error','maximum_energy_error'];arrays=['remaining_caps_by_new_arm','native_caps_by_new_arm','nonfinite_by_new_arm','charged_answer_seconds'];r={'status':'passed','hypothesis':'H2ba','protocol_sha256':sha(directory/'protocol.json'),'source_audit_sha256':p['source_audit_sha256'],'primary':{'additional_joint_rescues':gain,'joint_regressions':loss,'at_least31_additional_rescues':'met' if gain>=31 else 'not_met','zero_regressions':'met' if loss==0 else 'not_met'},'terminal_only_descriptive':{'known_joint_rescues':int((term&~old).sum()),'known_model_certifications':27},'joint_success':{k:int(v.sum()) for k,v in joint.items()},'model_success':{k:sum(sum(row['success'][v]) for row in rows) for v,k in enumerate(arms)},'archived_comparisons':{'joint_success':prior['joint_success'],'model_success':prior['model_success'],'answer_seconds_by_arm':prior['answer_seconds_by_arm']},**{k:sum(row[k] for row in rows) for k in sums},**{k:max(row[k] for row in rows) for k in maxes},**{k:np.sum([row[k] for row in rows],0).tolist() for k in arrays},'rows':rows,'worker_receipts':receipts,'workers':workers,'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'scope':'All4096 previously seen source inputs/all3models retained. Original10 inheritedveto caps unchanged. Terminal-only bookkeeping separate from167 new searches; same original256chain/nativepolicy. Everycost/geometry/choice/flags and post-round terminalcheck independentlyverified. No freshinputs/confidenceclaim.'};assert r['seconds']<300;(directory/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['rows','worker_receipts']},indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('parent',type=Path);a.add_argument('audit_source',type=Path);a.add_argument('root',type=Path);a.add_argument('--workers',type=int,default=16);v=a.parse_args();main(v.directory,v.parent,v.audit_source,v.root,v.workers)
