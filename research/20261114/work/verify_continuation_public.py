"""H2bg public-only independent full continuation checker; no Torch or decoders.

Reconstructs every published candidate, direct-factor cost, move and certificate.
The original H2be complete audit supplies archived source provenance and baseline
outcomes; raw original full-cohort outputs remain on Work.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_large_cycle_geometry import geometry
from verify_cycle_cut_geometry import cycle_rank
from verify_graph_block_source import connected
from audit_connectivity_screen import path_valid
from local_factor_delta import deltas
from verify_hard_negative_feasibility import valid_shortest

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def job_check(args):
    directory,job=args;DEST=Path(directory);SOURCE=DEST.parent/'loop80-transfer'
    started=time.perf_counter();run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert r['protocol_sha256']==sha(DEST/'protocol.json') and sha(run/'outputs.npz')==r['outputs_sha256'];assert sha(DEST/'inputs'/f'{job["id"]}.npz')==job['sha256']
    with np.load(DEST/'inputs'/f'{job["id"]}.npz') as z:d={k:z[k] for k in z.files}
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in z.files}
    ex=json.loads((SOURCE/'model-extraction.json').read_text());ref=next(v for v in ex['records'] if v['seed']==job['seed'] and v['arm']=='old_binary');mp=SOURCE/ref['path'];assert sha(mp)==ref['sha256']
    with np.load(mp) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    sourceaudit=json.loads((DEST.parent/'loop80-transfer-summary.json').read_text());byjob={r['job']['id']:r for r in sourceaudit['rows']}
    for index in d['index']:assert not byjob[f'shard{int(index)//256:02d}-s{job["seed"]}']['success'][5][int(index)%256]
    t=d['tokens'];a=d['couplings'].astype(LD);n=len(t);success=[];margerr=energyerr=costerr=0.;candidate_count=move_count=0;progress=[];stage_choices={};stage_caps={};terminal_count=0
    for arm in range(3):
        board=out['prediction'][arm];marg=np.where(t==1,flip(t,board,a,theta,mu,ao),np.inf).reshape(n,-1).min(1);err=float(abs(marg-out['margin'][arm]).max());assert err<1e-9;margerr=max(margerr,err)
        # Published BinaryQuadratic model energy includes the tiny binary entropy convention.
        calculated=energy(t,board,a,theta,mu,ao)+LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));offset=out['energy'][arm]-calculated;assert np.isfinite(offset).all();energyerr=max(energyerr,float(abs(offset).max()));assert np.max(abs(offset))<1e-6
        valid=np.array([path_valid(tok,b) for tok,b in zip(t,board)]);shortest=np.array([valid_shortest(tok,b) for tok,b in zip(t,board)]);assert np.array_equal(shortest,valid&(board.sum((1,2))==d['distance']+1));cert=out['stopped'][arm]&out['finite'][arm]&~out['native_cap'][arm]&~out['block_cap'][arm]&(marg>=-1e-6)&np.isfinite(out['energy'][arm]);success.append(shortest&cert&d['hidden_good'])
        assert np.array_equal(out['iterations'][arm],d['iterations']+out['extra_sweeps'][arm]);assert (out['extra_moves'][arm]<=([16,48,48][arm])).all()
    for rec in r['trace_archives']:
        name=rec['stage'];path=run/rec['path'];assert sha(path)==rec['sha256']
        with np.load(path) as z:ix=z['indices'];hist=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes());prestop=z['pre_terminal_stopped'];precap=z['pre_terminal_block_cap']
        v=0 if name=='extra16_limit256' else 1 if name=='extra32_limit256' else 2;limit=512 if v==2 else 256;maxrounds=[16,32,48][v];tok=t[ix];aa=a[ix];initial=d['board'][ix] if v!=1 else out['prediction'][0,ix];final=out['prediction'][v,ix];nn=len(ix);done=np.zeros(nn,bool);native_fail=np.zeros(nn,bool);nonfinite=np.zeros(nn,bool);moves=np.zeros(nn,int);sweeps=np.zeros(nn,int);choices=[[] for _ in ix];caps=[]
        assert len(hist)==len(trace['trace'])<=maxrounds and np.array_equal(hist[0],initial)
        if v==1:
            eligible=out['block_cap'][0]&out['finite'][0]&~out['native_cap'][0]&(out['margin'][0]>=-1e-8);assert np.array_equal(ix,np.flatnonzero(eligible));assert all(np.array_equal(out[k][0,~eligible],out[k][1,~eligible]) for k in out if k!='energy')
        else:assert np.array_equal(ix,np.arange(n))
        for ri,record in enumerate(trace['trace']):
            before=hist[ri];expected=[];geo=[]
            for i in range(nn):
                if done[i] or native_fail[i] or nonfinite[i]:continue
                cells,stats=geometry(tok[i],before[i],limit);expected.extend((i,c) for c in cells);geo.append({'local_input':i,'candidate_count':len(cells),**stats});caps.append({'round':ri,'local_input':int(ix[i]),'oversize_components':stats['oversize_components'],'candidate_count':len(cells),'maximum_candidate_size':max(map(len,cells),default=0)})
                if not cells:done[i]=True
            assert geo==record['geometry'] and expected==[(q['local_input'],tuple(q['cells'])) for q in record['candidates']];values=np.array([q['delta'] for q in record['candidates']]);assert np.isfinite(values).all();candidate_count+=len(values)
            for lo in range(0,len(expected),32):
                literal=deltas(tok,before,aa,theta,mu,ao,expected[lo:lo+32]);err=float(abs(literal-values[lo:lo+len(literal)]).max());assert err<1e-8;costerr=max(costerr,err)
            chosen=[]
            for i in range(nn):
                ids=[k for k,(owner,_) in enumerate(expected) if owner==i]
                if ids:chosen.append(min(ids,key=lambda k:expected[k][1]))
            assert chosen==record['chosen'];changed=before.copy();chosen_inputs=[]
            for k in chosen:
                i,cells=expected[k];changed[i].flat[list(cells)]=False;assert connected(tok[i],changed[i]);assert cycle_rank(changed[i])==cycle_rank(before[i])-1;choices[i].append(list(cells));moves[i]+=1;chosen_inputs.append(i)
            after=hist[ri+1] if ri+1<len(hist) else final;assert np.array_equal(after[tok!=1],(tok>=2)[tok!=1]);move_count+=len(chosen)
            if chosen:
                assert [v['local_input'] for v in record['native_results']]==chosen_inputs;assert (energy(tok,after,aa,theta,mu,ao)-energy(tok,changed,aa,theta,mu,ao)<=1e-8).all()
                for native in record['native_results']:
                    i=native['local_input'];sweeps[i]+=native['sweeps'];native_fail[i]|=not native['stopped'];nonfinite[i]|=not native['finite'];assert 1<=native['sweeps']<=512
                    if native['stopped']:assert flip(tok[i:i+1],after[i:i+1],aa[i:i+1],theta,mu,ao)[tok[i:i+1]==1].min()>=-1e-8
                    else:assert native['sweeps']==512
                    before_rank=cycle_rank(before[i]);block_rank=cycle_rank(changed[i]);native_rank=cycle_rank(after[i]);chosen_cells=choices[i][-1];progress.append({'stage':name,'round':ri,'local_input':int(ix[i]),'input_index':int(d['index'][ix[i]]),'rank_before':before_rank,'rank_after_block':block_rank,'rank_after_native':native_rank,'native_restored_cycle':native_rank>block_rank,'native_disconnected':not connected(tok[i],after[i]),'native_restored_deleted_vertices':int(after[i].flat[chosen_cells].sum())})
                for i in range(nn):
                    if i not in chosen_inputs:assert np.array_equal(before[i],after[i])
            else:assert np.array_equal(before,after) and ri==len(hist)-1
        raw_cap=~done&~native_fail&~nonfinite;assert np.array_equal(prestop,done&~native_fail&~nonfinite) and np.array_equal(precap,raw_cap);records=[];terminal=np.zeros(nn,bool)
        stable=out['finite'][v,ix]&~native_fail&(out['margin'][v,ix]>=-1e-8)
        for i in np.flatnonzero(raw_cap&stable):
            cells,stats=geometry(tok[i],final[i],limit);records.append({'local_input':int(i),'candidate_count':len(cells),'candidates':[list(c) for c in cells],'geometry':stats});terminal[i]=not cells;terminal_count+=1
        assert records==trace['terminal_checks'];done|=terminal
        assert np.array_equal(out['stopped'][v,ix],done&~native_fail&~nonfinite) and np.array_equal(out['block_cap'][v,ix],~done&~native_fail&~nonfinite);assert np.array_equal(out['native_cap'][v,ix],native_fail) and np.array_equal(out['finite'][v,ix],~nonfinite)
        assert np.array_equal(out['extra_moves'][v,ix]-(out['extra_moves'][0,ix] if v==1 else 0),moves);assert np.array_equal(out['extra_sweeps'][v,ix]-(out['extra_sweeps'][0,ix] if v==1 else 0),sweeps);stage_choices[name]={int(i):c for i,c in zip(ix,choices)};stage_caps[name]=caps
    choices256=[stage_choices['extra16_limit256'][i]+stage_choices.get('extra32_limit256',{}).get(i,[]) for i in range(n)];choices512=[stage_choices['extra48_limit512'][i] for i in range(n)];choice_equal=[x==y for x,y in zip(choices256,choices512)];board_equal=np.all(out['prediction'][1]==out['prediction'][2],axis=(1,2));flags_equal=np.stack([out[k][1]==out[k][2] for k in ['stopped','finite','native_cap','block_cap']]).all(0)
    if 'extra32_limit256' not in stage_choices:assert not out['block_cap'][0].any() and all(np.array_equal(out[k][0],out[k][1]) for k in out if k!='energy')
    cases=[{'input_index':int(d['index'][i]),'model_seed':job['seed'],'success':[bool(s[i]) for s in success],'choice_equal256_512':bool(choice_equal[i]),'board_equal256_512':bool(board_equal[i]),'flags_equal256_512':bool(flags_equal[i]),'extra_moves':out['extra_moves'][:,i].tolist(),'final_cycle_rank':[cycle_rank(out['prediction'][v,i]) for v in range(3)],'remaining_block_caps':out['block_cap'][:,i].tolist()} for i in range(n)]
    result={'status':'passed','job':job,'cases':cases,'candidate_checks':candidate_count,'accepted_moves':move_count,'terminal_checks':terminal_count,'maximum_cost_error':costerr,'maximum_margin_error':margerr,'maximum_energy_error':energyerr,'rank_progress':progress,'candidate_caps':stage_caps,'native_restored_cycles':sum(p['native_restored_cycle'] for p in progress),'native_disconnections':sum(p['native_disconnected'] for p in progress),'native_restored_deleted_vertices':sum(p['native_restored_deleted_vertices'] for p in progress),'seconds':time.perf_counter()-started};return result


def verify(directory,workers=8):
    started=time.perf_counter();manifest=json.loads((directory/'evidence-manifest.json').read_text());assert all(sha(directory/k)==h for k,h in manifest['files'].items());p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text());assert a['protocol_sha256']==sha(directory/'protocol.json')
    sourcepath=directory.parent/'loop80-transfer-summary.json';assert sha(sourcepath)==p['source_evidence_sha256']['artifacts/research/20261114/work/loop80_transfer/audit.json'];source=json.loads(sourcepath.read_text());inp=json.loads((directory/'inputs.json').read_text());assert inp['protocol_sha256']==a['protocol_sha256'];byjob={r['job']['id']:r for r in source['rows']}
    for gpu in [0,1]:
        gate=json.loads((directory/f'source-gate-gpu{gpu}.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==a['protocol_sha256']
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(job_check,[(str(directory),j) for j in inp['jobs']]))
    cases=[c for row in rows for c in row['cases']];identities={(c['model_seed'],c['input_index']) for c in cases};assert len(cases)==len(identities)==148
    expected={(seed,shard*256+i) for seed in range(3) for shard in range(16) for i,b in enumerate(byjob[f'shard{shard:02d}-s{seed}']['success'][5]) if not b};assert identities==expected
    baseline=np.stack([np.concatenate([byjob[f'shard{shard:02d}-s{seed}']['success'][5] for shard in range(16)]) for seed in range(3)]).astype(bool);full=np.stack([baseline.copy() for _ in range(3)])
    for c in cases:full[:,c['model_seed'],c['input_index']]=c['success']
    arms=p['arms'];joint=dict(zip(arms,map(int,full.all(1).sum(1))));rescues=dict(zip(arms,map(int,full.sum((1,2))-baseline.sum())));assert joint==a['joint_success'] and rescues==a['model_rescues'];contrast=all(c['choice_equal256_512'] and c['board_equal256_512'] and c['flags_equal256_512'] and c['success'][1]==c['success'][2] for c in cases)
    assert a['primary_all148_rescued_by96_limit256']['decision']==('supported' if rescues[arms[1]]==148 else 'falsified');assert a['separate512_no_changed_choices_or_outcomes']['decision']==('supported' if contrast else 'falsified')
    for name in ['candidate_checks','accepted_moves','terminal_checks','native_restored_cycles','native_disconnections','native_restored_deleted_vertices']:assert sum(r[name] for r in rows)==a[name]
    for row in rows:
        expectedrow=json.loads((directory/'audit-rows'/f'{row["job"]["id"]}.json').read_text());assert row['cases']==expectedrow['cases'] and row['rank_progress']==expectedrow['rank_progress'] and row['candidate_caps']==expectedrow['candidate_caps']
    prefix=sum(r['source_prefix_answer_seconds'] for r in inp['source_rows']);increment=np.zeros(3)
    for job in inp['jobs']:
        receipt=json.loads((directory/'runs'/job['id']/'result.json').read_text());tt={x['stage']:x['seconds'] for x in receipt['stage_timings']};increment+=np.array([tt['extra16_limit256'],tt['extra16_limit256']+tt.get('extra32_limit256',0),tt['extra48_limit512']])+receipt['setup_seconds']+np.array(receipt['routing_seconds'])
    assert prefix==a['charged_original_prefix_seconds'];assert np.allclose(prefix+increment,list(a['charged_answer_accounting_seconds'].values()),rtol=0,atol=1e-12)
    return {'status':'passed','all_source_rows_retained':4096,'all_source_caps_retained':148,'all_final_model_arm_states_checked':444,'candidate_checks':a['candidate_checks'],'accepted_moves':a['accepted_moves'],'terminal_checks':a['terminal_checks'],'joint_success':joint,'model_rescues':rescues,'candidate_size_contrast_all_exact':contrast,'native_restored_cycles':a['native_restored_cycles'],'source_provenance_audit_sha256':sha(sourcepath),'protocol_sha256':sha(directory/'protocol.json'),'source_sha256':sha(Path(__file__)),'evidence_manifest_sha256':sha(directory/'evidence-manifest.json'),'seconds':time.perf_counter()-started,'processes':workers,'scope':'All148 source caps and444 final states, every candidate/cost/choice/terminal flag/native transition, independent graph shortestness and original LP energies/margins. Full4096-row source outcomes reconstructed from the published hash-bound H2be audit; no fresh scientific evidence.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();print(json.dumps(verify(args.directory,args.workers),indent=2))
