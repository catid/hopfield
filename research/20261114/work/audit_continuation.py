"""H2bg independent NumPy all-candidate, transition, flag and output audit."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import numpy as np
import study as S
from verify_training_feasibility import energy,flip,LD
from verify_large_cycle_geometry import geometry
from verify_cycle_cut_geometry import cycle_rank
from verify_graph_block_source import connected
from audit_connectivity_screen import path_valid
from local_factor_delta import deltas

SOURCE=S.OUT/'loop80_transfer';DEST=S.OUT/'loop80_continuation'
ARMS=['total64_limit256','total96_limit256','total96_limit512']


def job_check(job):
    started=time.perf_counter();run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert r['protocol_sha256']==S.sha(DEST/'protocol.json') and S.sha(run/'outputs.npz')==r['outputs_sha256'];assert S.sha(S.ROOT/job['path'])==job['sha256']
    with np.load(S.ROOT/job['path']) as z:d={k:z[k] for k in z.files}
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in z.files}
    ex=json.loads((SOURCE/'model-extraction.json').read_text());ref=next(v for v in ex['records'] if v['seed']==job['seed'] and v['arm']=='old_binary');mp=SOURCE/ref['path'];assert S.sha(mp)==ref['sha256']
    with np.load(mp) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    manifest=json.loads((SOURCE/'datasets.json').read_text());sourceaudit=json.loads((SOURCE/'audit.json').read_text());source_rows={r['job']['id']:r for r in sourceaudit['rows']}
    for shard in sorted(set((d['index']//256).tolist())):
        positions=np.flatnonzero(d['index']//256==shard);local=d['index'][positions]%256;ds=manifest['datasets'][shard];jid=f'shard{shard:02d}-s{job["seed"]}';sr=json.loads((SOURCE/'runs'/jid/'result.json').read_text());assert S.sha(S.ROOT/ds['path'])==ds['sha256'] and S.sha(SOURCE/'runs'/jid/'outputs.npz')==sr['outputs_sha256']
        with np.load(S.ROOT/ds['path']) as z:assert np.array_equal(d['tokens'][positions],z['tokens'][local]) and np.array_equal(d['distance'][positions],z['shortest_edges'][local])
        with np.load(SOURCE/'runs'/jid/'outputs.npz') as z:
            assert z['block_cap'][3,local].all() and z['finite'][5,local].all() and not z['block_native_cap'][3,local].any() and (z['block_moves'][1:,local].sum(0)==48).all();assert not np.array(source_rows[jid]['success'][5])[local].any()
            for key,sourcekey in [('board','predictions'),('iterations','iterations')]:assert np.array_equal(d[key][positions],z[sourcekey][5,local])
            for key in ['couplings','hidden_good']:assert np.array_equal(d[key][positions],z[key][local])
    t=d['tokens'];a=d['couplings'].astype(LD);n=len(t);success=[];margerr=energyerr=costerr=0.;candidate_count=move_count=0;progress=[];stage_choices={};stage_caps={};terminal_count=0
    for arm in range(3):
        board=out['prediction'][arm];marg=np.where(t==1,flip(t,board,a,theta,mu,ao),np.inf).reshape(n,-1).min(1);err=float(abs(marg-out['margin'][arm]).max());assert err<1e-9;margerr=max(margerr,err)
        # Published BinaryQuadratic model energy includes the tiny binary entropy convention.
        calculated=energy(t,board,a,theta,mu,ao)+LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));offset=out['energy'][arm]-calculated;assert np.isfinite(offset).all();energyerr=max(energyerr,float(abs(offset).max()));assert np.max(abs(offset))<1e-6
        valid=np.array([path_valid(tok,b) for tok,b in zip(t,board)]);shortest=valid&(board.sum((1,2))==d['distance']+1);cert=out['stopped'][arm]&out['finite'][arm]&~out['native_cap'][arm]&~out['block_cap'][arm]&(marg>=-1e-6)&np.isfinite(out['energy'][arm]);success.append(shortest&cert&d['hidden_good'])
        assert np.array_equal(out['iterations'][arm],d['iterations']+out['extra_sweeps'][arm]);assert (out['extra_moves'][arm]<=([16,48,48][arm])).all()
    for rec in r['trace_archives']:
        name=rec['stage'];path=run/rec['path'];assert S.sha(path)==rec['sha256']
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
    result={'status':'passed','job':job,'cases':cases,'candidate_checks':candidate_count,'accepted_moves':move_count,'terminal_checks':terminal_count,'maximum_cost_error':costerr,'maximum_margin_error':margerr,'maximum_energy_error':energyerr,'rank_progress':progress,'candidate_caps':stage_caps,'native_restored_cycles':sum(p['native_restored_cycle'] for p in progress),'native_disconnections':sum(p['native_disconnected'] for p in progress),'native_restored_deleted_vertices':sum(p['native_restored_deleted_vertices'] for p in progress),'seconds':time.perf_counter()-started};S.save(DEST/'audit-rows'/f'{job["id"]}.json',result);return result


def main():
    started=time.perf_counter();p=json.loads((DEST/'protocol.json').read_text())
    for path,h in {**p['source_sha256'],**p['source_evidence_sha256']}.items():assert S.sha(S.ROOT/path)==h,path
    inp=json.loads((DEST/'inputs.json').read_text());assert inp['protocol_sha256']==S.sha(DEST/'protocol.json');rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(job_check,j) for j in inp['jobs']]):rows.append(f.result());assert time.perf_counter()-started<300
    source=json.loads((SOURCE/'audit.json').read_text());byjob={r['job']['id']:r for r in source['rows']};baseline=np.stack([np.concatenate([byjob[f'shard{shard:02d}-s{seed}']['success'][5] for shard in range(16)]) for seed in range(3)]).astype(bool);bp=np.stack([np.concatenate([byjob[f'shard{shard:02d}-s{seed}']['success'][0] for shard in range(16)]) for seed in range(3)]).astype(bool);full=np.stack([baseline.copy() for _ in range(3)]);cases=[c for r in sorted(rows,key=lambda r:r['job']['id']) for c in r['cases']];assert len(cases)==148 and len({(c['model_seed'],c['input_index']) for c in cases})==148
    for c in cases:assert not baseline[c['model_seed'],c['input_index']];full[:,c['model_seed'],c['input_index']]=c['success']
    assert np.all(full[:,baseline]);rescues=full.sum((1,2))-baseline.sum();joint=full.all(1).sum(1);contrast=all(c['choice_equal256_512'] and c['board_equal256_512'] and c['flags_equal256_512'] and c['success'][1]==c['success'][2] for c in cases);workers=[json.loads((DEST/f'worker{g}.json').read_text()) for g in [0,1]];assert all(w['status']=='complete' and w['seconds']<120 for w in workers)
    prefix=sum(r['source_prefix_answer_seconds'] for r in inp['source_rows']);increment=np.zeros(3);setup=0
    for job in inp['jobs']:
        receipt=json.loads((DEST/'runs'/job['id']/'result.json').read_text());tt={x['stage']:x['seconds'] for x in receipt['stage_timings']};setup+=receipt['setup_seconds'];increment+=np.array([tt['extra16_limit256'],tt['extra16_limit256']+tt.get('extra32_limit256',0),tt['extra48_limit512']])+receipt['setup_seconds']+np.array(receipt['routing_seconds'])
    result={'status':'passed','hypothesis':'H2bg','protocol_sha256':S.sha(DEST/'protocol.json'),'source_scope':'All148 archived H2be model caps /52joint failures; all4096 source rows retained. Source intervention only; zero fresh jobs/evaluations.','all_source_inputs':4096,'cases':cases,'source_model_success':int(baseline.sum()),'source_joint_success':int(baseline.all(0).sum()),'bp_source_joint_success':int(bp.all(0).sum()),'joint_success':dict(zip(ARMS,map(int,joint))),'model_rescues':dict(zip(ARMS,map(int,rescues))),'primary_all148_rescued_by96_limit256':{'rescued':int(rescues[1]),'total':148,'decision':'supported' if rescues[1]==148 else 'falsified'},'separate512_no_changed_choices_or_outcomes':{'all_cases_equal':contrast,'decision':'supported' if contrast else 'falsified'},'candidate_checks':sum(r['candidate_checks'] for r in rows),'accepted_moves':sum(r['accepted_moves'] for r in rows),'terminal_checks':sum(r['terminal_checks'] for r in rows),'native_restored_cycles':sum(r['native_restored_cycles'] for r in rows),'native_disconnections':sum(r['native_disconnections'] for r in rows),'native_restored_deleted_vertices':sum(r['native_restored_deleted_vertices'] for r in rows),'maximum_cost_error':max(r['maximum_cost_error'] for r in rows),'maximum_margin_error':max(r['maximum_margin_error'] for r in rows),'maximum_energy_error':max(r['maximum_energy_error'] for r in rows),'charged_original_prefix_seconds':prefix,'incremental_seconds':dict(zip(ARMS,increment.tolist())),'charged_answer_accounting_seconds':dict(zip(ARMS,(prefix+increment).tolist())),'timing_scope':'Archived complete source prefixes plus measured continuation setup/stages, not a rerun complete-policy latency benchmark. Input/archive/modelload outside decoder accounting. No speed endpoint.','workers':workers,'processes':8,'seconds':time.perf_counter()-started,'audit_rows':[{'path':str((DEST/'audit-rows'/f'{r["job"]["id"]}.json').relative_to(S.ROOT)),'sha256':S.sha(DEST/'audit-rows'/f'{r["job"]["id"]}.json')} for r in rows]};assert result['seconds']<300;S.save(DEST/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['cases','workers','audit_rows']},indent=2))


if __name__=='__main__':main()
