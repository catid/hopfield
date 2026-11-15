"""H2bk full original-prefix audit plus independent actual-subset/certificate checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from audit_confirmation96 import job_check as prefix_check,sha
from verify_training_feasibility import energy,flip,LD
from audit_connectivity_screen import path_valid
from fallback_confirmation_statistics import absolute,noninferiority


def save(path,value):
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def job_check(args):
    root,directory,job,p=args;started=time.perf_counter();base=prefix_check((root,directory,job,p,False));run=directory/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text())
    assert receipt['all_writes_closed_hashed_and_joined'] and receipt['archive_writer_metrics']['maximum_pending']<=2 and receipt['archive_writer_metrics']['maximum_rss_kib']<=8*1024**2
    assert sha(run/'hybrid-outputs.npz')==receipt['hybrid_outputs_sha256']
    with np.load(run/'outputs.npz') as z:old={k:z[k] for k in z.files}
    with np.load(run/'hybrid-outputs.npz') as z:h={k:z[k] for k in z.files}
    manifest=json.loads((directory/'datasets.json').read_text());ds=manifest['datasets'][job['shard']]
    with np.load(root/ds['path']) as z:tokens=z['tokens'];distance=z['shortest_edges']
    params=[]
    for arm in ['pl','old_binary']:
        with np.load(directory/f'model-seed{job["seed"]}-{arm}.npz') as z:params.append((z['theta'].astype(LD),z['mu_offsets'],z['aoffs']))
    accept=old['hidden_good']&old['finite'][7]&old['stopped'][7]&~old['block_cap'][5]&~old['block_native_cap'][5]&np.isfinite(old['minimum_flip_margin64'][7])&(old['minimum_flip_margin64'][7]>=-1e-6)
    assert np.array_equal(accept,h['lp_accepted']) and np.array_equal((~accept).astype(np.uint8),h['chosen_model'])
    assert np.array_equal(h['source_lp_margin'],old['minimum_flip_margin64'][7],equal_nan=True)
    for key in ['prediction','iterations','stopped','finite']:
        assert np.array_equal(h[key][accept],old['predictions' if key=='prediction' else key][7,accept])
    assert np.array_equal(h['output_own_margin'][accept],old['minimum_flip_margin64'][7,accept]) and np.array_equal(h['output_lp_margin'][accept],old['minimum_flip_margin64'][7,accept])
    assert len(receipt['subset_archives'])==len(receipt['hybrid_archives'])==len(receipt['hybrid_batch_timings'])==8
    checks=[];all_routed=[];max_margin=max_energy=0.;main_total=strict_total=0.
    for batch in range(8):
        lo=batch*32;subrecord=receipt['subset_archives'][batch];hyrecord=receipt['hybrid_archives'][batch]
        assert subrecord['offset']==hyrecord['offset']==lo
        assert sha(run/subrecord['path'])==subrecord['sha256'] and sha(run/hyrecord['path'])==hyrecord['sha256']
        with np.load(run/subrecord['path']) as z:sub={k:z[k] for k in z.files}
        with np.load(run/hyrecord['path']) as z:assert set(z.files)==set(h) and all(np.array_equal(z[k],h[k][lo:lo+32],equal_nan=True) for k in h)
        ix=sub['indices'];expected=np.flatnonzero(~accept[lo:lo+32]);assert np.array_equal(ix,expected);global_ix=ix+lo
        timing=receipt['hybrid_batch_timings'][batch];assert timing['actual_subset_inputs']==len(ix)
        values=timing['operations'];assert set(values)=={'lp_routing_certificate','routing_selection','fallback_gather','bp_subset_solve','bp_subset_cpu_return','bp_own_certificate','hybrid_mapping_and_flags','lp_cross_certificate','strict_certificate_flags'}
        assert all(np.isfinite(v) and v>=0 for v in values.values());main=sum(v for k,v in values.items() if k not in ['lp_cross_certificate','strict_certificate_flags']);strict=sum(values.values());baseline=receipt['batch_timings'][batch]['charged_answer_seconds'][7]
        assert abs(main-timing['main_added_seconds'])<1e-12 and abs(strict-timing['strict_added_seconds'])<1e-12
        assert abs(baseline+main-timing['main_charged_seconds'])<1e-12 and abs(baseline+strict-timing['strict_charged_seconds'])<1e-12
        main_total+=baseline+main;strict_total+=baseline+strict
        parity={}
        for k in ['prediction','iterations','stopped','finite']:
            assert np.array_equal(h[k][global_ix],sub[k],equal_nan=True)
            parity[k]=bool(np.array_equal(sub[k],old['predictions' if k=='prediction' else k][0,global_ix]))
        parity['final_logits']=bool(np.array_equal(sub['final_logits'],old['native_final_logits'][0,global_ix]))
        assert parity==timing['subset_vs_full_control_parity'];checks.append({'offset':lo,'n':len(ix),'parity':parity})
        if not len(ix):continue
        t=tokens[global_ix];b=sub['prediction'];aa=old['couplings'][global_ix].astype(LD)
        assert b.shape==(len(ix),180,180) and np.array_equal(b[t!=1],(t>=2)[t!=1])
        assert np.array_equal(b,sub['final_logits']>0) and np.array_equal(sub['finite'],np.isfinite(sub['final_logits']).all((1,2)))
        assert np.array_equal(sub['stopped'],sub['finite']&(sub['minimum_margin32']>=-1e-6))
        assert np.all((sub['iterations']%16==0)&(sub['iterations']>=16)&(sub['iterations']<=8192))
        for v,(theta,mu,ao) in enumerate(params):
            margins=np.where(t==1,flip(t,b,aa,theta,mu,ao),np.inf).reshape(len(ix),-1).min(1)
            entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));energies=energy(t,b,aa,theta,mu,ao)+entropy
            mk='own_margin' if v==0 else 'lp_margin';ek='own_energy' if v==0 else 'lp_energy';finite=np.isfinite(margins)&np.isfinite(sub[mk]);assert np.array_equal(np.isfinite(margins),np.isfinite(sub[mk]))
            if finite.any():err=float(abs(margins[finite]-sub[mk][finite]).max());assert err<1e-9;max_margin=max(max_margin,err)
            finite=np.isfinite(energies)&np.isfinite(sub[ek]);assert np.array_equal(np.isfinite(energies),np.isfinite(sub[ek]))
            if finite.any():err=float(abs(energies[finite]-sub[ek][finite]).max());assert err<1e-6;max_energy=max(max_energy,err)
            assert np.array_equal(h['output_own_margin' if v==0 else 'output_lp_margin'][global_ix],sub[mk],equal_nan=True)
        for i,gi in enumerate(global_ix):
            all_routed.append({'input_index':job['shard']*256+int(gi),'model_seed':job['seed'],
                'bp_own_margin':float(sub['own_margin'][i]),'cross_lp_margin':float(sub['lp_margin'][i]),
                'bp_finite':bool(sub['finite'][i]),'bp_stopped':bool(sub['stopped'][i]),
                'accepted_main':bool(h['accepted_main'][gi]),'accepted_dual':bool(h['accepted_dual'][gi]),
                'bp_control_board_equal':bool(np.array_equal(sub['prediction'][i],old['predictions'][0,gi]))})
    assert receipt['actual_subset_inputs']==len(all_routed)==int((~accept).sum())
    assert abs(main_total-receipt['main_hybrid_charged_seconds'])<1e-10 and abs(strict_total-receipt['strict_hybrid_charged_seconds'])<1e-10
    main=old['hidden_good']&h['finite']&h['stopped']&np.isfinite(h['output_own_margin'])&(h['output_own_margin']>=-1e-6)
    dual=main&np.isfinite(h['output_lp_margin'])&(h['output_lp_margin']>=-1e-6)
    assert np.array_equal(main,h['accepted_main']) and np.array_equal(dual,h['accepted_dual'])
    valid=np.array([path_valid(t,b) for t,b in zip(tokens,h['prediction'])]);shortest=valid&(h['prediction'].sum((1,2))==distance+1)
    success_main=main&shortest;success_dual=dual&shortest
    assert not np.any(np.asarray(base['success'][7],bool)&~success_main)
    row={'job':job,'prefix':base,'success':[base['success'][0],base['success'][7],success_main.astype(int).tolist(),success_dual.astype(int).tolist()],
         'routed':all_routed,'subset_control_checks':checks,'accepted_lp':int(accept.sum()),
         'wrong_certified_main':int((main&~shortest).sum()),'wrong_certified_dual':int((dual&~shortest).sum()),
         'accepted_lp_wrong_shortest':int((accept&~shortest).sum()),'bp_fallback_numerical_failures':int((~accept&~main).sum()),
         'cross_lp_rejections':int((main&~dual).sum()),'main_charged_seconds':main_total,'strict_charged_seconds':strict_total,
         'max_margin_error':max_margin,'max_energy_error':max_energy,'seconds':time.perf_counter()-started}
    save(directory/'partial-rows'/f'{job["id"]}.json',row);return row


def main(directory,root,workers):
    started=time.perf_counter();(directory/'partial-rows').mkdir(exist_ok=True);p=json.loads((directory/'protocol.json').read_text());inputs=json.loads((directory/'independent-input-audit.json').read_text());assert inputs['status']=='passed' and inputs['inputs_checked']==4096
    assert inputs['protocol_sha256']==sha(directory/'protocol.json')
    for field in ['source_sha256','source_evidence_sha256']:
        for path,h in p[field].items():assert sha(root/path)==h,path
    ex=json.loads((directory/'model-extraction.json').read_text());assert ex['protocol_sha256']==sha(directory/'protocol.json')
    for r in ex['records']:assert sha(directory/r['path'])==r['sha256'] and r['checkpoint']==p['models'][str(r['seed'])][r['arm']]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job_check,(root,directory,j,p)) for j in p['jobs']]
        for future in as_completed(futures):
            row=future.result();rows.append(row);print(json.dumps({'event':'fallback_confirmation_job_audited','job':row['job']['id'],'completed_jobs':len(rows)}),flush=True)
    rows.sort(key=lambda r:(r['job']['shard'],r['job']['seed']));byjob={r['job']['id']:r for r in rows};arms=p['reported_arms']
    model={arm:np.concatenate([np.stack([byjob[f'shard{s:02d}-s{seed}']['success'][v] for seed in range(3)]) for s in range(16)],axis=1) for v,arm in enumerate(arms)}
    joint={arm:value.all(0) for arm,value in model.items()};g=int((joint['hybrid']&~joint['pl']).sum());l=int((~joint['hybrid']&joint['pl']).sum())
    endpoints={'hybrid_absolute':absolute(int(joint['hybrid'].sum())),'strict_absolute':absolute(int(joint['dual_certificate'].sum())),'bp_noninferiority':noninferiority(g,l)}
    workers_state=[json.loads((directory/f'eval-worker{gpu}.json').read_text()) for gpu in [0,1]]
    assert all(s['status']=='complete' and s['wall_seconds']<=600 and s['peak_rss_kib']<=8*1024**2 for s in workers_state)
    bp_load=sum(s['additional_bp_load_seconds_charged_once'] for s in workers_state)
    for s in workers_state:assert bp_load>=0 and abs(s['additional_bp_load_seconds_charged_once']-sum(x['pl'] for x in s['model_load_seconds_by_seed'].values()))<1e-12
    lp_cost=sum(r['prefix']['outcomes'][7]['answer_seconds'] for r in rows);main_cost=sum(r['main_charged_seconds'] for r in rows)+bp_load;strict_cost=sum(r['strict_charged_seconds'] for r in rows)+bp_load
    cost={'lp96_charged_seconds':lp_cost,'main_hybrid_charged_seconds_including_bp_load':main_cost,'strict_hybrid_charged_seconds_including_bp_load':strict_cost,'additional_bp_model_load_seconds':bp_load,'maximum_overhead_percent':10.,'main_overhead_percent':100*(main_cost-lp_cost)/lp_cost,'strict_overhead_percent':100*(strict_cost-lp_cost)/lp_cost};cost['decision']='supported' if cost['main_overhead_percent']<=10 else 'falsified'
    cases=[{'input_index':i,**{a:int(x[i]) for a,x in joint.items()}} for i in range(4096)]
    modelcases=[{'input_index':i,'model_seed':s,'arm':a,'success':int(x[s,i])} for a,x in model.items() for s in range(3) for i in range(4096)]
    for name,values in [('per-input.csv',cases),('per-model-input.csv',modelcases)]:
        with (directory/name).open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    base=[r['prefix'] for r in rows]
    result={'status':'passed','hypothesis':'H2bk','protocol_sha256':sha(directory/'protocol.json'),'primary_endpoints':endpoints,'cost_endpoint':cost,
        'joint_success':{a:int(x.sum()) for a,x in joint.items()},'model_success':{a:int(x.sum()) for a,x in model.items()},
        'reported_model_maze_arm_evaluations':49152,'all_internal_and_reported_state_checks':122880,
        'original_prefix_states':sum(r['all_final_states'] for r in base),'actual_fallback_trajectories':sum(len(r['routed']) for r in rows),
        'accepted_lp_model_inputs_reused':sum(r['accepted_lp'] for r in rows),'all_routed_cases':[x for r in rows for x in r['routed']],
        'candidate_literal_checks':sum(r['candidate_literal_checks'] for r in base),'accepted_moves':sum(r['accepted_moves'] for r in base),
        'maximum_candidate_error':max(r['max_literal_error'] for r in base),'maximum_prefix_margin_error':max(r['max_final_margin_error'] for r in base),
        'maximum_prefix_energy_error':max(r['max_final_energy_error'] for r in base),'maximum_fallback_margin_error':max(r['max_margin_error'] for r in rows),
        'maximum_fallback_energy_error':max(r['max_energy_error'] for r in rows),
        'wrong_certified_main':sum(r['wrong_certified_main'] for r in rows),'wrong_certified_dual':sum(r['wrong_certified_dual'] for r in rows),
        'accepted_lp_wrong_shortest':sum(r['accepted_lp_wrong_shortest'] for r in rows),'bp_fallback_numerical_failures':sum(r['bp_fallback_numerical_failures'] for r in rows),
        'cross_lp_rejections':sum(r['cross_lp_rejections'] for r in rows),'rows':rows,'worker_receipts':workers_state,
        'seconds':time.perf_counter()-started,'processes':workers,'source_sha256':sha(Path(__file__)),
        'scope':'All4096 fresh inputs and three fixed models; four reported arms. All eight original prefixes/candidates/transitions independently audited; hybrid mappings and every actual BP subset checked under both original models. Separate absolute/NI/cost decisions, no familywise confidence claim and no source pooling.'}
    assert result['seconds']<=1200;save(directory/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','worker_receipts','all_routed_cases']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('root',type=Path);parser.add_argument('--workers',type=int,default=16);a=parser.parse_args();main(a.directory,a.root,a.workers)
