"""Independent public H2bj routing, parity, direct-factor and graph checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_hard_negative_feasibility import valid_shortest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text())
    d=directory/'public-fixtures';m=json.loads((d/'manifest.json').read_text());ph=sha(directory/'protocol.json')
    assert ph==a['protocol_sha256']==m['protocol_sha256'];assert sha(directory/'audit.json')==m['audit_sha256']
    for name,h in m['files'].items():assert sha(d/name)==h,name
    source=json.loads((d/'source-audit.json').read_text());assert source['status']=='passed'
    source_key='artifacts/research/20261114/work/loop80_confirmation/audit.json'
    assert sha(d/'source-audit.json')==p['source_evidence_sha256'][source_key]==m['source_audit_sha256']
    with np.load(d/'routing-fields.npz') as z:f={k:z[k] for k in z.files}
    assert all(x.shape==(3,4096) for x in f.values())
    accepted=f['hidden_good']&f['finite']&f['stopped']&~f['native_cap']&~f['block_cap']&np.isfinite(f['margin'])&(f['margin']>=-1e-6)
    assert np.array_equal(accepted,f['accepted']) and np.array_equal((~accepted).astype(np.uint8),f['chosen_model'])
    actual=sorted((int(i),int(s)) for s,i in np.argwhere(~accepted));assert actual==[(4078,s) for s in range(3)]
    assert int(accepted.sum())==12285
    bp=np.zeros((3,4096),bool);lp=bp.copy()
    for row in source['rows']:
        job=row['job'];sl=slice(job['shard']*256,(job['shard']+1)*256)
        bp[job['seed'],sl]=row['success'][0];lp[job['seed'],sl]=row['success'][7]
    assert not np.any(lp&~accepted)
    primary=lp.copy();strict=lp.copy();rows=[];max_margin_error=max_energy_error=0.;fullgate=0
    for fixture in m['fixtures']:
        seed=fixture['model_seed'];i=fixture['batch_index'];index=fixture['input_index']
        def arrays(prefix):
            with np.load(d/f'{prefix}-seed{seed}.npz') as z:return {k:z[k] for k in z.files}
        ref=arrays('source-bp-batch');gate=arrays('gate');replay=arrays('replay');diag=arrays('diagnostics');context=arrays('context')
        assert set(ref)==set(gate)==set(replay)
        assert all(np.array_equal(ref[k],gate[k]) and np.array_equal(ref[k][i:i+1],replay[k]) for k in ref)
        fullgate+=len(ref['prediction']);assert len(ref['prediction'])==32
        rr=json.loads((d/f'replay-seed{seed}.json').read_text());gr=json.loads((d/f'gate-seed{seed}.json').read_text())
        assert rr['status']==gr['status']=='passed' and rr['protocol_sha256']==ph
        assert rr['output_sha256']==sha(d/f'replay-seed{seed}.npz') and gr['output_sha256']==sha(d/f'gate-seed{seed}.npz')
        assert rr['diagnostics_sha256']==sha(d/f'diagnostics-seed{seed}.npz')
        assert gr['source_batch_sha256']==fixture['source_batch_sha256']
        sj=next(j for j in p['source_jobs'] if j['id']==fixture['job_id'])
        assert sj['batch_sha256'][f'batch{fixture["batch"]:02d}.npz']==fixture['source_batch_sha256']
        assert sj['outputs_sha256']==fixture['source_outputs_sha256']
        assert rr['bp_checkpoint_sha256']==p['models'][str(seed)]['pl']['sha256']
        assert rr['lp_checkpoint_sha256']==p['models'][str(seed)]['old_binary']['sha256']
        t=context['tokens'];assert np.array_equal(t,diag['tokens']);assert valid_shortest(t[0],context['labels'][0])
        aa=diag['couplings'].astype(LD);theta=context['theta'].astype(LD);mu=context['mu_offsets'];ao=context['aoffs'];b=replay['prediction']
        assert np.array_equal(b[t!=1],(t>=2)[t!=1]);margins=[];energies=[];locations=[]
        for v in [0,1]:
            delta=np.where(t==1,flip(t,b,aa,theta[v],mu,ao),np.inf);margin=float(delta.min());margins.append(margin)
            locations.append(list(map(int,np.unravel_index(int(delta.argmin()),delta.shape)[1:])))
            entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2))
            value=float((energy(t,b,aa,theta[v],mu,ao)+entropy)[0]);energies.append(value)
            err=abs(margin-float(diag['margins'][v,0]));assert err<1e-9;max_margin_error=max(max_margin_error,err)
            err=abs(value-float(diag['energies'][v,0]));assert err<1e-6;max_energy_error=max(max_energy_error,err)
        graph=bool(valid_shortest(t[0],b[0]));own=bool(diag['hidden_good'][0] and replay['finite'][0] and replay['stopped'][0] and np.isfinite(margins[0]) and margins[0]>=-1e-6)
        dual=bool(own and np.isfinite(margins[1]) and margins[1]>=-1e-6);primary[seed,index]=own and graph;strict[seed,index]=dual and graph
        expected=next(r for r in a['all_routed_cases'] if r['model_seed']==seed)
        assert expected['input_index']==index and expected['bp_model_certificate']==own and expected['original_lp_certificate']==dual and expected['graph_valid_shortest']==graph
        assert expected['bp_original_margin']==margins[0] and expected['lp_original_margin']==margins[1]
        assert expected['minimum_flip_locations_bp_lp']==locations and expected['original_energies_bp_lp']==energies
        # The rejected LP state is retained and remains a finite native-stable cap.
        old=context['predictions'];delta=np.where(t==1,flip(t,old,aa,theta[1],mu,ao),np.inf)
        assert abs(float(delta.min())-float(context['minimum_flip_margin64'][0]))<1e-9
        assert context['block_cap'][0] and not context['native_cap'][0] and context['finite'][0] and not context['stopped'][0]
        assert not valid_shortest(t[0],old[0]);assert not accepted[seed,index]
        rows.append({'model_seed':seed,'bp_margin':margins[0],'lp_margin':margins[1],'dual_certificate':dual})
    arms={'bp_adamw_native':bp,'lp96':lp,'lp96_bp_fallback':primary,'lp96_bp_dual_certificate':strict}
    joint={k:int(x.all(0).sum()) for k,x in arms.items()};model={k:int(x.sum()) for k,x in arms.items()}
    assert joint==a['joint_success'] and model==a['model_success']
    assert a['execution_decision']==('supported' if primary.all() else 'falsified')
    assert a['cross_certificate_decision']==('supported' if all(r['dual_certificate'] for r in rows) else 'falsified')
    with (directory/'per-input.csv').open() as stream:table=list(csv.DictReader(stream))
    assert len(table)==4096
    for i,row in enumerate(table):
        assert int(row['input_index'])==i and int(row['routed_models'])==int((~accepted[:,i]).sum())
        assert all(int(row[k])==int(x[:,i].all()) for k,x in arms.items())
    with (directory/'per-model-input.csv').open() as stream:table=list(csv.DictReader(stream))
    assert len(table)==12288;seen=set()
    for row in table:
        i=int(row['input_index']);s=int(row['model_seed']);assert (s,i) not in seen;seen.add((s,i));ok=bool(accepted[s,i])
        assert int(row['lp96_accepted'])==int(ok) and int(row['routed_to_bp'])==int(not ok)
        assert row['output_model']==('LP' if ok else 'BP')
        assert row['output_checkpoint_sha256']==p['models'][str(s)]['old_binary' if ok else 'pl']['sha256']
        for k,arr in [('bp_success',bp),('lp96_success',lp),('fallback_success',primary),('strict_success',strict)]:assert int(row[k])==int(arr[s,i])
    return {'status':'passed','model_input_routes':12288,'accepted_source_lp_states':12285,'actual_subset_replays':3,'source_full_batch_parity_states':fullgate,
            'all_routed_case_margins':rows,'joint_success':joint,'model_success':model,
            'execution_decision':a['execution_decision'],'cross_certificate_decision':a['cross_certificate_decision'],
            'maximum_margin_error':max_margin_error,'maximum_energy_error':max_energy_error,
            'source_sha256':sha(Path(__file__)),'manifest_sha256':sha(d/'manifest.json'),'seconds':time.perf_counter()-started,
            'scope':'All published routing fields,96 full-batch parity states,3 actual subset replays, all direct BP/LP certificates and source outcome mapping. Accepted-state correctness inherits the complete hash-bound H2bh audit. No new accuracy evidence or global/shortestness claim from local certificates.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);print(json.dumps(verify(parser.parse_args().directory),indent=2))
