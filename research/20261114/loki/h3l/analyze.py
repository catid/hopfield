"""Independent output audit for the superqueen hypothesis-family boundary."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3l';OUT=ROOT/'artifacts/research/20261114/loki/h3l';PUBLIC=ROOT/'results/research/loki/h3l'
spec=importlib.util.spec_from_file_location('h3h_audit',SRC.parent/'h3h/analyze.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
HASHES={}
def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v
def read(path):sha(path);return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def true_graph():
    ids=np.arange(169);r=ids//13;c=ids%13;dr=np.abs(r[:,None]-r);dc=np.abs(c[:,None]-c)
    attack=(dr==0)|(dc==0)|(dr==dc)|((dr==1)&(dc==2))|((dr==2)&(dc==1))
    np.fill_diagonal(attack,False)
    J=13*attack.astype(np.int64)-1;np.fill_diagonal(J,0)
    return attack,J


def search_audit(arm,seed,data,keys,unused_keys,model):
    folder=OUT/f'search/{arm}/s{seed}';rec=read(folder/'result.json')
    assert rec['status'] in ['complete','derived_by_exact_identity'] and rec['protocol_sha256']==sha(SRC/'protocol.json')
    assert rec['source_sha256']==sha(SRC/'driver.py') and rec['backend_sha256']==sha(SRC.parent/'h3i/backend.py')
    assert rec['records_sha256']==sha(folder/'records.npz') and rec['model_sha256']==sha(ROOT/rec['model_path'])
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);inputs=data['test_inputs']
    assert np.array_equal(inputs,z['inputs']) and (boards>=inputs[:,None,:]).all()
    assert not any(row.tobytes() in unused_keys for samples in boards for row in samples)
    valid=np.array([[row.tobytes() in keys and row[given].all() for row in samples] for given,samples in zip(inputs,boards)])
    assert np.array_equal(valid,z['valid']);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({row.tobytes() for row in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    assert np.array_equal(found,z['found']) and np.array_equal(denom,z['denom']) and np.array_equal(z['solution_counts'],data['test_solution_counts'])
    original=z['energy'].argmin(1);selected=[]
    for energy,stopped in zip(z['energy'],z['stopped']):
        pool=np.flatnonzero(stopped)
        if not len(pool):pool=np.arange(20)
        selected.append(int(pool[energy[pool].argmin()]))
    selected=np.array(selected);ids=np.arange(len(inputs))
    assert np.array_equal(original,z['original_index']) and np.array_equal(selected,z['certificate_index'])
    assert np.array_equal(valid[ids,original],z['n20']) and np.array_equal(valid[ids,selected],z['n20_certificate']) and np.array_equal(valid[:,0],z['n1'])
    assert ((z['moves']>=0)&(z['moves']<=20000)).all() and (z['moves'][~z['stopped']]==20000).all()
    assert np.array_equal(z['flop'],57461+859*z['moves'])
    wrong=z['stopped']&~valid;identical=(boards==inputs[:,None,:]).all(-1)
    assert (boards.sum(-1)[z['stopped']]==model['q']).all()
    assert np.array_equal(z['cardinality_integer_additions'],168*(z['moves']+1))
    assert not ((z['moves']==0)&z['stopped']).any()
    assert rec['wrong_stops']==int(wrong.sum()) and rec['capped_chains']==int((~z['stopped']).sum())
    assert rec['original_accuracy_pct']==100*float(z['n20'].mean()) and rec['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())
    assert rec['coverage_pct']==100*float(found.sum()/denom.sum()) and rec['mean_moves']==float(z['moves'].mean())
    counts=boards.sum(-1);assert np.array_equal(counts,z['queen_counts'])
    occupancy={name:{label:int((mask&condition).sum()) for label,condition in [('below_q',counts<13),('equal_q',counts==13),('above_q',counts>13)]}
               for name,mask in [('stopped_invalid',wrong),('capped_invalid',~z['stopped']&~valid)]}
    J=model['S'][:,:,1,1].numpy().astype(np.float64);_,trueJ=true_graph();flat=boards.reshape(-1,169);givens=np.repeat(inputs,20,0)
    gaps=[];energies=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);fields=x@J;energies.append(.5*(x*fields).sum(1))
        margins=(1-2*x)*fields;margins[givens[lo:lo+4096]]=np.inf;gaps.append(margins.min(1)-(model['delta']-.001))
    assert np.array_equal(np.concatenate(energies).reshape(z['energy'].shape),z['energy'])
    gaps=np.concatenate(gaps)[z['stopped'].ravel()];assert (gaps>=0).all()
    wrong_rows=flat[wrong.ravel()].astype(np.float64);wrong_given=givens[wrong.ravel()]
    counter=(1-2*wrong_rows)*(wrong_rows@trueJ);counter[wrong_given]=np.inf
    counter_reject=counter.min(1)<model['delta']-.001
    assert counter_reject.all()
    diag=dict(arm=arm,seed=seed,returned_training_pattern_matches=0,invalid_occupancies=occupancy,
              strict_stopped_certificate_deficits=int((gaps<0).sum()),minimum_stopped_gap=float(gaps.min()) if len(gaps) else None,
              wrong_stops_identical_to_initial_state=int((wrong&identical).sum()),zero_move_wrong_stops=int((wrong&(z['moves']==0)).sum()),
              wrong_stops_rejected_by_supplied_true_graph=int(counter_reject.sum()),
              counterfactual_scope='Rescore identical stopped boards under the supplied target oracle; this removes false exclusions and restores missed ones, without rerunning search.')
    attack,_=true_graph();learned_forbidden=J>0;missed=attack&~learned_forbidden
    true_conflicts=.5*(wrong_rows*(wrong_rows@attack.astype(float))).sum(1)
    missed_conflicts=.5*(wrong_rows*(wrong_rows@missed.astype(float))).sum(1)
    assert np.array_equal(true_conflicts,missed_conflicts) and (true_conflicts>0).all()
    ti,tj=np.triu_indices(169,1);dr=tj//13-ti//13;dc=tj%13-ti%13
    diag['wrong_stop_true_conflict_edges']=int(true_conflicts.sum())
    diag['all_wrong_stop_true_conflicts_absent_from_learned_graph']=True
    diag['wrong_stop_conflict_edge_occurrences_by_offset']={str((int(r),int(c))):int((wrong_rows[:,ti[(dr==r)&(dc==c)]]*wrong_rows[:,tj[(dr==r)&(dc==c)]]).sum())
        for r,c in [(0,12),(12,0),(12,12),(12,-12),(1,2),(1,-2),(2,1),(2,-1)]}
    diag['selection_corrected_inputs']=int((z['n20_certificate']&~z['n20']).sum())
    diag['selection_regressed_inputs']=int((z['n20']&~z['n20_certificate']).sum())
    for name,index in [('original',original),('certificate_first',selected)]:
        bad=~valid[ids,index];selected_count=counts[ids,index];selected_stopped=z['stopped'][ids,index]
        diag[f'{name}_invalid_selected']={label:int((bad&mask).sum()) for label,mask in [
            ('below_q',selected_count<13),('equal_q',selected_count==13),('above_q',selected_count>13),('stopped',selected_stopped),('capped',~selected_stopped)]}
    row=dict(arm=arm,updates=0,seed=seed,unique_inputs=len(inputs),chains=20,
             original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),
             coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=int(wrong.sum()),capped_chains=rec['capped_chains'],
             mean_moves=rec['mean_moves'],search_seconds=rec['search_seconds'],model_sha256=rec['model_sha256'],records_sha256=rec['records_sha256'],result_sha256=sha(folder/'result.json'))
    return z,row,diag


def main():
    started=time.perf_counter();torch.set_num_threads(1);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    preflight=read(OUT/'independent-preflight.json');assert preflight['status']=='passed' and preflight['all_identities_exact']
    for path,digest in preflight['input_sha256'].items():assert sha(ROOT/path)==digest,path
    validations=[read(OUT/f'backend-validation-{device}.json') for device in ['cpu','cuda']]
    for rec in validations:
        assert rec['status']=='passed' and rec['backend_sha256']==sha(SRC.parent/'h3i/backend.py')
        assert rec['source_sha256']==sha(SRC/'driver.py') and rec['protocol_sha256']==sha(SRC/'protocol.json')
    plan=read(OUT/'identity-plan.json');mapping=read(OUT/'identity-mapping.json');groups=read(OUT/'energy-groups.json')
    assert mapping['status']=='complete' and mapping['plan_sha256']==sha(OUT/'identity-plan.json')
    assert plan['identity_receipt_sha256']==sha(OUT/'independent-preflight.json') and plan['source_sha256']==sha(SRC/'reuse_identities.py')
    assert plan['groups']==groups['groups'] and plan['actual_unique_search_jobs']==groups['distinct_energies']*3
    for item in mapping['mappings']:
        source=read(ROOT/item['source_result']);derived=read(ROOT/item['derived_result'])
        assert sha(ROOT/item['source_result'])==item['source_result_sha256'] and sha(ROOT/item['derived_result'])==item['derived_result_sha256']
        assert source['status']=='complete' and derived['status']=='derived_by_exact_identity'
        assert source['records_sha256']==derived['records_sha256']==item['records_sha256']
    data=dict(np.load(OUT/'data.npz'));keys={row.tobytes() for row in data['all_solutions']};assert len(keys)==1876
    n=len(data['test_inputs']);PUBLIC.mkdir(parents=True,exist_ok=True)
    rows=[];perseed=[];diagnostics=[];evidence={};capacity=[];model_audits=[];models={}
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=read(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        model=torch.load(folder/'model.pt',weights_only=False);models[arm]=model
        model_audits.append(dict(rec,**{k:v for k,v in next(x for x in preflight['target_models'] if x['arm']==arm).items() if k!='arm'}))
        counts,cap=H.feasible_certificates(model,data);capacity.append(dict(arm=arm,**cap));zs=[];current=[]
        for seed in [0,1,2]:
            z,row,diag=search_audit(arm,seed,data,keys,set(),model)
            assert not (z['stopped']&z['valid'])[counts==0].any()
            zs.append(z);current.append(row);perseed.append(row);diagnostics.append(diag)
        arr={k:np.stack([z[k] for z in zs]) for k in ['n20','n20_certificate','n1','found']};evidence[arm]=arr
        rows.append(dict(arm=arm,target_training_patterns=0,unique_inputs=n,chains=20,search_seeds=[0,1,2],
            original_accuracy_pct=100*float(arr['n20'].mean()),certificate_first_accuracy_pct=100*float(arr['n20_certificate'].mean()),
            n1_accuracy_pct=100*float(arr['n1'].mean()),coverage_pct=100*float(arr['found'].sum(1).mean()/zs[0]['denom'].sum()),
            wrong_stops=sum(r['wrong_stops'] for r in current),capped_chains=sum(r['capped_chains'] for r in current),
            mean_moves=float(np.mean([r['mean_moves'] for r in current])),search_seconds_total=sum(r['search_seconds'] for r in current)))
    for group in groups['groups']:
        base=models[group['representative']]
        for arm in group['members']:
            assert all(torch.equal(base[k],models[arm][k]) for k in ['S','B']) and all(base[k]==models[arm][k] for k in ['q','delta','tau','unit'])
    comparisons=[];denom=np.minimum(data['test_solution_counts'],20)
    for arm in ['hybrid','line-only','direct-offset']:
        a=evidence[arm];b=evidence['true-graph-oracle']
        stats=H.G.A.bootstrap(dict(n20=a['n20_certificate'],found=a['found']),dict(n20=b['n20_certificate'],found=b['found']),denom)
        regression=(b['n20_certificate']&~a['n20_certificate']).any(0);upper=H.G.A.cp_upper(int(regression.sum()),n)
        wrong=next(r['wrong_stops'] for r in rows if r['arm']==arm)
        support=upper<=.001 and wrong==0;falsify=stats['accuracy_one_sided_95_upper_pp']<-.1 or wrong>0
        comparisons.append(dict(arm=arm,control='true-graph-oracle',primary=arm=='hybrid',
            decision=('supported' if support else 'falsified' if falsify else 'inconclusive') if arm=='hybrid' else 'descriptive',
            any_paired_regression_inputs=int(regression.sum()),paired_regression_probability_exact95_upper=upper,
            conservative_accuracy95_lower_pp=-100*upper,wrong_stops=wrong,**stats))
    workers=[read(OUT/f'worker-g{g}.json') for g in [0,1]];assert all(w['status']=='complete' for w in workers)
    for w in workers:
        assert dt.datetime.fromisoformat(preflight['utc'])<dt.datetime.fromisoformat(w['started_utc'])
        assert dt.datetime.fromisoformat(plan['frozen_utc'])<dt.datetime.fromisoformat(w['started_utc'])
        assert all(dt.datetime.fromisoformat(v['utc'])<dt.datetime.fromisoformat(w['started_utc']) for v in validations)
    gpu_seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds() for w in workers)
    assert gpu_seconds<1200
    fit=read(OUT/'line-fit.json');feasibility=read(OUT/'feasibility.json')
    for path in [Path(__file__),SRC.parent/'h3h/analyze.py',SRC.parent/'h3g/analyze.py',SRC.parent/'analyze.py']:sha(path)
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3l',decision=comparisons[0]['decision'],
        primary=comparisons[0],comparisons=comparisons,rows=rows,dataset=p['dataset'],calibration=p['calibration'],line_fit=fit,
        model_audits=model_audits,output_diagnostics=diagnostics,posthoc_valid_certificate_capacity=capacity,
        source_support_diagnostic=preflight['source_support_diagnostic'],feasibility=feasibility,
        bp_adamw='Not trained or evaluated on superqueens; ordinary-queen results are not matched controls.',
        independent_saved_boards=groups['actual_unique_search_jobs']*n*20,derived_record_boards=groups['avoided_identical_jobs']*n*20,
        identity_plan=plan,identity_mapping=mapping,backend_validations=validations,worker_receipts=workers,gpu_worker_seconds=gpu_seconds,
        input_sha256=HASHES,limits=p['limits'],audit_seconds=time.perf_counter()-started)
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    for name,values in [('per_seed.csv',perseed),('extra_source_offsets.csv',preflight['source_support_diagnostic']['extra_offsets'])]:
        with (PUBLIC/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print(json.dumps(dict(status='validated',primary=comparisons[0],rows=rows,capacity=capacity,
        independent_saved_boards=summary['independent_saved_boards'],gpu_worker_seconds=gpu_seconds,audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()

