"""Independent saved-output audit for frozen fresh-input dihedral experiment H3i."""
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
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3i';OUT=ROOT/'artifacts/research/20261114/loki/h3i';PUBLIC=ROOT/'results/research/loki/h3i'
spec=importlib.util.spec_from_file_location('h3h_audit',SRC.parent/'h3h/analyze.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
HASHES={}
def sha(path):
    path=Path(path)
    with path.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))]=v;return v
def read(path):sha(path);return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def search_audit(arm,seed,data,keys,union_keys,model):
    folder=OUT/f'search/{arm}/s{seed}';rec=read(folder/'result.json')
    assert rec['status']=='complete' and rec['protocol_sha256']==sha(SRC/'protocol.json')
    assert rec['source_sha256']==sha(SRC/'driver.py') and rec['backend_sha256']==sha(SRC/'backend.py')
    assert rec['records_sha256']==sha(folder/'records.npz') and rec['model_sha256']==sha(ROOT/rec['model_path'])
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:144].astype(bool);inputs=data['test_inputs']
    assert np.array_equal(inputs,z['inputs']) and (boards>=inputs[:,None,:]).all()
    assert not any(row.tobytes() in union_keys for samples in boards for row in samples)
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
    assert np.array_equal(z['flop'],41761+734*z['moves'])
    wrong=z['stopped']&~valid;identical=(boards==inputs[:,None,:]).all(-1)
    assert (boards.sum(-1)[z['stopped']]==model['q']).all()
    assert np.array_equal(z['cardinality_integer_additions'],143*(z['moves']+1))
    assert not ((z['moves']==0)&z['stopped']).any()
    assert rec['wrong_stops']==int(wrong.sum()) and rec['capped_chains']==int((~z['stopped']).sum())
    assert rec['original_accuracy_pct']==100*float(z['n20'].mean()) and rec['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())
    assert rec['coverage_pct']==100*float(found.sum()/denom.sum()) and rec['mean_moves']==float(z['moves'].mean())
    counts=boards.sum(-1);assert np.array_equal(counts,z['queen_counts'])
    occupancy={name:{label:int((mask&condition).sum()) for label,condition in [('below_q',counts<12),('equal_q',counts==12),('above_q',counts>12)]}
               for name,mask in [('stopped_invalid',wrong),('capped_invalid',~z['stopped']&~valid)]}
    J=model['S'][:,:,1,1].numpy().astype(np.float64);_,trueJ=H.true_graph();flat=boards.reshape(-1,144);givens=np.repeat(inputs,20,0)
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
              wrong_stops_rejected_after_removing_only_false_exclusions=int(counter_reject.sum()),
              counterfactual_scope='Evaluate the same stopped boards under the true graph; no causal search rerun or performance claim.')
    row=dict(arm=arm,updates=0,seed=seed,unique_inputs=len(inputs),chains=20,
             original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),n1_correct=int(z['n1'].sum()),
             coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=int(wrong.sum()),capped_chains=rec['capped_chains'],
             mean_moves=rec['mean_moves'],search_seconds=rec['search_seconds'],model_sha256=rec['model_sha256'],records_sha256=rec['records_sha256'],result_sha256=sha(folder/'result.json'))
    return z,row,diag


def main():
    started=time.perf_counter();torch.set_num_threads(1);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    preflight=read(OUT/'independent-preflight.json');assert preflight['status']=='passed'
    for path,digest in preflight['input_sha256'].items():assert sha(ROOT/path)==digest,path
    validations=[read(OUT/f'backend-validation-{device}.json') for device in ['cpu','cuda']]
    for receipt in validations:
        assert receipt['status']=='passed' and receipt['backend_sha256']==sha(SRC/'backend.py')
        assert receipt['source_sha256']==sha(SRC/'driver.py') and receipt['protocol_sha256']==sha(SRC/'protocol.json')
    d=dict(np.load(OUT/'data.npz'));keys={row.tobytes() for row in d['all_solutions']}
    union_keys={row.tobytes() for row in d['all_solutions'][d['union_indices']]};n=len(d['test_inputs'])
    PUBLIC.mkdir(parents=True,exist_ok=True);rows=[];perseed=[];diagnostics=[];capacity=[];evidence={};wrong_inputs={};model_audits=[]
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=read(folder/'model.json')
        assert rec['model_sha256']==sha(folder/'model.pt') and rec['construction_sha256']==sha(folder/'construction.npz')
        model=torch.load(folder/'model.pt',weights_only=False);model_audits.append(rec)
        counts,cap=H.feasible_certificates(model,d);capacity.append(dict(arm=arm,**cap))
        zs=[];current=[]
        for seed in [0,1,2]:
            z,row,diag=search_audit(arm,seed,d,keys,union_keys,model)
            assert not (z['stopped']&z['valid'])[counts==0].any()
            zs.append(z);current.append(row);perseed.append(row);diagnostics.append(diag)
        arr={k:np.stack([z[k] for z in zs]) for k in ['n20','n20_certificate','n1','found']};evidence[arm]=arr
        wrong_inputs[arm]=np.stack([(z['stopped']&~z['valid']).any(1) for z in zs]).any(0)
        rows.append(dict(arm=arm,raw_patterns=cfg['raw_patterns'],unique_training_patterns=rec['unique_patterns'],pooling=cfg['pooling'],
            unique_inputs=n,chains=20,search_seeds=[0,1,2],original_accuracy_pct=100*float(arr['n20'].mean()),
            certificate_first_accuracy_pct=100*float(arr['n20_certificate'].mean()),n1_accuracy_pct=100*float(arr['n1'].mean()),
            coverage_pct=100*float(arr['found'].sum(1).mean()/zs[0]['denom'].sum()),wrong_stops=sum(x['wrong_stops'] for x in current),
            any_wrong_stop_inputs=int(wrong_inputs[arm].sum()),capped_chains=sum(x['capped_chains'] for x in current),
            mean_moves=float(np.mean([x['mean_moves'] for x in current])),search_seconds_total=sum(x['search_seconds'] for x in current)))
    denom=np.minimum(d['test_solution_counts'],20);comparisons=[]
    for arm in ['augmented128','raw1024','offset128']:
        a=evidence[arm];b=evidence['raw128']
        stats=H.G.A.bootstrap(dict(n20=a['n20_certificate'],found=a['found']),dict(n20=b['n20_certificate'],found=b['found']),denom)
        k=int(wrong_inputs[arm].sum());upper=H.G.A.cp_upper(k,n);lower=1-H.G.A.cp_upper(n-k,n)
        support=stats['accuracy_one_sided_95_lower_pp']>=10 and upper<=.001
        falsify=stats['accuracy_one_sided_95_upper_pp']<10 or lower>.001
        comparisons.append(dict(arm=arm,control='raw128',primary=arm=='augmented128',
            decision=('supported' if support else 'falsified' if falsify else 'inconclusive') if arm=='augmented128' else 'descriptive',
            any_wrong_stop_inputs=k,any_wrong_stop_probability_exact95_upper=upper,any_wrong_stop_probability_exact95_lower=lower,
            wrong_stop_bound_unit='unique input with any wrong stopped chain across all20chains and3searchseeds; conservative for per-chain probability',**stats))
    workers=[read(OUT/f'worker-g{g}.json') for g in [0,1]];assert all(w['status']=='complete' for w in workers)
    for w in workers:
        assert dt.datetime.fromisoformat(preflight['utc'])<dt.datetime.fromisoformat(w['started_utc'])
        assert all(dt.datetime.fromisoformat(v['utc'])<dt.datetime.fromisoformat(w['started_utc']) for v in validations)
    gpu_seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds() for w in workers)
    assert gpu_seconds<1200
    for path in [Path(__file__),SRC.parent/'h3h/analyze.py',SRC.parent/'h3g/analyze.py',SRC.parent/'analyze.py']:sha(path)
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3i',decision=comparisons[0]['decision'],
        primary=comparisons[0],comparisons=comparisons,rows=rows,dataset=p['dataset'],model_audits=model_audits,
        output_diagnostics=diagnostics,posthoc_valid_certificate_capacity=capacity,
        bp_adamw='Not run on this fresh cohort. H3g BP AdamW used another bank/cohort and lacks the q gate, so its numerical results are not a matched comparison.',
        independent_saved_boards=len(perseed)*n*20,every_valid_returned_pattern_outside_union=True,
        independent_preflight=preflight,backend_validations=validations,worker_receipts=workers,gpu_worker_seconds=gpu_seconds,
        input_sha256=HASHES,limits=p['limits'],audit_seconds=time.perf_counter()-started)
    save(OUT/'summary.json',summary);save(PUBLIC/'summary.json',summary)
    with (PUBLIC/'per_seed.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(perseed[0]));writer.writeheader();writer.writerows(perseed)
    print(json.dumps(dict(status='validated',primary=summary['primary'],rows=rows,capacity=capacity,
        independent_saved_boards=summary['independent_saved_boards'],gpu_worker_seconds=gpu_seconds,audit_seconds=summary['audit_seconds'])))


if __name__=='__main__':main()

