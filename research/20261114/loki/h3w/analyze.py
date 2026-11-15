"""Independent fixed-q, integer-certificate and full saved-state audit for H3w."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv,datetime as dt,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3w';OUT=ROOT/'artifacts/research/20261114/loki/h3w';PUBLIC=ROOT/'results/research/loki/h3w';HASHES={}
spec=importlib.util.spec_from_file_location('loki_stat',SRC.parent/'analyze.py');BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
def sha(p):
    p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows):
    with p.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def capacity(m,data):
    y=data['all_solutions'];x=y.astype(np.int64);F=m['F'].astype(np.int64);counts=(x*(x@F)).sum(1)//2;valid_zero=counts==0;inputs=data['test_inputs'];compatible=np.array([y[:,given].all(1) for given in inputs]);capacity=(compatible&valid_zero).sum(1)
    assert np.array_equal(compatible.sum(1),data['test_solution_counts']);info=dict(valid_complete_target_solutions_with_zero_conflicts=int(valid_zero.sum()),total_complete_target_solutions=len(y),inputs_with_any_zero_conflict_completion=int((capacity>0).sum()),inputs_without_zero_conflict_completion=int((capacity==0).sum()),compatible_zero_conflict_completions=int(capacity.sum()),total_compatible_completions=int(data['test_solution_counts'].sum()),certified_primary_ceiling_pct=100*float((capacity>0).mean()),clues_with_internal_Fconflict_inputs=int(((inputs.astype(np.int64)*(inputs.astype(np.int64)@F)).sum(1)>0).sum()),interpretation='Exact ceiling for abstaining certified primary. Given-given conflicts count here, unlike a local-margin certificate that ignores clamped-cell removal margins. Not a ceiling for valid fallback accuracy.')
    # Literal occupied-pair counting independently validates every full target solution.
    i,j=np.triu_indices(169,1);literal=(y[:,i]&y[:,j])@F[i,j];assert np.array_equal(literal,counts)
    return capacity,info

def audit(arm,seed,m,data,keys,capacity):
    folder=OUT/f'search/{arm}/s{seed}';r=read(folder/'result.json');assert r['status']=='complete'and r['records_sha256']==sha(folder/'records.npz')and r['model_sha256']==sha(OUT/f'models/{arm}/model.npz');assert r['protocol_sha256']==sha(SRC/'protocol.json')and r['source_sha256']==sha(SRC/'driver.py')and r['backend_sha256']==sha(SRC/'backend.py' if arm!='bp-adamw-arbitrary' else SRC.parent/'h3v/backend.py')
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);inputs=data['test_inputs'];assert np.array_equal(inputs,z['inputs'])and(boards>=inputs[:,None,:]).all()and(boards.sum(-1)==13).all()
    grid=boards.reshape(-1,13,13);permutation_states=(grid.sum(1)==1).all(1)&(grid.sum(2)==1).all(1)
    if arm!='bp-adamw-arbitrary':assert permutation_states.all()
    valid=np.array([[row.tobytes()in keys and row[given].all()for row in samples]for given,samples in zip(inputs,boards)]);assert np.array_equal(valid,z['valid']);stopped=z['stopped'];primary=(stopped&valid).any(1);assert np.array_equal(primary,z['certified_success']);assert not(primary&(capacity==0)).any()
    original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));assert np.array_equal(original,z['original_index'])and np.array_equal(selected,z['certificate_index']);assert np.array_equal(valid[ids,original],z['n20'])and np.array_equal(valid[ids,selected],z['n20_certificate'])
    denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({row.tobytes()for row in samples[ok]}),d)for samples,ok,d in zip(boards,valid,denom)]);assert np.array_equal(found,z['found'])and np.array_equal(denom,z['denom'])
    assert((z['moves']>=0)&(z['moves']<=20000)).all()and(z['moves'][~stopped]==20000).all();assert((z['accepted']>=0)&(z['accepted']<=z['moves'])).all();assert r['proposal_moves']==int(z['moves'].sum())and r['accepted_swaps']==int(z['accepted'].sum())
    J=m['J'].astype(np.float64);F=m['F'].astype(np.int64);flat=boards.reshape(-1,169);energies=[];conflicts=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);energies.append((.5*x*(x@J)).sum(1));conflicts.append(((x*(x@F.astype(np.float64))).sum(1)/2).astype(np.int64))
    energy=np.concatenate(energies).reshape(stopped.shape);count=np.concatenate(conflicts).reshape(stopped.shape);assert np.array_equal(count,z['conflicts']);assert(count[stopped]==0).all()and(energy[stopped]==0).all();assert(count[~stopped]>0).all()
    error=abs(energy-z['energy']);exact_original=energy.argmin(1);exact_selected=np.where(stopped.any(1),np.where(stopped,energy,np.inf).argmin(1),exact_original);wrong=int((stopped&~valid).sum());assert r['wrong_stops']==wrong
    assert r['certified_success_pct']==100*float(primary.mean())and r['original_accuracy_pct']==100*float(z['n20'].mean())and r['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())and r['coverage_pct']==100*float(found.sum()/denom.sum())
    diag=dict(arm=arm,seed=seed,all_states_q13=True,all_givens_preserved=True,permutation_returned_states=int(permutation_states.sum()),total_returned_states=len(permutation_states),permutation_kernel=arm!='bp-adamw-arbitrary',all_integer_conflict_counts_exact=True,stopped_positive_conflict_counts=0,stopped_nonzero_exact_energy=0,wrong_stops=wrong,
      running_energy_absolute_error_max=float(error.max()),running_energy_error_quantiles={str(q):float(np.quantile(error,q))for q in [0,.5,.9,.99,1]},numerical_gate_passed=bool(error.max()<=.1),
      original_exact_energy_selector_disagreements=int((original!=exact_original).sum()),certificate_exact_energy_selector_disagreements=int((selected!=exact_selected).sum()),exact_energy_original_accuracy_pct=100*float(valid[ids,exact_original].mean()),exact_energy_certificate_accuracy_pct=100*float(valid[ids,exact_selected].mean()),
      fallback_success_without_zero_conflict_capacity=int(((capacity==0)&z['n20_certificate']).sum()),failed_certified_despite_zero_set_capacity=int(((capacity>0)&~primary).sum()),
      invalid_capped_chains=int((~valid&~stopped).sum()),valid_capped_chains=int((valid&~stopped).sum()),minimum_cap_conflicts=int(count[~stopped].min())if(~stopped).any()else None)
    row=dict(arm=arm,seed=seed,unique_inputs=len(inputs),chains=20,certified_valid_inputs=int(primary.sum()),original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=wrong,capped_chains=int((~stopped).sum()),proposed_swaps=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),search_seconds=r['search_seconds'],model_sha256=r['model_sha256'],records_sha256=r['records_sha256'])
    return z,row,diag

def main():
    start=time.perf_counter();PUBLIC.mkdir(parents=True,exist_ok=True);p=read(SRC/'protocol.json')
    for group in ['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    pre=read(OUT/'independent-preflight.json');assert pre['status']=='passed'
    for rel,h in pre['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    data=dict(np.load(OUT/'data.npz'));keys={row.tobytes()for row in data['all_solutions']};perseed=[];diags=[];rows=[];caps={};cap_arrays={};records={};models={}
    for cfg in p['arms']:
        arm=cfg['arm'];models[arm]=dict(np.load(OUT/f'models/{arm}/model.npz'));cap,info=capacity(models[arm],data);caps[arm]=info;cap_arrays[arm]=cap;zz=[]
        for seed in [0,1,2]:
            z,row,diag=audit(arm,seed,models[arm],data,keys,cap);zz.append(z);perseed.append(row);diags.append(diag)
        records[arm]=dict(n20=np.stack([z['certified_success']for z in zz]),found=np.stack([z['found']for z in zz]));subset=[r for r in perseed if r['arm']==arm]
        rows.append(dict(arm=arm,unique_inputs=4096,chains=20,search_seeds=[0,1,2],certified_success_pct=100*float(records[arm]['n20'].mean()),original_accuracy_pct=100*float(np.stack([z['n20']for z in zz]).mean()),certificate_first_accuracy_pct=100*float(np.stack([z['n20_certificate']for z in zz]).mean()),coverage_pct=100*float(records[arm]['found'].sum()/(3*zz[0]['denom'].sum())),wrong_stops=sum(r['wrong_stops']for r in subset),capped_chains=sum(r['capped_chains']for r in subset),proposed_swaps=sum(r['proposed_swaps']for r in subset),accepted_swaps=sum(r['accepted_swaps']for r in subset),search_seconds_total=sum(r['search_seconds']for r in subset),microseconds_per_proposal=1e6*sum(r['search_seconds']for r in subset)/sum(r['proposed_swaps']for r in subset)))
    assert np.array_equal(models['bp-adamw-permutation']['F'],models['uniform-filtered-permutation']['F']);assert np.array_equal(cap_arrays['bp-adamw-permutation'],cap_arrays['uniform-filtered-permutation']);assert np.array_equal(cap_arrays['bp-adamw-permutation'],cap_arrays['bp-adamw-arbitrary'])
    comparisons=[];denom=np.minimum(data['test_solution_counts'],20)
    for a,b in [('bp-adamw-permutation','bp-adamw-arbitrary'),('bp-adamw-permutation','uniform-filtered-permutation'),('bp-adamw-permutation','oracle-permutation'),('uniform-filtered-permutation','oracle-permutation')]:comparisons.append(dict(arm=a,control=b,metric='abstaining certified valid completion from20chains',primary=len(comparisons)==0,**BASE.bootstrap(records[a],records[b],denom)))
    primary=comparisons[0];wrong=rows[0]['wrong_stops'];numeric=all(x['numerical_gate_passed']for x in diags if x['arm']in['bp-adamw-permutation','bp-adamw-arbitrary'])
    if wrong or primary['accuracy_one_sided_95_upper_pp']<50:decision='falsified'
    elif not numeric:decision='inconclusive_numerical_audit'
    elif primary['accuracy_one_sided_95_lower_pp']>=50:decision='supported'
    else:decision='inconclusive'
    primary.update(decision=decision,wrong_stops=wrong,numerical_gate_passed=numeric,required_gain_pp=50.)
    workers=[read(OUT/f'worker-g{x}.json')for x in [0,1]];assert all(w['status']=='complete'for w in workers);seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds()for w in workers);budget=read(OUT/'global-budget.json')
    assert seconds<=1200;assert max(dt.datetime.fromisoformat(w['completed_utc']).timestamp()for w in workers)<=budget['deadline_unix']
    validations=[read(OUT/f'kernel-validation-{x}.json')for x in ['cpu','cuda']];count_validation=read(ROOT/'results/research/loki/h3v/audit-count-validation.json');assert count_validation['status']=='passed';sha(SRC/'analyze.py');sha(SRC.parent/'analyze.py');sha(SRC.parent/'h3v/validate_audit_counts.py')
    np.savez_compressed(PUBLIC/'per-input-capacity.npz',keys=data['test_keys'],weighted=cap_arrays['bp-adamw-permutation'],arbitrary=cap_arrays['bp-adamw-arbitrary'],uniform=cap_arrays['uniform-filtered-permutation'],oracle=cap_arrays['oracle-permutation'],compatible=data['test_solution_counts'])
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3w',decision=decision,primary=primary,comparisons=comparisons,rows=rows,dataset=p['dataset'],exact_zero_set_capacities=caps,output_diagnostics=diags,independent_preflight=pre,kernel_validation=validations,audit_count_validation=count_validation,worker_receipts=workers,global_budget=budget,gpu_worker_seconds=seconds,independent_saved_boards=12*4096*20,optimizer_refits=0,weighted_uniform_zero_sets_identical=True,limits=p['limits'],input_sha256=HASHES,capacity_export_sha256=sha(PUBLIC/'per-input-capacity.npz'),audit_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);csvsave(PUBLIC/'per_seed.csv',perseed);print(json.dumps(dict(decision=decision,primary=primary,rows=rows,capacities=caps,gpu_worker_seconds=seconds,max_energy_error=max(d['running_energy_absolute_error_max']for d in diags))))
if __name__=='__main__':main()
