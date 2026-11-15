"""Independent source-time, fresh-cost and all-state audit for H3ab."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv,datetime as dt,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3ab';OUT=ROOT/'artifacts/research/20261114/loki/h3ab';OLD=ROOT/'artifacts/research/20261114/loki/h3aa';BASE=ROOT/'artifacts/research/20261114/loki';PUBLIC=ROOT/'results/research/loki/h3ab';HASHES={}
def load(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
STATS=load(SRC.parent/'analyze.py','ab_statistics');CAP=load(SRC.parent/'h3w/analyze.py','ab_capacity')
def sha(p):p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def audit(kernel,seed,cap,m,data,keys,capacity):
    folder=OUT/f'search/{kernel}/s{seed}';r=read(folder/'result.json');assert r['status']=='complete'and r['cap']==cap and r['records_sha256']==sha(folder/'records.npz')
    assert r['model_sha256']==sha(OLD/'model.npz')and r['source_sha256']==sha(SRC/'driver.py')and r['backend_sha256']==sha(SRC.parent/f'h3aa/{kernel}_backend.py')and r['calibration_selection_sha256']==sha(OUT/'calibration-selection.json')and r['protocol_sha256']==sha(SRC/'protocol.json')
    assert kernel==('arbitrary'if(r['gpu']+seed)%2==0 else'permutation')
    path=OUT/f'initial/s{seed}.npz';initial=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:169].astype(bool);assert r['initial_file_sha256']==sha(path)and r['initial_unpacked_bytes_sha256']==hashlib.sha256(initial.tobytes()).hexdigest()
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);inputs=data['test_inputs'];assert np.array_equal(inputs,z['inputs'])and(boards>=inputs[:,None,:]).all()and(boards.sum(-1)==13).all()
    grid=boards.reshape(-1,13,13);perm=(grid.sum(1)==1).all(1)&(grid.sum(2)==1).all(1)
    if kernel=='permutation':assert perm.all()
    valid=np.array([[row.tobytes()in keys and row[given].all()for row in samples]for given,samples in zip(inputs,boards)]);assert np.array_equal(valid,z['valid']);stopped=z['stopped'];primary=(stopped&valid).any(1);assert np.array_equal(primary,z['certified_success'])and not(primary&(capacity==0)).any()
    original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));assert np.array_equal(original,z['original_index'])and np.array_equal(selected,z['certificate_index'])and np.array_equal(valid[ids,original],z['n20'])and np.array_equal(valid[ids,selected],z['n20_certificate'])
    denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({row.tobytes()for row in samples[ok]}),d)for samples,ok,d in zip(boards,valid,denom)]);assert np.array_equal(found,z['found'])and np.array_equal(denom,z['denom'])
    assert((z['moves']>=0)&(z['moves']<=cap)).all()and(z['moves'][~stopped]==cap).all()and((z['accepted']>=0)&(z['accepted']<=z['moves'])).all()
    assert r['proposal_moves']==int(z['moves'].sum())and r['accepted_swaps']==int(z['accepted'].sum())
    flat=boards.reshape(-1,169);J=m['J'].astype(np.float64);F=m['F'].astype(np.float64);energies=[];conflicts=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);energies.append((.5*x*(x@J)).sum(1));conflicts.append(((x*(x@F)).sum(1)/2).astype(np.int64))
    energy=np.concatenate(energies).reshape(stopped.shape);count=np.concatenate(conflicts).reshape(stopped.shape);assert np.array_equal(count,z['conflicts'])and(count[stopped]==0).all()and(energy[stopped]==0).all()and(count[~stopped]>0).all()
    wrong=int((stopped&~valid).sum());assert wrong==r['wrong_stops'];error=np.abs(energy-z['energy']);exact_original=energy.argmin(1);exact_selected=np.where(stopped.any(1),np.where(stopped,energy,np.inf).argmin(1),exact_original)
    assert r['certified_success_pct']==100*float(primary.mean())and r['original_accuracy_pct']==100*float(z['n20'].mean())and r['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())and r['coverage_pct']==100*float(found.sum()/denom.sum())
    row=dict(kernel=kernel,seed=seed,gpu=r['gpu'],cap=cap,unique_inputs=4096,chains=20,certified_valid_inputs=int(primary.sum()),original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=wrong,capped_chains=int((~stopped).sum()),proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),worker_solve_seconds=r['worker_solve_seconds'],warmup_seconds=r['warmup_seconds'],initial_file_sha256=r['initial_file_sha256'],initial_unpacked_bytes_sha256=r['initial_unpacked_bytes_sha256'],records_sha256=r['records_sha256'])
    diagnostic=dict(kernel=kernel,seed=seed,all_q13_and_givens_preserved=True,returned_permutation_states=int(perm.sum()),all_integer_conflict_counts_exact=True,wrong_stops=wrong,running_energy_absolute_error_max=float(error.max()),numerical_gate_passed=bool(error.max()<=.1),
        exact_energy_original_accuracy_pct=100*float(valid[ids,exact_original].mean()),exact_energy_certificate_accuracy_pct=100*float(valid[ids,exact_selected].mean()),failed_certified_despite_capacity=int(((capacity>0)&~primary).sum()),fallback_success_without_capacity=int(((capacity==0)&z['n20_certificate']).sum()))
    return z,row,diagnostic

def main():
    start=time.perf_counter();PUBLIC.mkdir(parents=True,exist_ok=True);p=read(SRC/'protocol.json');selection=read(OUT/'calibration-selection.json');assert selection['status']=='matched_source_median_times'
    for group in['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    for rel,h in selection['calibration_result_sha256'].items():assert sha(ROOT/rel)==h,rel
    groups=selection['all_calibration_groups'];ref=groups['arbitrary-c20000'];best=min([g for g in groups.values()if g['kernel']=='permutation'],key=lambda g:(abs(np.median([r['worker_solve_seconds']for r in g['trials']])/ref['median_seconds']-1),g['cap']))
    for group in groups.values():
        assert len(group['trials'])==6 and {(r['gpu'],r['seed'])for r in group['trials']}=={(g,s)for g in[0,1]for s in[101,102,103]}
        assert group['median_seconds']==float(np.median([r['worker_solve_seconds']for r in group['trials']]))
    assert best['cap']==selection['selected_permutation_cap']and abs(selection['source_median_time_ratio']-1)<=.05
    metadata=read(OUT/'calibration-data.json');caldata=np.load(OUT/'calibration-data.npz');assert sha(OUT/'calibration-data.npz')==metadata['data_sha256'];source=np.load(BASE/'h3t/solutions-n12.npy');embedded=np.zeros((176,169),bool)
    for k,board in enumerate(source):cells=np.flatnonzero(board);embedded[k,cells//12*13+cells%12]=True
    for given in caldata['inputs']:assert embedded[:,given].all(1).any()
    dataset=read(OUT/'dataset.json');pre=read(OUT/'preflight.json');assert pre['status']=='passed'and dt.datetime.fromisoformat(selection['utc'])<dt.datetime.fromisoformat(dataset['utc'])
    for rel,h in pre['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    data=dict(np.load(OUT/'data.npz'));old=np.load(OLD/'data.npz');prior=np.union1d(old['prior_target_test_keys'],old['test_keys']);excluded=np.union1d(prior,caldata['keys']);universe=np.load(BASE/'h3t/input-universe.npz');eligible=np.setdiff1d(universe['keys'],excluded);expected=np.sort(np.random.default_rng(2026112515).permutation(eligible)[:4096]);assert np.array_equal(expected,data['test_keys'])and np.array_equal(excluded,data['all_excluded_keys'])and not np.isin(expected,caldata['keys']).any()
    m=dict(np.load(OLD/'model.npz'));capacity,capinfo=CAP.capacity(m,data);keys={row.tobytes()for row in data['all_solutions']};records={};rows=[];perseed=[];diagnostics=[]
    for kernel in['arbitrary','permutation']:
        cap=dataset['selected_caps'][kernel];jobs=[]
        for seed in[0,1,2]:z,row,diag=audit(kernel,seed,cap,m,data,keys,capacity);jobs.append(z);perseed.append(row);diagnostics.append(diag)
        records[kernel]=dict(n20=np.stack([z['certified_success']for z in jobs]),found=np.stack([z['found']for z in jobs]));subset=[r for r in perseed if r['kernel']==kernel];times=[r['worker_solve_seconds']for r in subset];proposals=sum(r['proposal_moves']for r in subset)
        rows.append(dict(kernel=kernel,cap=cap,certified_success_pct=100*float(records[kernel]['n20'].mean()),original_accuracy_pct=100*float(np.stack([z['n20']for z in jobs]).mean()),certificate_first_accuracy_pct=100*float(np.stack([z['n20_certificate']for z in jobs]).mean()),coverage_pct=100*float(records[kernel]['found'].sum()/(3*jobs[0]['denom'].sum())),wrong_stops=sum(r['wrong_stops']for r in subset),proposal_moves=proposals,accepted_swaps=sum(r['accepted_swaps']for r in subset),capped_chains=sum(r['capped_chains']for r in subset),median_worker_solve_seconds=float(np.median(times)),minimum_worker_solve_seconds=min(times),maximum_worker_solve_seconds=max(times),total_worker_solve_seconds=sum(times),amortized_nanoseconds_per_proposal=1e9*sum(times)/proposals))
    for seed in[0,1,2]:
        a,b=[r for r in perseed if r['seed']==seed];assert a['initial_file_sha256']==b['initial_file_sha256']and a['initial_unpacked_bytes_sha256']==b['initial_unpacked_bytes_sha256']
    difference=STATS.bootstrap(records['permutation'],records['arbitrary'],np.minimum(data['test_solution_counts'],20));wrong=rows[1]['wrong_stops'];numeric=all(d['numerical_gate_passed']for d in diagnostics)
    quality='falsified'if wrong or difference['accuracy_one_sided_95_upper_pp']<50 else'inconclusive_numerical_audit'if not numeric else'supported'if difference['accuracy_one_sided_95_lower_pp']>=50 else'inconclusive'
    ratio=rows[1]['median_worker_solve_seconds']/rows[0]['median_worker_solve_seconds'];match=abs(ratio-1)<=.10
    decision='falsified_wrong_stop'if wrong else'inconclusive_target_timing_mismatch'if not match else quality
    workers=[read(path)for path in sorted(OUT.glob('calibration-worker-*.json'))]+[read(path)for path in sorted(OUT.glob('target-worker-*.json'))];assert all(w['status']=='complete'for w in workers);seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds()for w in workers);budget=read(OUT/'global-budget.json');assert seconds<=1200 and max(dt.datetime.fromisoformat(w['completed_utc']).timestamp()for w in workers)<=budget['deadline_unix']
    calibration_workers=[w for w in workers if w['phase']!='target'];target_workers=[w for w in workers if w['phase']=='target'];worker_seconds=lambda group:sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds()for w in group)
    np.savez_compressed(PUBLIC/'per-input-capacity.npz',keys=data['test_keys'],zero_conflict_completions=capacity,total_completions=data['test_solution_counts'])
    sha(__file__);sha(SRC.parent/'h3w/analyze.py');count_validation=read(ROOT/'results/research/loki/h3v/audit-count-validation.json');assert count_validation['status']=='passed'
    summary=dict(status='validated',utc=now(),hypothesis='H3ab',decision=decision,quality_decision=quality,primary=dict(required_gain_pp=50.,wrong_stops=wrong,numerical_gate_passed=numeric,**difference),
        target_timing=dict(median_permutation_to_arbitrary_ratio=ratio,relative_difference=ratio-1,matching_tolerance=.10,matching_gate_passed=match,retuned_after_target=False),
        rows=rows,dataset=dataset,calibration_selection=selection,calibration_data=metadata,
        calibration_access_clarification='The frozen shared integrity guard SHA256-hashes historical target artifact bytes during calibration. No target arrays,keys or labels are decoded or used for timing-count selection,and no targetaccuracy is computed. Source-only describes data used in calibration,not filesystem isolation; the protocol no-access wording was broader than implementation.',
        output_diagnostics=diagnostics,exact_zero_set_capacity=capinfo,
        independent_timed_target_states=6*4096*20,shared_per_seed_initial_bytes_verified=True,calibration_gpu_worker_seconds=worker_seconds(calibration_workers),target_gpu_worker_seconds=worker_seconds(target_workers),total_gpu_worker_seconds=seconds,
        calibration_timed_solve_seconds=sum(r['worker_solve_seconds']for g in groups.values()for r in g['trials']),calibration_warmup_seconds=sum(warm['seconds']for w in calibration_workers for warm in w['warmups']),
        target_warmup_seconds=sum(r['warmup_seconds']for r in perseed),worker_receipts=workers,global_budget=budget,optimizer_refits=0,limits=p['limits'],input_sha256=HASHES,
        capacity_export_sha256=sha(PUBLIC/'per-input-capacity.npz'),audit_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary)
    with(PUBLIC/'per_seed.csv').open('w')as stream:writer=csv.DictWriter(stream,fieldnames=list(perseed[0]));writer.writeheader();writer.writerows(perseed)
    save(PUBLIC/'calibration-selection.json',selection)
    print(json.dumps({k:summary[k]for k in['decision','quality_decision','primary','target_timing','rows','calibration_gpu_worker_seconds','target_gpu_worker_seconds','total_gpu_worker_seconds']}),flush=True)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
if __name__=='__main__':main()
