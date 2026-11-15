"""Audit all H3aa states, shared starts, paired comparisons and compute costs."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv,datetime as dt,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa';OUT=ROOT/'artifacts/research/20261114/loki/h3aa';PUBLIC=ROOT/'results/research/loki/h3aa';HASHES={}
def load(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
STATS=load(SRC.parent/'analyze.py','loki_stats');OLD=load(SRC.parent/'h3w/analyze.py','h3w_capacity')
def sha(p):p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def audit(cfg,seed,m,data,keys,capacity):
    arm=cfg['arm'];folder=OUT/f'search/{arm}/s{seed}';r=read(folder/'result.json');assert r['status']=='complete'and r['records_sha256']==sha(folder/'records.npz')
    assert r['model_sha256']==sha(OUT/'model.npz')and r['driver_sha256']==sha(SRC/'driver.py')and r['backend_sha256']==sha(SRC/f'{cfg["kernel"]}_backend.py')and r['protocol_sha256']==sha(SRC/'protocol.json')
    initpath=OUT/f'initial/{cfg["initialization"]}/s{seed}.npz';initial=np.unpackbits(np.load(initpath)['boards'],axis=-1)[...,:169].astype(bool)
    assert r['initial_file_sha256']==sha(initpath)and r['initial_unpacked_bytes_sha256']==hashlib.sha256(initial.tobytes()).hexdigest()
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);inputs=data['test_inputs'];assert np.array_equal(inputs,z['inputs'])and(boards>=inputs[:,None,:]).all()and(boards.sum(-1)==13).all()
    grid=boards.reshape(-1,13,13);permutation=(grid.sum(1)==1).all(1)&(grid.sum(2)==1).all(1)
    if cfg['kernel']=='permutation':assert permutation.all()
    valid=np.array([[row.tobytes()in keys and row[given].all()for row in samples]for given,samples in zip(inputs,boards)]);assert np.array_equal(valid,z['valid']);stopped=z['stopped'];primary=(stopped&valid).any(1);assert np.array_equal(primary,z['certified_success'])and not(primary&(capacity==0)).any()
    original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));assert np.array_equal(original,z['original_index'])and np.array_equal(selected,z['certificate_index'])
    assert np.array_equal(valid[ids,original],z['n20'])and np.array_equal(valid[ids,selected],z['n20_certificate'])
    denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({row.tobytes()for row in samples[ok]}),d)for samples,ok,d in zip(boards,valid,denom)]);assert np.array_equal(found,z['found'])and np.array_equal(denom,z['denom'])
    assert((z['moves']>=0)&(z['moves']<=20000)).all()and(z['moves'][~stopped]==20000).all()and((z['accepted']>=0)&(z['accepted']<=z['moves'])).all()
    assert r['proposal_moves']==int(z['moves'].sum())and r['accepted_swaps']==int(z['accepted'].sum())
    flat=boards.reshape(-1,169);J=m['J'].astype(np.float64);F=m['F'].astype(np.float64);energies=[];conflicts=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);energies.append((.5*x*(x@J)).sum(1));conflicts.append(((x*(x@F)).sum(1)/2).astype(np.int64))
    energy=np.concatenate(energies).reshape(stopped.shape);count=np.concatenate(conflicts).reshape(stopped.shape)
    assert np.array_equal(count,z['conflicts'])and(count[stopped]==0).all()and(energy[stopped]==0).all()and(count[~stopped]>0).all()
    initial_count=(initial.astype(np.float64)*(initial.astype(np.float64)@F)).sum(-1)/2;done=initial_count==0
    assert stopped[done].all()and(z['moves'][done]==0).all()and np.array_equal(boards[done],initial[done])
    wrong=int((stopped&~valid).sum());assert r['wrong_stops']==wrong;error=np.abs(energy-z['energy']);exact_original=energy.argmin(1);exact_selected=np.where(stopped.any(1),np.where(stopped,energy,np.inf).argmin(1),exact_original)
    assert r['certified_success_pct']==100*float(primary.mean())and r['original_accuracy_pct']==100*float(z['n20'].mean())and r['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())and r['coverage_pct']==100*float(found.sum()/denom.sum())
    row=dict(arm=arm,seed=seed,unique_inputs=4096,chains=20,certified_valid_inputs=int(primary.sum()),original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),
        wrong_stops=wrong,capped_chains=int((~stopped).sum()),proposed_swaps=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),search_seconds=r['search_seconds'],model_sha256=r['model_sha256'],initial_file_sha256=r['initial_file_sha256'],initial_unpacked_bytes_sha256=r['initial_unpacked_bytes_sha256'],records_sha256=r['records_sha256'])
    diagnostic=dict(arm=arm,seed=seed,all_states_q13=True,all_givens_preserved=True,returned_permutation_states=int(permutation.sum()),returned_states=len(flat),all_integer_conflicts_exact=True,wrong_stops=wrong,
        stopped_nonzero_conflicts=0,stopped_nonzero_exact_energy=0,initial_zero_conflict_states_preserved=int(done.sum()),running_energy_absolute_error_max=float(error.max()),numerical_gate_passed=bool(error.max()<=.1),
        exact_energy_original_accuracy_pct=100*float(valid[ids,exact_original].mean()),exact_energy_certificate_accuracy_pct=100*float(valid[ids,exact_selected].mean()),
        exact_original_selector_disagreements=int((original!=exact_original).sum()),exact_certificate_selector_disagreements=int((selected!=exact_selected).sum()),
        failed_certified_despite_capacity=int(((capacity>0)&~primary).sum()),fallback_success_without_capacity=int(((capacity==0)&z['n20_certificate']).sum()),
        valid_capped_chains=int((~stopped&valid).sum()),invalid_capped_chains=int((~stopped&~valid).sum()))
    return z,row,diagnostic

def main():
    start=time.perf_counter();PUBLIC.mkdir(parents=True,exist_ok=True);p=read(SRC/'protocol.json');pre=read(OUT/'preflight.json');assert pre['status']=='passed'
    for group in['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    for rel,h in pre['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    dataset=read(OUT/'dataset.json');data=dict(np.load(OUT/'data.npz'));m=dict(np.load(OUT/'model.npz'));keys={row.tobytes()for row in data['all_solutions']};capacity,capinfo=OLD.capacity(m,data)
    rows=[];perseed=[];diagnostics=[];records={}
    for cfg in p['arms']:
        jobs=[]
        for seed in[0,1,2]:
            z,row,diagnostic=audit(cfg,seed,m,data,keys,capacity);jobs.append(z);perseed.append(row);diagnostics.append(diagnostic)
        arm=cfg['arm'];records[arm]=dict(n20=np.stack([z['certified_success']for z in jobs]),found=np.stack([z['found']for z in jobs]));subset=[r for r in perseed if r['arm']==arm]
        seconds=sum(r['search_seconds']for r in subset);proposals=sum(r['proposed_swaps']for r in subset)
        rows.append(dict(arm=arm,initialization=cfg['initialization'],kernel=cfg['kernel'],unique_inputs=4096,chains=20,search_seeds=[0,1,2],certified_success_pct=100*float(records[arm]['n20'].mean()),
            original_accuracy_pct=100*float(np.stack([z['n20']for z in jobs]).mean()),certificate_first_accuracy_pct=100*float(np.stack([z['n20_certificate']for z in jobs]).mean()),
            coverage_pct=100*float(records[arm]['found'].sum()/(3*jobs[0]['denom'].sum())),wrong_stops=sum(r['wrong_stops']for r in subset),capped_chains=sum(r['capped_chains']for r in subset),
            proposed_swaps=proposals,accepted_swaps=sum(r['accepted_swaps']for r in subset),search_seconds_total=seconds,amortized_nanoseconds_per_proposal=1e9*seconds/proposals))
    shared=[]
    for seed in[0,1,2]:
        a=next(r for r in perseed if r['arm']=='permutation-init-arbitrary-swaps'and r['seed']==seed);b=next(r for r in perseed if r['arm']=='permutation-init-permutation-swaps'and r['seed']==seed)
        assert a['initial_file_sha256']==b['initial_file_sha256']and a['initial_unpacked_bytes_sha256']==b['initial_unpacked_bytes_sha256'];shared.append(dict(seed=seed,shared_file_sha256=a['initial_file_sha256'],shared_unpacked_bytes_sha256=a['initial_unpacked_bytes_sha256']))
    comparisons=[];denom=np.minimum(data['test_solution_counts'],20)
    for name,a,b in[('proposal_effect','permutation-init-permutation-swaps','permutation-init-arbitrary-swaps'),('initialization_effect','permutation-init-arbitrary-swaps','arbitrary-init-arbitrary-swaps'),('joint_effect','permutation-init-permutation-swaps','arbitrary-init-arbitrary-swaps')]:
        comparisons.append(dict(effect=name,arm=a,control=b,primary=name=='proposal_effect',**STATS.bootstrap(records[a],records[b],denom)))
    primary=comparisons[0];wrong=next(r['wrong_stops']for r in rows if r['arm']==primary['arm']);numeric=all(d['numerical_gate_passed']for d in diagnostics if d['arm']in[primary['arm'],primary['control']])
    decision='falsified'if wrong or primary['accuracy_one_sided_95_upper_pp']<50 else'inconclusive_numerical_audit'if not numeric else'supported'if primary['accuracy_one_sided_95_lower_pp']>=50 else'inconclusive'
    primary.update(decision=decision,required_gain_pp=50.,wrong_stops=wrong,numerical_gate_passed=numeric)
    workers=[read(OUT/f'worker-g{gpu}.json')for gpu in[0,1]];assert all(w['status']=='complete'for w in workers);budget=read(OUT/'global-budget.json');seconds=sum((dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds()for w in workers)
    assert seconds<=1200 and max(dt.datetime.fromisoformat(w['completed_utc']).timestamp()for w in workers)<=budget['deadline_unix']
    validation=read(ROOT/'results/research/loki/h3v/audit-count-validation.json');assert validation['status']=='passed';sha(SRC.parent/'h3w/analyze.py');sha(__file__)
    np.savez_compressed(PUBLIC/'per-input-capacity.npz',keys=data['test_keys'],zero_conflict_completions=capacity,total_completions=data['test_solution_counts'])
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3aa',decision=decision,primary=primary,comparisons=comparisons,rows=rows,dataset=dataset,
        shared_initial_state_identities=shared,exact_zero_set_capacity=capinfo,output_diagnostics=diagnostics,source_gates=[read(OUT/f'gate-{device}.json')for device in['cpu','cuda']],
        preflight=pre,worker_receipts=workers,global_budget=budget,gpu_worker_seconds=seconds,independent_saved_boards=9*4096*20,optimizer_refits=0,identity_mappings=0,
        limits=p['limits'],input_sha256=HASHES,capacity_export_sha256=sha(PUBLIC/'per-input-capacity.npz'),audit_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary)
    with(PUBLIC/'per_seed.csv').open('w')as stream:w=csv.DictWriter(stream,fieldnames=list(perseed[0]));w.writeheader();w.writerows(perseed)
    print(json.dumps(dict(decision=decision,primary=primary,comparisons=comparisons,rows=rows,capacity=capinfo,gpu_worker_seconds=seconds,max_energy_error=max(d['running_energy_absolute_error_max']for d in diagnostics))),flush=True)
if __name__=='__main__':main()
