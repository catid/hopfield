"""Independent source-time, fresh-cost and all-state audit for H3ac."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv,datetime as dt,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3ac';OUT=ROOT/'artifacts/research/20261114/loki/h3ac';BASE=ROOT/'artifacts/research/20261114/loki';PUBLIC=ROOT/'results/research/loki/h3ac';HASHES={}
def load(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
CAP=load(SRC.parent/'h3w/analyze.py','h3ac_capacity')
def sha(p):p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def audit(bank,kernel,seed,cap,m,data,keys,capacity):
    folder=OUT/f'b{bank}/search/{kernel}/s{seed}';r=read(folder/'result.json');assert r['status']=='complete'and r['cap']==cap and r['records_sha256']==sha(folder/'records.npz')
    assert r['model_sha256']==sha(OUT/f'b{bank}/model.npz')and r['source_sha256']==sha(SRC/'driver.py')and r['backend_sha256']==sha(SRC.parent/f'h3aa/{kernel}_backend.py')and r['calibration_selection_sha256']==sha(OUT/f'b{bank}/calibration-selection.json')and r['protocol_sha256']==sha(SRC/'protocol.json')
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
    row=dict(bank=bank,kernel=kernel,seed=seed,gpu=r['gpu'],cap=cap,unique_inputs=4096,chains=20,certified_valid_inputs=int(primary.sum()),original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=wrong,capped_chains=int((~stopped).sum()),proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),worker_solve_seconds=r['worker_solve_seconds'],warmup_seconds=r['warmup_seconds'],initial_file_sha256=r['initial_file_sha256'],initial_unpacked_bytes_sha256=r['initial_unpacked_bytes_sha256'],records_sha256=r['records_sha256'])
    diagnostic=dict(bank=bank,kernel=kernel,seed=seed,all_q13_and_givens_preserved=True,returned_permutation_states=int(perm.sum()),all_integer_conflict_counts_exact=True,wrong_stops=wrong,running_energy_absolute_error_max=float(error.max()),numerical_gate_passed=bool(error.max()<=.1),
        exact_energy_original_accuracy_pct=100*float(valid[ids,exact_original].mean()),exact_energy_certificate_accuracy_pct=100*float(valid[ids,exact_selected].mean()),exact_energy_certificate_count_change=int(valid[ids,exact_selected].sum())-int(z['n20_certificate'].sum()),exact_energy_original_count_change=int(valid[ids,exact_original].sum())-int(z['n20'].sum()),failed_certified_despite_capacity=int(((capacity>0)&~primary).sum()),fallback_success_without_capacity=int(((capacity==0)&z['n20_certificate']).sum()))
    return z,row,diagnostic

def difference(a,b):
    delta=(a.astype(np.float64)-b).mean(0)*100;rng=np.random.default_rng(20261114);estimates=[]
    for _ in range(8000):estimates.append(float(delta[rng.integers(0,len(delta),len(delta))].mean()))
    lo,hi=np.quantile(estimates,[.0125,.9875]);return dict(gain_pp=float(delta.mean()),one_sided_98_75_lower_pp=float(lo),one_sided_98_75_upper_pp=float(hi),bootstrap_draws=8000,input_clusters=4096,search_seeds_per_input=3,interpretation='Familywise95% lower bounds across4banks, and separately familywise95% upper bounds. These are not simultaneous95% two-sided intervals.')

def verify_calibration(bank):
    import itertools
    import torch
    torch.set_num_threads(1)
    folder=OUT/f'b{bank}';selection=read(folder/'calibration-selection.json');metadata=read(folder/'calibration-data.json');cal=dict(np.load(folder/'calibration-data.npz'));assert metadata['data_sha256']==sha(folder/'calibration-data.npz');source=np.load(folder/'source-design.npz')['labels'];possible={k:set()for k in[1,2,3]};embedded=np.zeros((128,169),bool)
    for t,board in enumerate(source):
        cells=np.flatnonzero(board);positions=(cells//12*13+cells%12).tolist();embedded[t,positions]=True
        for clues in[1,2,3]:possible[clues].update(sum((cell+1)*170**(2-j)for j,cell in enumerate(subset))for subset in itertools.combinations(positions,clues))
    rng=np.random.default_rng(2026112710+bank);expected=np.sort(np.concatenate([rng.choice(np.array(sorted(possible[k]),np.int64),n,replace=False)for k,n in[(1,2),(2,205),(3,3889)]]));assert np.array_equal(expected,cal['keys'])
    decoded=np.zeros((len(expected),169),bool)
    for power in[2,1,0]:
        cells=expected//170**power%170;ok=cells>0;decoded[np.flatnonzero(ok),cells[ok]-1]=True
    assert np.array_equal(decoded,cal['inputs'])
    for given in cal['inputs']:assert embedded[:,given].all(1).any()
    for r in metadata['initial_states']:
        path=ROOT/r['path'];assert sha(path)==r['file_sha256'];initial=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:169].astype(bool);assert hashlib.sha256(initial.tobytes()).hexdigest()==r['unpacked_bytes_sha256']
        assert(initial>=cal['inputs'][:,None,:]).all()and(initial.sum(-1)==13).all()
        INITIAL=load(SRC.parent/'h3aa/initialization.py','h3ac_initial_audit');assert np.array_equal(initial,INITIAL.make(cal['inputs'],q=13,chains=20,seed=r['seed'],kind='permutation'))
    groups=selection['all_calibration_groups'];ref=groups['arbitrary-c20000'];best=min([g for g in groups.values()if g['kernel']=='permutation'],key=lambda g:(abs(np.median([r['worker_solve_seconds']for r in g['trials']])/ref['median_seconds']-1),g['cap']))
    for group in groups.values():
        assert len(group['trials'])==6 and{(r['gpu'],r['seed'])for r in group['trials']}=={(g,s)for g in[0,1]for s in[101,102,103]};assert group['median_seconds']==float(np.median([r['worker_solve_seconds']for r in group['trials']]))
        for r in group['trials']:assert r['model_sha256']==sha(folder/'model.npz')and r['calibration_data_sha256']==sha(folder/'calibration-data.npz')and r['source_sha256']==sha(SRC/'calibration.py')and r['protocol_sha256']==sha(SRC/'protocol.json')and r['no_target_accuracy_computed']
    for rel,h in selection['calibration_result_sha256'].items():assert sha(ROOT/rel)==h
    assert best['cap']==selection['selected_permutation_cap'];ratio=best['median_seconds']/ref['median_seconds'];assert ratio==selection['source_median_time_ratio'];assert (abs(ratio-1)<=.05)==(selection['status']=='matched_source_median_times')
    if selection['interpolation_used']:
        request=read(folder/'refinement-request.json');old=request['initial_grid_groups'];pairs=sorted([v for v in old.values()if v['kernel']=='permutation'],key=lambda v:v['cap']);a,b=next((a,b)for a,b in zip(pairs[:-1],pairs[1:])if a['median_seconds']<old['arbitrary-c20000']['median_seconds']<b['median_seconds']);raw=a['cap']+(old['arbitrary-c20000']['median_seconds']-a['median_seconds'])/(b['median_seconds']-a['median_seconds'])*(b['cap']-a['cap']);assert request['cap']==int(np.rint(raw/128)*128)
    return selection,metadata

def source_audit(bank,population):
    import torch
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0,str(SRC));SOURCE=load(SRC/'source.py','h3ac_source_audit');folder=OUT/f'b{bank}';d=dict(np.load(folder/'source-design.npz'));indices=np.sort(np.random.default_rng(2026112700+bank).choice(176,128,replace=False));assert np.array_equal(indices,d['population_indices'])and np.array_equal(d['labels'],population[indices]);literal=SOURCE.independent_design(d);params=np.load(folder/'parameters.npz');m,info=SOURCE.construct(torch.tensor(params['A']),torch.tensor(params['b']),d);stored=dict(np.load(folder/'model.npz'));r=read(folder/'model.json')
    for k,v in m.items():assert np.array_equal(v,stored[k]),(bank,k)
    for k,v in info.items():assert r[k]==v
    assert r['parameters_sha256']==sha(folder/'parameters.npz')and r['source_design_sha256']==sha(folder/'source-design.npz')and r['model_sha256']==sha(folder/'model.npz')and r['protocol_sha256']==sha(SRC/'protocol.json');assert r['source_eligible'];validation=read(folder/'loss-validation.json');assert validation['status']=='passed'
    i,j=np.triu_indices(169,1);dr=j//13-i//13;dc=j%13-i%13;truth=(dr==0)|(dc==0)|(abs(dr)==abs(dc))|((dr==1)&(abs(dc)==3))|((dr==3)&(abs(dc)==1));support=stored['F'][i,j].astype(bool)
    return stored,dict(bank=bank,**literal,**info,source_indices=indices.tolist(),source_design_sha256=sha(folder/'source-design.npz'),model_sha256=sha(folder/'model.npz'),fit_seconds=r['fit_seconds'],source_loss_validation=validation,support_array_sha256=hashlib.sha256(stored['F'].tobytes()).hexdigest(),weights_array_sha256=hashlib.sha256(stored['J'].tobytes()).hexdigest(),target_true_forbidden_pairs=int(truth.sum()),target_false_forbidden_pairs=int((support&~truth).sum()),target_missed_forbidden_pairs=int((~support&truth).sum()))

def main():
    start=time.perf_counter();PUBLIC.mkdir(parents=True,exist_ok=True);p=read(SRC/'protocol.json');campaign=read(OUT/'campaign.json');assert campaign['status']=='complete'
    for group in['source_sha256','source_data_sha256','target_data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    source_banks=read(OUT/'source-banks.json');assert dt.datetime.fromisoformat(p['frozen_utc'])<dt.datetime.fromisoformat(source_banks['utc']);dataset=read(OUT/'dataset.json');pre=read(OUT/'preflight.json');assert pre['status']=='passed'
    for rel,h in pre['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    data=dict(np.load(OUT/'data.npz'));previous=np.load(BASE/'h3ab/data.npz');prior=np.union1d(previous['all_excluded_keys'],previous['test_keys']);calkeys=np.unique(np.concatenate([np.load(OUT/f'b{k}/calibration-data.npz')['keys']for k in range(4)]));excluded=np.union1d(prior,calkeys);universe=np.load(BASE/'h3t/input-universe.npz');eligible=np.setdiff1d(universe['keys'],excluded);expected=np.sort(np.random.default_rng(2026112516).permutation(eligible)[:4096]);assert np.array_equal(expected,data['test_keys'])and np.array_equal(excluded,data['all_excluded_keys'])and np.array_equal(prior,data['prior_search_and_calibration_keys'])and not np.isin(expected,excluded).any()
    decoded=np.zeros((len(expected),169),bool)
    for power in[2,1,0]:
        cells=expected//170**power%170;ok=cells>0;decoded[np.flatnonzero(ok),cells[ok]-1]=True
    assert np.array_equal(decoded,data['test_inputs'])
    keys={r.tobytes()for r in data['all_solutions']};population=np.load(BASE/'h3t/solutions-n12.npy');perseed=[];diagnostics=[];bankresults=[];sourceinfo=[];allrows=[];capacity_export={'keys':data['test_keys'],'total_completions':data['test_solution_counts']};selections={}
    for bank in range(4):
        model,src=source_audit(bank,population);sourceinfo.append(src);selection,calmetadata=verify_calibration(bank);selections[str(bank)]=selection;assert dt.datetime.fromisoformat(selection['utc'])<dt.datetime.fromisoformat(dataset['utc']);capacity,capinfo=CAP.capacity(model,data);capacity_export[f'b{bank}']=capacity;rows=[];records={};diags=[];jobsby={}
        if selection['status']!='matched_source_median_times':
            bankresults.append(dict(bank=bank,decision='inconclusive_source_timing_match',quality_decision='not_run',source_time_ratio=selection['source_median_time_ratio'],source_time_gate_passed=False,exact_zero_set_capacity=capinfo,rows=[]));continue
        for kernel in['arbitrary','permutation']:
            cap=dataset['selected_caps'][str(bank)][kernel];jobs=[];subset=[]
            for seed in[0,1,2]:z,row,diag=audit(bank,kernel,seed,cap,model,data,keys,capacity);jobs.append(z);perseed.append(row);subset.append(row);diagnostics.append(diag);diags.append(diag)
            jobsby[kernel]=jobs;records[kernel]=np.stack([z['certified_success']for z in jobs]);times=[r['worker_solve_seconds']for r in subset];proposals=sum(r['proposal_moves']for r in subset)
            rows.append(dict(bank=bank,arm='BP AdamW '+kernel,kernel=kernel,cap=cap,certified_success_pct=100*float(records[kernel].mean()),original_accuracy_pct=100*float(np.stack([z['n20']for z in jobs]).mean()),certificate_first_accuracy_pct=100*float(np.stack([z['n20_certificate']for z in jobs]).mean()),coverage_pct=100*float(np.stack([z['found']for z in jobs]).sum()/(3*jobs[0]['denom'].sum())),wrong_stops=sum(r['wrong_stops']for r in subset),proposal_moves=proposals,accepted_swaps=sum(r['accepted_swaps']for r in subset),median_worker_solve_seconds=float(np.median(times)),minimum_worker_solve_seconds=min(times),maximum_worker_solve_seconds=max(times),total_worker_solve_seconds=sum(times),amortized_nanoseconds_per_proposal=1e9*sum(times)/proposals))
        delta=difference(records['permutation'],records['arbitrary']);wrong=sum(r['wrong_stops']for r in rows);numeric=all(d['numerical_gate_passed']for d in diags);quality='falsified'if wrong or delta['one_sided_98_75_upper_pp']<50 else'inconclusive_numerical_audit'if not numeric else'supported'if delta['one_sided_98_75_lower_pp']>=50 else'inconclusive';ratio=rows[1]['median_worker_solve_seconds']/rows[0]['median_worker_solve_seconds'];match=abs(ratio-1)<=.10;decision='falsified'if quality=='falsified'else'inconclusive_target_timing_mismatch'if not match else quality
        bankresults.append(dict(bank=bank,decision=decision,quality_decision=quality,primary=delta,wrong_stops=wrong,numerical_gate_passed=numeric,target_time_ratio=ratio,target_time_gate_passed=match,source_time_ratio=selection['source_median_time_ratio'],source_time_gate_passed=True,rows=rows,exact_zero_set_capacity=capinfo));allrows.extend(rows);print('AUDITED',bank,decision,delta,ratio,flush=True)
    for seed in[0,1,2]:
        group=[r for r in perseed if r['seed']==seed];assert len({r['initial_file_sha256']for r in group})==len({r['initial_unpacked_bytes_sha256']for r in group})==1
    overlap=[[len(set(a['source_indices'])&set(b['source_indices']))for b in sourceinfo]for a in sourceinfo];assert overlap==source_banks['overlap_counts'];decisions=[b['decision']for b in bankresults];decision='falsified'if'falsified'in decisions else'supported'if all(d=='supported'for d in decisions)else'inconclusive';workers=[read(path)for path in sorted(OUT.rglob('*worker-*.json'))];assert all(w['status']=='complete'for w in workers);budget=read(OUT/'global-budget.json');seconds=lambda w:(dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds();total=sum(map(seconds,workers));latest=max(dt.datetime.fromisoformat(w['completed_utc']).timestamp()for w in workers);assert total<=3600 and latest<=budget['deadline_unix'];workerphases={phase:sum(seconds(w)for w in workers if w['phase']==phase)for phase in sorted({w['phase']for w in workers})}
    np.savez_compressed(PUBLIC/'per-input-capacity.npz',**capacity_export);sha(SRC/'analyze.py');sha(SRC.parent/'h3w/analyze.py');count_validation=read(ROOT/'results/research/loki/h3v/audit-count-validation.json');assert count_validation['status']=='passed'
    summary=dict(status='validated',utc=now(),hypothesis='H3ac',decision=decision,bank_results=bankresults,rows=allrows,source_banks=sourceinfo,source_bank_overlap_counts=overlap,independent_source_bank_draws=4,independent_rule_families=1,distinct_support_arrays=len({v['support_array_sha256']for v in sourceinfo}),distinct_fitted_weight_arrays=len({v['weights_array_sha256']for v in sourceinfo}),optimizer_fits=4,dataset=dataset,calibration_selections=selections,output_diagnostics=diagnostics,shared_per_seed_initial_bytes_all_banks_and_kernels=True,audited_timed_target_states=len(perseed)*4096*20,total_gpu_worker_seconds=total,worker_seconds_by_phase=workerphases,whole_study_wall_seconds=latest-(budget['deadline_unix']-1800),fit_seconds_total=sum(s['fit_seconds']for s in sourceinfo),worker_receipts=workers,global_budget=budget,limits=p['limits'],input_sha256=HASHES,capacity_export_sha256=sha(PUBLIC/'per-input-capacity.npz'),audit_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary)
    with(PUBLIC/'per_seed.csv').open('w')as stream:w=csv.DictWriter(stream,fieldnames=list(perseed[0]));w.writeheader();w.writerows(perseed)
    save(PUBLIC/'calibration-selections.json',selections);save(PUBLIC/'source-banks.json',source_banks)
    for bank in range(4):
        folder=PUBLIC/f'b{bank}';folder.mkdir(exist_ok=True)
        for name in['parameters.npz','source-design.npz','model.npz','model.json','loss-validation.json']:(folder/name).write_bytes((OUT/f'b{bank}'/name).read_bytes())
    print(json.dumps({k:summary[k]for k in['decision','total_gpu_worker_seconds','worker_seconds_by_phase','whole_study_wall_seconds','fit_seconds_total']}),flush=True)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
if __name__=='__main__':main()
