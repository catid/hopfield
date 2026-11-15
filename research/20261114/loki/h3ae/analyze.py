"""Audit frozen GPU controls and every exact-cache return, including full costs."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3ae';OUT=ROOT/'artifacts/research/20261114/loki/h3ae';OLD=ROOT/'artifacts/research/20261114/loki/h3ad';PUBLIC=ROOT/'results/research/loki/h3ae';HASHES={}
def sha(p):p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
GPU=load(SRC.parent/'h3ad/analyze.py','h3ae_GPU_audit');GPU.ROOT=ROOT;GPU.SRC=SRC;GPU.OUT=OUT;GPU.sha=sha;GPU.read=read;CAP=GPU.CAP

def audit_query(bank,seed,cache,data,keys):
    folder=OUT/f'b{bank}/exact/s{seed}';r=read(folder/'result.json');assert r['source_sha256']==sha(SRC/'query.py')and r['protocol_sha256']==sha(SRC/'protocol.json')and r['records_sha256']==sha(folder/'records.npz')and r['source_cache_sha256']==sha(OUT/f'b{bank}/cache.npz');inputs=data['test_inputs'];assert r['inputs_bytes_sha256']==hashlib.sha256(inputs.tobytes()).hexdigest();z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:169].astype(bool);expected=np.array([cache[:,given].all(1).sum()for given in inputs]);assert np.array_equal(z['available'],expected)and np.array_equal(z['abstained'],expected==0)and np.array_equal(z['stopped'],np.repeat((expected>0)[:,None],20,axis=1));valid=[];found=[];denom=np.minimum(data['test_solution_counts'],20)
    for t,(samples,given)in enumerate(zip(boards,inputs)):
        if expected[t]:
            ids=z['indices'][t];assert((ids>=0)&(ids<len(cache))).all()and np.array_equal(samples,cache[ids])and(samples.sum(1)==13).all()and(samples>=given).all();assert len(set(ids))==min(20,expected[t])
        else:assert(z['indices'][t]==-1).all()and not samples.any()
        ok=np.array([b.tobytes()in keys and b[given].all()for b in samples]);valid.append(ok);found.append(min(len({b.tobytes()for b,v in zip(samples,ok)if v}),int(denom[t])))
    valid=np.array(valid);found=np.array(found);primary=(valid&z['stopped']).any(1);wrong=int((z['stopped']&~valid).sum());row=dict(bank=bank,kernel='exact',seed=seed,unique_inputs=4096,outputs=20,certified_valid_inputs=int(primary.sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),abstained_inputs=int(z['abstained'].sum()),wrong_stops=wrong,query_seconds=r['query_seconds'],records_sha256=r['records_sha256']);diag=dict(bank=bank,seed=seed,every_output_identically_maps_to_verified_zero_F_cache=True,all_literal_clue_match_counts_verified=True,unique_then_padding_contract_verified=True,abstention_sentinels_verified=True,target_labels_only_used_in_post_query_audit=True,wrong_stops=wrong)
    return primary,found,row,diag

def main():
    start=time.perf_counter();PUBLIC.mkdir(parents=True,exist_ok=True);p=read(SRC/'protocol.json');assert read(OUT/'campaign.json')['status']=='complete'
    for group in['source_sha256','source_data_sha256','target_data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    gate=read(OUT/'compilation-gate.json');assert gate['status']=='complete_verified'and gate['aggregate_cpu_seconds']<=120 and gate['protocol_sha256']==sha(SRC/'protocol.json');query_worker=read(OUT/'query-worker.json');assert query_worker['status']=='complete';previous=read(OLD/'datasets.json');prior=np.union1d(previous['all_prior_and_calibration_excluded_keys'],previous['all_new_family_test_keys']);prior_new=np.array([],np.int64);families=[];allrows=[];gpuper=[];exactper=[];gpudiags=[];querydiags=[];capacities={}
    for bank,cfg in enumerate(p['models']):
        folder=OUT/f'b{bank}';comp=read(folder/'compilation.json');assert comp['source_model_sha256']==cfg['model_sha256']==sha(folder/'model.npz')and comp['cache_sha256']==sha(folder/'cache.npz')and comp['source_sha256']==sha(SRC/'compiler.py');m=dict(np.load(folder/'model.npz'));cachedata=np.load(folder/'cache.npz');cache=cachedata['boards'];assert len(cache)==comp['zero_F_solutions']and np.array_equal(np.packbits(cache.T,axis=1,bitorder='little'),cachedata['index']);columns=cachedata['columns'];reconstructed=np.zeros_like(cache);reconstructed[np.arange(len(cache))[:,None],13*np.arange(13)+columns]=True;assert np.array_equal(cache,reconstructed)and(cache.sum(1)==13).all()and len(np.unique(cache,axis=0))==len(cache);x=cache.astype(np.int64);assert not(x*(x@m['F'].astype(np.int64))).sum(1).any();assert not m['h'].any()and(m['J'][m['F'].astype(bool)]>0).all()and not m['J'][~m['F'].astype(bool)].any()
        dataset=read(folder/'dataset.json');pre=read(folder/'preflight.json');assert pre['status']=='passed'
        for rel,h in pre['input_sha256'].items():assert sha(ROOT/rel)==h,rel
        assert dt.datetime.fromisoformat(gate['utc'])<dt.datetime.fromisoformat(dataset['utc']);data=dict(np.load(folder/'data.npz'));assert np.array_equal(np.load(folder/'query-inputs.npy'),data['test_inputs']);assert np.array_equal(cache,data['all_solutions']) # Post-compilation diagnostic: both fixed models have exact attack supports.
        universe=np.load(OLD/f'b{bank}/input-universe.npz');excluded=np.union1d(prior,prior_new);eligible=np.setdiff1d(universe['keys'],excluded);expected=np.sort(np.random.default_rng(2026112519+bank).permutation(eligible)[:4096]);assert np.array_equal(expected,data['test_keys'])and np.array_equal(excluded,data['all_excluded_keys'])and not np.isin(expected,excluded).any();decoded=np.zeros((4096,169),bool)
        for power in[2,1,0]:
            cells=expected//170**power%170;ok=cells>0;decoded[np.flatnonzero(ok),cells[ok]-1]=True
        assert np.array_equal(decoded,data['test_inputs']);prior_new=np.union1d(prior_new,expected);capacity,capinfo=CAP.capacity(m,data);capacities[f'b{bank}_keys']=expected;capacities[f'b{bank}_capacity']=capacity;capacities[f'b{bank}_completions']=data['test_solution_counts'];keys={b.tobytes()for b in data['all_solutions']};rows=[];primaries={};localdiags=[]
        for kernel in['arbitrary','permutation']:
            cap=dataset['selected_caps'][kernel];jobs=[];subset=[]
            for seed in[0,1,2]:z,row,diag=GPU.audit(bank,kernel,seed,cap,m,data,keys,capacity);jobs.append(z);subset.append(row);gpuper.append(row);gpudiags.append(diag);localdiags.append(diag)
            primaries[kernel]=np.stack([z['certified_success']for z in jobs]);times=[r['worker_solve_seconds']for r in subset];rows.append(dict(bank=bank,family=cfg['family'],arm='BP AdamW '+kernel,kernel=kernel,cap=cap,certified_success_pct=100*float(primaries[kernel].mean()),certificate_first_accuracy_pct=100*float(np.stack([z['n20_certificate']for z in jobs]).mean()),coverage_pct=100*float(np.stack([z['found']for z in jobs]).sum()/(3*jobs[0]['denom'].sum())),wrong_stops=sum(r['wrong_stops']for r in subset),median_solve_seconds=float(np.median(times)),three_solve_seconds=sum(times),proposals=sum(r['proposal_moves']for r in subset),accepted=sum(r['accepted_swaps']for r in subset)))
        for seed in[0,1,2]:
            group=[r for r in gpuper if r['bank']==bank and r['seed']==seed];assert len({r['initial_file_sha256']for r in group})==len({r['initial_unpacked_bytes_sha256']for r in group})==1
        qs=[];found=[];subset=[]
        for seed in[0,1,2]:primary,f,r,d=audit_query(bank,seed,cache,data,keys);qs.append(primary);found.append(f);subset.append(r);exactper.append(r);querydiags.append(d)
        primaries['exact']=np.stack(qs);qtime=query_worker['families'][bank];assert abs(qtime['three_query_seconds']-sum(r['query_seconds']for r in subset))<1e-12 and qtime['compile_wall_seconds']==comp['compile_wall_seconds'];cost=comp['compile_wall_seconds']+qtime['cache_load_seconds']+sum(r['query_seconds']for r in subset);assert cost==qtime['cpu_compile_load_three_query_seconds'];exact=dict(bank=bank,family=cfg['family'],arm='BP AdamW exact zero-F compilation',kernel='exact',certified_success_pct=100*float(primaries['exact'].mean()),coverage_pct=100*float(np.stack(found).sum()/(3*np.minimum(data['test_solution_counts'],20).sum())),wrong_stops=sum(r['wrong_stops']for r in subset),abstained_inputs=sum(r['abstained_inputs']for r in subset),compile_wall_seconds=comp['compile_wall_seconds'],cache_load_seconds=qtime['cache_load_seconds'],three_query_seconds=qtime['three_query_seconds'],compile_load_three_query_seconds=cost);rows.append(exact)
        gain=exact['certified_success_pct']-rows[1]['certified_success_pct'];numeric=all(d['numerical_gate_passed']for d in localdiags);wrong=sum(r['wrong_stops']for r in rows);quality=bool(gain>=-.1 and wrong==0 and numeric);speed=rows[1]['three_solve_seconds']/cost;decision='supported'if quality and speed>=10 else'falsified';families.append(dict(bank=bank,family=cfg['family'],decision=decision,quality_correctness_gate_passed=quality,empirical_exact_minus_permutation_pp=gain,wrong_stops=wrong,numerical_GPU_gate_passed=numeric,speed_ratio=speed,required_speed_ratio=10.,CPU_denominator_seconds=cost,GPU_numerator_seconds=rows[1]['three_solve_seconds'],compilation=comp,complete_cache_identical_to_independently_generated_true_target_bank=True,rows=rows,dataset=dataset,exact_zero_set_capacity=capinfo));allrows.extend(rows);print('AUDITED',cfg['family'],decision,'speed',speed,'gain',gain,flush=True)
    workers=[read(path)for path in sorted(OUT.glob('b*/target-worker-g*.json'))];assert all(w['status']=='complete'for w in workers);budget=read(OUT/'global-budget.json');duration=lambda w:(dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds();total=sum(map(duration,workers));bygpu={str(g):sum(duration(w)for w in workers if w['gpu']==g)for g in[0,1]};latest=max(dt.datetime.fromisoformat(w['completed_utc']).timestamp()for w in workers);assert total<=300 and all(t<=150 for t in bygpu.values())and latest<=budget['deadline_unix'];decision='supported'if all(f['decision']=='supported'for f in families)else'falsified';np.savez_compressed(PUBLIC/'per-input-capacity.npz',**capacities);sha(__file__);sha(SRC.parent/'h3ad/analyze.py');sha(SRC.parent/'h3w/analyze.py')
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3ae',decision=decision,family_results=families,rows=allrows,compilation_gate=gate,query_worker=query_worker,GPU_output_diagnostics=gpudiags,exact_query_diagnostics=querydiags,audited_GPU_states=len(gpuper)*4096*20,audited_exact_returned_states=len(exactper)*4096*20,all_exact_returns_checked_by_byte_identity_to_independently_F_verified_cache=True,total_gpu_worker_seconds=total,gpu_worker_seconds_by_device=bygpu,GPU_phase_wall_seconds=latest-(budget['deadline_unix']-180),worker_receipts=workers,global_budget=budget,limits=p['limits'],input_sha256=HASHES,capacity_export_sha256=sha(PUBLIC/'per-input-capacity.npz'),audit_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary)
    for name,rows in[('gpu_per_seed.csv',gpuper),('exact_per_seed.csv',exactper)]:
        with(PUBLIC/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for bank in range(2):
        folder=PUBLIC/f'b{bank}';folder.mkdir(exist_ok=True)
        for name in['cache.npz','compilation.json','model.npz','dataset.json']:(folder/name).write_bytes((OUT/f'b{bank}'/name).read_bytes())
    for name in['compilation-gate.json','query-worker.json','query-validation.json']:(PUBLIC/name).write_bytes((OUT/name).read_bytes())
    print(json.dumps({k:summary[k]for k in['decision','total_gpu_worker_seconds','gpu_worker_seconds_by_device','GPU_phase_wall_seconds']}),flush=True)
if __name__=='__main__':main()
