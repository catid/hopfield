"""Independent N15 compilation, cohort, returned-state and full-cost audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
import gpu_audit as GPU
PUBLIC=ROOT/'results/research/loki/h3ag'
HASHES={}
def hashed(path):
    path=Path(path);value=sha(path);HASHES[str(path.relative_to(ROOT))]=value;return value

def audit_query(bank,seed,cache,data):
    folder=OUT/f'b{bank}/exact/s{seed}';r=read(folder/'result.json');assert r['source_sha256']==hashed(SRC/'query.py')and r['protocol_sha256']==hashed(SRC/'protocol.json')and r['records_sha256']==hashed(folder/'records.npz')and r['source_cache_sha256']==hashed(OUT/f'b{bank}/cache.npz')
    inputs=data['test_inputs'];assert r['inputs_bytes_sha256']==hashlib.sha256(inputs.tobytes()).hexdigest();z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:225].astype(bool);expected=np.array([cache[:,given].all(1).sum()for given in inputs]);assert np.array_equal(z['available'],expected)and np.array_equal(z['abstained'],expected==0)and np.array_equal(z['stopped'],np.repeat((expected>0)[:,None],20,axis=1));found=[];denom=np.minimum(data['test_solution_counts'],20)
    for t,(samples,given)in enumerate(zip(boards,inputs)):
        assert expected[t]>0
        ids=z['indices'][t];assert((ids>=0)&(ids<len(cache))).all()and np.array_equal(samples,cache[ids])and(samples.sum(1)==15).all()and(samples>=given).all();assert len(set(ids))==min(20,expected[t]);found.append(len(set(ids)))
    found=np.array(found);assert np.array_equal(found,denom);primary=np.ones(len(inputs),bool)
    row=dict(bank=bank,kernel='exact',seed=seed,unique_inputs=4096,outputs=20,certified_valid_inputs=int(primary.sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),abstained_inputs=0,wrong_stops=0,query_seconds=r['query_seconds'],records_sha256=r['records_sha256'])
    return primary,found,row,dict(bank=bank,seed=seed,every_return_matches_verified_cache_id=True,literal_clue_match_counts_verified=True,unique_then_padding_verified=True,all_q15_and_givens_preserved=True,wrong_stops=0)

def cohort_audit(bank,cache,columns,data,excluded,p):
    folder=OUT/f'b{bank}';base=226;hist=np.zeros(base**3,np.int64)
    # Independently accumulate every 1/2/3-clue key from the proved-complete cache.
    for clues in [1,2,3]:
        ix=np.array(list(itertools.combinations(range(15),clues)));powers=np.array([base**(2-j)for j in range(clues)],np.int64)
        for lo in range(0,len(columns),2048):
            cells=15*np.arange(15)[None,:]+columns[lo:lo+2048];encoded=((cells[:,ix]+1)*powers).sum(-1);hist+=np.bincount(encoded.reshape(-1),minlength=base**3)
    universe=np.flatnonzero(hist);saved=np.load(folder/'input-universe.npz');assert np.array_equal(universe,saved['keys'])and np.array_equal(hist[universe],saved['counts'])and hist.sum()==len(cache)*575
    eligible=np.setdiff1d(universe,excluded);keys=np.sort(np.random.default_rng(p['target']['input_seeds'][bank]).permutation(eligible)[:4096]);assert np.array_equal(keys,data['test_keys'])and np.array_equal(excluded,data['all_excluded_keys'])and np.array_equal(hist[keys],data['test_solution_counts']);decoded=np.zeros((4096,225),bool)
    for power in [2,1,0]:
        cells=keys//base**power%base;ok=cells>0;decoded[np.flatnonzero(ok),cells[ok]-1]=True
    assert np.array_equal(decoded,data['test_inputs'])and np.array_equal(np.load(folder/'query-inputs.npy'),decoded)
    literal=np.array([cache[:,g].all(1).sum()for g in decoded]);assert np.array_equal(literal,data['test_solution_counts'])and(literal>0).all()
    return keys,literal,dict(all_1_2_3_clue_keys_reconstructed=True,all_compatible_counts_literal_checked=True,uniform_remaining_eligible_selection_replayed=True,complete_universe_keys=len(universe),eligible_keys=len(eligible),excluded_keys=len(excluded))

def main():
    start=time.perf_counter();p=frozen(target=True);assert read(OUT/'campaign.json')['status']=='complete';PUBLIC.mkdir(parents=True,exist_ok=True)
    gate=read(OUT/'compilation-gate.json');graph=read(OUT/'external-graph-gate.json');validation=read(OUT/'verifier-validation.json');query=read(OUT/'query-worker.json');assert gate['status']=='complete_verified'and gate['aggregate_cpu_seconds']<=120 and gate['wall_seconds']<=120 and graph['status']=='passed'and validation['status']=='passed'and query['status']=='complete';assert len(validation['cases'])==44 and all(r['full_solution_set_parity']for r in validation['cases']);assert validation['binary_sha256']==hashed(OUT/'reverse-literal')and validation['cpp_source_sha256']==hashed(SRC/'reverse_literal.cpp')
    for rel,h in p['source_sha256'].items():assert hashed(ROOT/rel)==h
    GPU.sha=hashed
    families=[];allrows=[];gpurows=[];exactrows=[];gpu_diags=[];exact_diags=[];exports={};prior=np.array([],np.int64)
    for bank,cfg in enumerate(p['models']):
        folder=OUT/f'b{bank}';m=dict(np.load(folder/'model.npz'));comp=read(folder/'compilation.json');cachedata=dict(np.load(folder/'cache.npz'));cache=cachedata['boards'];columns=cachedata['columns'];assert comp['source_model_sha256']==hashed(folder/'model.npz')and comp['cache_sha256']==hashed(folder/'cache.npz')and comp['source_sha256']==hashed(SRC/'compiler.py');assert np.array_equal(columns,np.load(folder/'unverified-forward-columns.npy'))and np.array_equal(columns,np.fromfile(OUT/f'verifier-io/N15-b{bank}.columns.bin',np.uint8).reshape(-1,15))
        assert len(cache)==comp['zero_F_solutions']and len(np.unique(columns,axis=0))==len(cache);cells=15*np.arange(15)[None,:]+columns;expected=np.zeros_like(cache);expected[np.arange(len(cache))[:,None],cells]=True;assert np.array_equal(expected,cache)and np.array_equal(np.packbits(cache.T,axis=1,bitorder='little'),cachedata['index']);ii,jj=np.triu_indices(15,1);assert not m['F'][cells[:,ii],cells[:,jj]].any();assert not m['h'].any()and(m['J'][m['F'].astype(bool)]>0).all()and not m['J'][~m['F'].astype(bool)].any()
        i,j=np.triu_indices(225,1);dr=j//15-i//15;dc=abs(j%15-i%15);a,b=[(2,3),(2,5)][bank];truth=(dr==0)|(dc==0)|(dr==dc)|((dr==a)&(dc==b))|((dr==b)&(dc==a));assert np.array_equal(truth,m['F'][i,j].astype(bool))and graph['families'][bank]['model_sha256']==hashed(folder/'model.npz');assert dt.datetime.fromisoformat(gate['utc'])<dt.datetime.fromisoformat(graph['utc'])
        dataset=read(folder/'dataset.json');pre=read(folder/'preflight.json');assert pre['status']=='passed'and dt.datetime.fromisoformat(graph['utc'])<dt.datetime.fromisoformat(dataset['utc'])
        for rel,h in pre['input_sha256'].items():assert hashed(ROOT/rel)==h
        data=dict(np.load(folder/'data.npz'));assert np.array_equal(cache,data['all_solutions']);keys,capacity,cohort=cohort_audit(bank,cache,columns,data,prior,p);prior=np.union1d(prior,keys);exports[f'b{bank}_keys']=keys;exports[f'b{bank}_completions']=capacity;rows=[];primaries={};local=[];valid_keys={row.tobytes()for row in cache}
        for kernel in ['arbitrary','permutation']:
            cap=dataset['selected_caps'][kernel];jobs=[];subset=[]
            for seed in [0,1,2]:
                z,r,d=GPU.audit(bank,kernel,seed,cap,m,data,valid_keys,capacity);jobs.append(z);subset.append(r);gpurows.append(r);gpu_diags.append(d);local.append(d)
            primaries[kernel]=np.stack([z['certified_success']for z in jobs]);found=np.stack([z['found']for z in jobs]);exports[f'b{bank}_{kernel}_primary']=primaries[kernel];exports[f'b{bank}_{kernel}_found']=found;times=[r['worker_solve_seconds']for r in subset];rows.append(dict(bank=bank,family=cfg['family'],arm='BP AdamW '+kernel,kernel=kernel,cap=cap,certified_success_pct=100*float(primaries[kernel].mean()),coverage_pct=100*float(found.sum()/(3*np.minimum(capacity,20).sum())),wrong_stops=sum(r['wrong_stops']for r in subset),three_solve_seconds=sum(times),median_solve_seconds=float(np.median(times)),proposals=sum(r['proposal_moves']for r in subset),accepted=sum(r['accepted_swaps']for r in subset)))
        for seed in [0,1,2]:
            group=[r for r in gpurows if r['bank']==bank and r['seed']==seed];assert len({r['initial_file_sha256']for r in group})==len({r['initial_unpacked_bytes_sha256']for r in group})==1
        qs=[];fs=[];subset=[]
        for seed in [0,1,2]:a,f,r,d=audit_query(bank,seed,cache,data);qs.append(a);fs.append(f);subset.append(r);exactrows.append(r);exact_diags.append(d)
        exports[f'b{bank}_exact_primary']=np.stack(qs);exports[f'b{bank}_exact_found']=np.stack(fs);qt=query['families'][bank];shared=gate['shared_prerequisite_wall_seconds'];fixed=shared+comp['compile_wall_seconds']+graph['families'][bank]['wall_seconds']+qt['cache_load_seconds'];queries=sum(r['query_seconds']for r in subset);denominator=fixed+queries;assert abs(denominator-qt['cpu_compile_load_three_query_seconds'])<1e-10
        exact=dict(bank=bank,family=cfg['family'],arm='BP AdamW exact zero-F compilation',kernel='exact',certified_success_pct=100*float(np.stack(qs).mean()),coverage_pct=100*float(np.stack(fs).sum()/(3*np.minimum(capacity,20).sum())),wrong_stops=0,abstained_inputs=0,shared_build_parity_monitor_seconds=shared,compile_wall_seconds=comp['compile_wall_seconds'],external_graph_gate_seconds=graph['families'][bank]['wall_seconds'],cache_load_seconds=qt['cache_load_seconds'],three_query_seconds=queries,compile_load_three_query_seconds=denominator);rows.append(exact);gain=exact['certified_success_pct']-rows[1]['certified_success_pct'];numeric=all(d['numerical_gate_passed']for d in local);wrong=sum(r['wrong_stops']for r in rows);quality=gain>=-.1 and wrong==0 and numeric;ratio=rows[1]['three_solve_seconds']/denominator;decision='supported'if quality and ratio>=10 else'falsified';tcpu=queries/3;tgpu=rows[1]['three_solve_seconds']/3
        def threshold(multiplier):
            d=tgpu-multiplier*tcpu
            return dict(fractional_batches=multiplier*fixed/d,ceiling_batches=math.ceil(multiplier*fixed/d))if d>0 else None
        amortization=dict(fixed_CPU_seconds=fixed,mean_CPU_query_seconds=tcpu,mean_GPU_solve_seconds=tgpu,break_even=threshold(1),tenfold=threshold(10),observed_batches=3,interpretation='Descriptive linear cache-reuse extrapolation beyond the three measured batches; no new measurements or tuning.')
        families.append(dict(bank=bank,family=cfg['family'],decision=decision,quality_correctness_gate_passed=quality,numerical_GPU_gate_passed=numeric,empirical_exact_minus_permutation_pp=gain,speed_ratio=ratio,required_speed_ratio=10,CPU_denominator_seconds=denominator,GPU_numerator_seconds=rows[1]['three_solve_seconds'],compilation=comp,cohort_audit=cohort,dataset=dataset,rows=rows,amortization=amortization));allrows.extend(rows);print('AUDITED',bank,decision,'ratio',ratio,flush=True)
    workers=[read(path)for path in sorted(OUT.glob('b*/target-worker-g*.json'))];assert len(workers)==4 and all(w['status']=='complete'for w in workers);budget=read(OUT/'global-budget.json');duration=lambda w:(dt.datetime.fromisoformat(w['completed_utc'])-dt.datetime.fromisoformat(w['started_utc'])).total_seconds();total=sum(map(duration,workers));bygpu={str(g):sum(duration(w)for w in workers if w['gpu']==g)for g in [0,1]};latest=max(dt.datetime.fromisoformat(w['completed_utc']).timestamp()for w in workers);assert total<=600 and all(v<=300 for v in bygpu.values())and latest<=budget['deadline_unix'];np.savez_compressed(PUBLIC/'per-input-metrics.npz',**exports)
    files=['campaign.json','compilation-gate.json','compiler-result.json','verifier-validation.json','query-validation.json','external-graph-gate.json','query-worker.json','datasets.json','global-budget.json']
    for bank in range(2):files += [f'b{bank}/{name}'for name in ['model.npz','cache.npz','compilation.json','transfer.json','dataset.json']]
    for rel in files:
        dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest);hashed(OUT/rel)
    for name,rows in [('gpu_per_seed.csv',gpurows),('exact_per_seed.csv',exactrows)]:
        with(PUBLIC/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='validated',utc=now(),hypothesis='H3ag',decision='supported'if all(f['decision']=='supported'for f in families)else'falsified',family_results=families,rows=allrows,compilation_gate=gate,external_graph_gate=graph,compiled_verifier_validation=validation,query_worker=query,GPU_output_diagnostics=gpu_diags,exact_query_diagnostics=exact_diags,audited_GPU_states=len(gpurows)*4096*20,audited_exact_returned_states=len(exactrows)*4096*20,total_GPU_worker_seconds=total,GPU_worker_seconds_by_device=bygpu,GPU_phase_wall_seconds=latest-(budget['deadline_unix']-360),GPU_worker_receipts=workers,global_budget=budget,limits=p['limits'],input_sha256=HASHES,public_sha256={rel:sha(PUBLIC/rel)for rel in files},per_input_metrics_sha256=sha(PUBLIC/'per-input-metrics.npz'),audit_source_sha256=sha(__file__),GPU_audit_source_sha256=sha(SRC/'gpu_audit.py'),audit_wall_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);print(json.dumps({k:summary[k]for k in ['decision','total_GPU_worker_seconds','GPU_worker_seconds_by_device','GPU_phase_wall_seconds']}),flush=True)
if __name__=='__main__':main()
