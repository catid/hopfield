"""Final CPU-only audits of Figure3a, failed-ablation reconstructions and long BPTT.

Recompute saved arrays and hash all evidence, including each complete BPTT log.
No fitting, development probing, or test inference is invoked.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'upstream'
EVIDENCE={}


def sha(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    value=h.hexdigest();EVIDENCE[str(path.relative_to(ROOT))]=value;return value


def read(path):
    path=Path(path);b=path.read_bytes();EVIDENCE[str(path.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return json.loads(b)


def close(a,b):assert np.isclose(a,b,rtol=1e-12,atol=1e-7),(a,b)


def stats(x):return dict(mean=statistics.mean(x),sample_sd=statistics.stdev(x),per_seed=x)


def arrays(rec):
    pp=rec['per_puzzle'];ids=np.asarray(pp['puzzle']);right=np.asarray(pp['solved'],bool)
    stop=np.asarray(pp['stopped'],bool);fl=np.asarray(pp['flops'],float)
    assert ids.shape==right.shape==stop.shape==fl.shape
    assert np.isfinite(fl).all() and (fl>=0).all()
    counts=dict(n=len(ids),solved=int(right.sum()),stopped=int(stop.sum()),wrong_stops=int((stop&~right).sum()),
                budget_failures=int((~stop).sum()),unstopped_but_right=int((~stop&right).sum()))
    assert all(rec['summary'][k]==v for k,v in counts.items())
    for key,value in dict(mean_flop=fl.mean(),median_flop=np.median(fl),p99_flop=np.quantile(fl,.99),
                          p999_flop=np.quantile(fl,.999),max_flop=fl.max()).items():close(value,rec['summary'][key])
    return ids,right,stop,fl,counts


def difficulty():
    base=ROOT/'artifacts/difficulty';figure=read(base/'figure3a.json');final=read(base/'finalization.json')
    assert final['status']=='complete' and final['exit_code']==0
    for path,h in final['outputs'].items():assert sha(ROOT/path)==h
    assert figure['subset']['full_all_rows']==[10000,20000] and figure['subset']['n']==10000
    assert figure['provenance']['dataset_sha256']==sha(UP/'data/full-test/full-all.npz')
    assert figure['provenance']['published_sha256']==sha(ROOT/figure['provenance']['published_source'])
    assert figure['provenance']['wrapper_sha256']==sha(ROOT/'scripts/export_sudoku_difficulty.py')
    data=np.load(UP/'data/full-test/full-all.npz');ratings=data['ratings'][10000:20000]
    bins=np.asarray([0,1,2,4,8,16,32,64,10**9]);counts=[int(((ratings>=a)&(ratings<b)).sum()) for a,b in zip(bins[:-1],bins[1:])]
    assert counts==figure['bin_counts'] and sum(counts)==10000;close(ratings.mean(),figure['subset']['rating_mean'])
    counters=read(ROOT/'artifacts/sudoku_operation_counts.json');over=counters['sudoku_general_overheads_s0']
    init=over['init_per_puzzle']['total'];poll=over['check_per_puzzle']['total']
    full=[];subset=[];work=[];equality=[]
    for seed in range(3):
        p=base/f'full/provenance-s{seed}.json';prov=read(p)
        assert prov['status']=='complete' and prov['seed']==seed and prov['test_size']=='full'
        assert prov['stop']=='zero' and prov['test_rows']==[10000,422786] and prov['figure_rows']==[10000,20000]
        assert prov['source_commit']=='62bbe830c7d6f87f2ba18f30a28039258b986454'
        for path,h in prov['source_sha256'].items():assert sha(UP/path)==h
        assert prov['wrapper_sha256']==sha(ROOT/'scripts/run_sudoku_difficulty.py')
        model=base/f'full/full-aug-s{seed}/model.pt'
        assert sha(model)==prov['model_sha256']==sha(ROOT/prov['fresh_model_source'])
        rec=read(base/f'full/test/cram0-hb-s{seed}.json');sig=rec['signature']
        expected=dict(model='full-aug',split=seed,seed=seed,model_sha256=prov['model_sha256'],data='full-all',lo=10000,
                      n=412786,test='zero',scale='hbar',C=4,cap=4e11,check=4096,tol=1e-3,rel=[.4,.31,.45])
        assert all(sig[k]==v for k,v in expected.items()) and 'tau' not in sig
        close(sig['theta'],.45*sig['hbar']);assert sig['anneal']==[.4*sig['hbar'],.31*sig['hbar'],524288]
        ids,right,stop,raw,total=arrays(rec);assert np.array_equal(ids,np.arange(10000,422786))
        assert rec['summary']==prov['test_summary']
        dev=read(base/f'full/dev/cram0-hb-s{seed}.json');di,dr,ds,df,dc=arrays(dev)
        assert dev['signature']['data']=='dev10k' and np.array_equal(di,np.arange(10000))
        assert dev['summary']==prov['development_summary'] and dc['solved']>=9999 and dc['wrong_stops']==0
        for key in set(sig)-{'data','lo','n'}:assert sig[key]==dev['signature'][key],key
        steps=raw/(sig['C']*sig['fl_move']);assert np.allclose(steps,np.rint(steps),atol=1e-6,rtol=0);steps=np.rint(steps)
        max_steps=math.ceil(rec['per_puzzle']['T_cap']/4096)*4096;assert steps.max()<=max_steps
        move=counters[f'sudoku_general_move_D{sig["D"]}']['total']
        fc2=steps*sig['C']*move+init+np.ceil(steps/4096)*poll
        full.append(dict(seed=seed,**total,solved_percent=100*float(right.mean()),paper_fc2_mean=float(fc2.mean()),
                         raw_mean=float(raw.mean()),seconds=rec['per_puzzle']['seconds'],resumes=rec['resumes']))
        r,s,f=right[:10000],stop[:10000],fc2[:10000]
        sub=dict(seed=seed,n=10000,solved=int(r.sum()),stopped=int(s.sum()),wrong_stops=int((s&~r).sum()),
                 budget_failures=int((~s).sum()),solved_percent=100*float(r.mean()),paper_fc2_mean=float(f.mean()))
        fresh=figure['fresh']['CRAM'];close(sub['solved_percent'],fresh['solved_pct'][seed]);close(f.mean(),fresh['mean_flop'][seed])
        assert sub['wrong_stops']==fresh['wrong_stops'][seed] and sub['budget_failures']==fresh['budget_failures'][seed]
        assert figure['source_records'][seed]['sha256']==sha(base/f'full/test/cram0-hb-s{seed}.json')
        assert figure['source_records'][seed]['full_summary']==rec['summary']
        subset.append(sub);work.append((r,s,f))
        main=read(UP/f'experiments/runs/CRAM-S/full/test/cram0-hb-fit-s{seed}.json')
        equality.append(dict(seed=seed,arrays_equal_to_separately_run_fit_threshold={k:rec['per_puzzle'][k]==main['per_puzzle'][k] for k in ['puzzle','solved','stopped','flops']},
                             distinct_runtime_seconds=rec['per_puzzle']['seconds']!=main['per_puzzle']['seconds'],fitted_tau=main['signature']['tau']))
    fresh=figure['fresh']['CRAM']
    for index,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
        mask=(ratings>=lo)&(ratings<hi);pooled=np.concatenate([r[2][mask] for r in work]);b=fresh['bins'][index]
        for k,v in dict(flop_median=np.median(pooled),flop_p10=np.percentile(pooled,10),flop_p90=np.percentile(pooled,90),flop_mean=pooled.mean()).items():close(v,b[k])
        for seed,(r,_,_) in enumerate(work):close(100*r[mask].mean(),b['solved_pct'][seed])
    assert np.allclose(np.logspace(6,12,121),figure['budgets'],rtol=2e-15,atol=0)
    budgets=np.asarray(figure['budgets'])  # Preserve exported float boundaries across NumPy versions.
    for seed,(r,_,fl) in enumerate(work):
        for i,b in enumerate(budgets):close(100*(r&(fl<=b)).mean(),fresh['solved_within_budget_pct'][seed][i])
    ratio=fresh['bins'][-1]['flop_median']/fresh['bins'][0]['flop_median'];close(ratio,fresh['hardest_to_easiest_median_compute'])
    return dict(status='validated',full_rows_half_open=[10000,422786],full_per_seed=full,
                full_solved_percent=stats([r['solved_percent'] for r in full]),subset_rows_half_open=[10000,20000],
                subset_per_seed=subset,subset_solved_percent=stats([r['solved_percent'] for r in subset]),
                bin_counts=counts,budget_points_checked_per_seed=121,hardest_to_easiest_median_compute=ratio,
                separate_protocol_comparison=equality,
                limitation='Saved per-puzzle flags and counts are independently aggregated; final Sudoku boards were not retained. Identical observed arrays across zero and fitted stopping thresholds do not make the protocols interchangeable.')


def maze_ablations():
    base=ROOT/'artifacts/maze_hard_ablations';summary=read(base/'summary.json')
    assert summary['status']=='complete' and not summary['test_data_used']
    source=UP/'experiments/recipes/maze-hard-latent.json';successful=read(source)['recipe']
    order=np.random.default_rng(0).permutation(1000);fit=np.sort(order[100:]);dev=np.sort(order[:100]);rows=[]
    for variant,seed in [('token-source',0),('token-source',1),('token-source',2),('edge-gates',0)]:
        folder=base/variant;prov=read(folder/f'provenance-s{seed}.json');rec=read(folder/f'dev-s{seed}.json')
        assert prov['status']=='complete' and prov['variant']==variant and prov['seed']==seed==rec['seed']
        assert not prov['test_data_used'] and not prov['tuning'];assert np.array_equal(prov['fit_rows'],fit) and np.array_equal(prov['development_rows'],dev)
        change={'source':'token'} if variant=='token-source' else {'coupling':'gate'}
        assert prov['changed_fields']==change and rec['recipe']==prov['recipe']==dict(successful,**change)
        assert prov['recipe_source_sha256']==sha(source) and prov['wrapper_sha256']==sha(ROOT/'scripts/run_maze_hard_ablations.py')
        assert prov['train_csv_sha256']==sha(UP/'data/maze-hard/train.csv')
        for p,h in prov['source_sha256'].items():assert sha(UP/p)==h
        assert sha(folder/f'model-s{seed}.pt')==prov['model_sha256']
        curve=rec['curve'];assert curve[-1]['u']==4000 and all(a['u']<b['u'] for a,b in zip(curve,curve[1:]))
        assert all(math.isfinite(c['pl']) for c in curve)
        exact=np.asarray(rec['dev100_per_maze']['exact'],bool);stopped=np.asarray(rec['dev100_per_maze']['stopped'],bool)
        flop=np.asarray(rec['dev100_per_maze']['flop'],float)
        assert exact.shape==stopped.shape==flop.shape==(100,) and np.isfinite(flop).all()
        values=dict(n=100,solved=int(exact.sum()),wrong_stops=int((stopped&~exact).sum()),budget_failures=int((~stopped).sum()))
        assert all(rec['dev100'][k]==v for k,v in values.items()) and rec['dev100']==prov['result']
        reported=next(r for r in summary['groups'][variant]['per_seed'] if r['seed']==seed)
        for k,v in values.items():assert reported[k]==v
        for path,key in [(folder/f'dev-s{seed}.json','result_sha256'),(folder/f'provenance-s{seed}.json','provenance_sha256'),(folder/f'model-s{seed}.pt','model_sha256')]:assert sha(path)==reported[key]
        rows.append(dict(variant=variant,seed=seed,**values,fit_updates=4000,params=rec['params'],fit_seconds=rec['times']['total_s'],final_loss=curve[-1]['pl']))
    for host,gpu,seed,jobs in [('foureyes',3,0,[['token-source',0],['edge-gates',0]]),('ripper',2,1,[['token-source',1]]),('ripper',3,2,[['token-source',2]])]:
        q=read(base/f'queue-{host}-gpu{gpu}.json');assert q['status']=='complete' and q['jobs']==jobs
        assert q['marker_seed']==seed and [[r['variant'],r['seed']] for r in q['completed']]==jobs
        assert q['dependency_sha256']==sha(ROOT/f'artifacts/long_bptt/collected/seed{seed}/completion.json')
    for variant,group in summary['groups'].items():
        selected=[r for r in rows if r['variant']==variant]
        assert group['fresh_reconstruction']==dict(solved=sum(r['solved'] for r in selected),n=sum(r['n'] for r in selected))
    return dict(status='validated',rows=rows,counts_match_published=True,
                limitation='Single-field controlled reconstructions of successful recipe. Historical failed recipes/checkpoints/logs/seeds were not released, so matching zero counts are not exact historical-run reproduction. Saved development flags, not final boards, were retained.')


def long_bptt():
    from collect_external_bptt import collect,validate_completed,validate_timing_log
    collected=collect();assert collected['complete'] and len(collected['jobs'])==3
    def one(seed):
        folder=ROOT/f'artifacts/long_bptt/collected/seed{seed}'
        marker=read(folder/'completion.json');receipt=read(folder/'collection.json')
        for name,h in receipt['files_sha256'].items():assert sha(folder/name)==h,name
        fit,test,config=validate_completed(folder,marker,seed)
        timing=validate_timing_log(folder/'log.jsonl',fit);assert timing==receipt['timing']
        raw=np.asarray(test['flops']);assert raw.shape==(50000,) and np.isfinite(raw).all()
        for k,v in dict(mean_flop=raw.mean(),median_flop=np.median(raw),p99_flop=np.quantile(raw,.99),p999_flop=np.quantile(raw,.999),max_flop=raw.max()).items():close(v,test['test'][k])
        s=test['test'];assert s['n']==s['stopped']+s['budget_failures']
        assert s['solved']==s['stopped']-s['wrong_stops']+s['unstopped_but_right']
        chosen=min(fit['curve'],key=lambda r:(-r['solved']/r['n'],r['update']))
        assert chosen['update']==fit['selected']['update']==test['selected_update']
        return dict(seed=seed,status='validated',updates=fit['updates'],selected_update=test['selected_update'],
                    test=s,solved_percent=100*s['solved']/s['n'],full_log_rehashed_and_reparsed=True,
                    recorded_updates=timing['timing_log_updates'],resume_entries=timing['resume_entries'])
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,range(3)))
    return dict(status='validated',per_seed=rows,solved_percent=stats([r['solved_percent'] for r in rows]),
                limitation='50,000 test FLOP arrays and aggregate correctness accounting validated; evaluator did not retain per-puzzle correctness flags or final boards.')


def main():
    begin=time.monotonic();out=ROOT/'artifacts/final_controls_audit.json'
    result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
                difficulty=difficulty(),maze_hard_ablations=maze_ablations(),long_bptt=long_bptt())
    result.update(status='validated',elapsed_seconds=time.monotonic()-begin,evidence_sha256=EVIDENCE)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),status=result['status'],elapsed_seconds=result['elapsed_seconds'],
                         full=result['difficulty']['full_per_seed'],subset=result['difficulty']['subset_per_seed'],
                         long_bptt=result['long_bptt']['solved_percent'])),flush=True)


if __name__=='__main__':main()
