"""Diagnose every fixed H3y case only after the global source seal."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import csv
import datetime as dt
import shutil
import socket
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,save,read

PUBLIC=ROOT/'results/research/loki/h3y'
def main(expected_seal):
    assert socket.gethostname()=='loki';start=time.perf_counter();p=read(SRC/'protocol.json');seal=read(OUT/'source-seals.json')
    assert sha(OUT/'source-seals.json')==expected_seal and seal['status']=='all18sources_sealed_before_any_target_diagnosis'
    assert seal['protocol_sha256']==sha(SRC/'protocol.json')and len(seal['source_result_sha256'])==18
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    generation=read(OUT/'generation.json');assert generation['status']=='complete'
    source_results={}
    for case in p['cases']:
        cid=case['id'];path=OUT/f'cases/{cid}/result.json';assert sha(path)==seal['source_result_sha256'][cid];r=read(path)
        assert r['protocol_sha256']==sha(SRC/'protocol.json')and r['no_target_file_access']
        assert dt.datetime.fromisoformat(r['completed_utc'])<=dt.datetime.fromisoformat(seal['utc'])
        if r['status']=='source_certified':
            for name,key in [('certificate.npz','certificate_sha256'),('verification.json','verification_sha256'),('checked-subsets.json','checked_subsets_sha256')]:assert sha(path.parent/name)==r[key]
            assert read(path.parent/'verification.json')['status']=='passed'
        source_results[cid]=r
    # All eighteen source hashes and certificates are checked before target truth.
    rows=[];diagnostics={}
    for case in p['cases']:
        cid=case['id'];r=source_results[cid];family=[tuple(o)for o in p['families'][case['family_id']]];n=case['target_n'];i,j=np.triu_indices(n*n,1);dr=j//n-i//n;dc=j%n-i%n
        truth=(dr==0)|(dc==0)|(dr==np.abs(dc))
        for u,v in family:truth|=((dr==u)&(np.abs(dc)==v))|((dr==v)&(np.abs(dc)==u))
        minimum_diagnostics=[]
        if r['status']=='source_certified':
            for selected in r['minima']:
                offsets={tuple(o)for k in selected for o in r['candidate_orbits'][k]}
                predicted=np.array([any(a*int(x)+b*int(y)==0 for a,b in r['inferred_lines'])or(int(x),int(y))in offsets for x,y in zip(dr,dc)])
                fp=np.flatnonzero(predicted&~truth);fn=np.flatnonzero(~predicted&truth)
                minimum_diagnostics.append(dict(orbit_indices=selected,generators=[r['candidate_generators'][k]for k in selected],
                    target_pairs=int(predicted.sum()),false_pairs=len(fp),missed_pairs=len(fn),exact_graph=len(fp)==len(fn)==0,
                    first_false_edge=[int(i[fp[0]]),int(j[fp[0]])]if len(fp)else None,
                    first_missed_edge=[int(i[fn[0]]),int(j[fn[0]])]if len(fn)else None))
        if r['status'].startswith('incomplete'):outcome='incomplete'
        elif r['status']=='empty_source':outcome='empty_source'
        elif r['status']!='source_certified':outcome=r['status']
        elif len(minimum_diagnostics)!=1:outcome='ambiguous_minimum'
        elif not minimum_diagnostics[0]['exact_graph']:outcome='unique_minimum_wrong_graph'
        else:outcome='unique_exact_transfer'
        diagnostics[cid]=dict(source_result=r,target_true_pairs=int(truth.sum()),all_minima=minimum_diagnostics,selected_after_target_comparison=False)
        rows.append(dict(id=cid,family_id=case['family_id'],family=str(p['families'][case['family_id']]),source_n=case['n'],target_n=n,
            source_patterns=r['source_patterns'],source_status=r['status'],candidate_orbits=len(r.get('candidate_generators',[])),
            minimum_cardinality=r.get('minimum_cardinality'),minimum_count=len(r.get('minima',[])),checked_subsets=r.get('checked_subset_count',0),
            exact_minimum_count=sum(m['exact_graph']for m in minimum_diagnostics),false_pairs_best=min((m['false_pairs']for m in minimum_diagnostics),default=None),
            false_pairs_worst=max((m['false_pairs']for m in minimum_diagnostics),default=None),missed_pairs_best=min((m['missed_pairs']for m in minimum_diagnostics),default=None),
            missed_pairs_worst=max((m['missed_pairs']for m in minimum_diagnostics),default=None),outcome=outcome,source_cpu_seconds=r['cpu_seconds'],source_wall_seconds=r['wall_seconds']))
    counts={label:sum(row['outcome']==label for row in rows)for label in sorted({r['outcome']for r in rows})}
    failed=sum(r['outcome']not in['unique_exact_transfer','incomplete']for r in rows)
    decision='falsified'if failed else'inconclusive'if counts.get('incomplete',0)else'supported'
    summary=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3y',decision=decision,
        primary=p['primary'],source_seal_sha256=expected_seal,protocol_sha256=sha(SRC/'protocol.json'),families=p['families'],source_sizes=p['source_sizes'],
        cases=18,outcome_counts=counts,rows=rows,case_diagnostics=diagnostics,generation=generation,source_seals=seal,
        source_cpu_seconds=sum(r['cpu_seconds']for r in source_results.values()),source_worker_wall_seconds=sum(r['wall_seconds']for r in source_results.values()),
        source_campaign_wall_seconds=seal['campaign_wall_seconds'],generation_cpu_seconds=generation['cpu_seconds'],generation_wall_seconds=generation['wall_seconds'],
        reused_universe_cache_cases=sum(r.get('cache_reused',False)for r in source_results.values()),unique_universe_cache_keys=len({r['cache_key']for r in source_results.values()if'cache_key'in r}),
        total_checked_subsets=sum(r.get('checked_subset_count',0)for r in source_results.values()),target_diagnostic_seconds=time.perf_counter()-start,
        new_gpu_search_jobs=0,new_optimizer_fits=0,target_tie_selections=0,limits=p['limits'],diagnostic_source_sha256=sha(__file__))
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for directory in ['sources','cases','cache']:
        shutil.copytree(OUT/directory,PUBLIC/directory,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.lock'))
    for name in ['generation.json','source-seals.json']:shutil.copy2(OUT/name,PUBLIC/name)
    with(PUBLIC/'per_case.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary['export_sha256']={str(path.relative_to(PUBLIC)):sha(path)for path in sorted(PUBLIC.rglob('*'))if path.is_file()and path.name not in['summary.json','README.md']and path.suffix not in['.png','.svg']and path.name!='plot-provenance.json'}
    save(PUBLIC/'summary.json',summary)
    print('DECISION',decision,'COUNTS',counts,'SOURCE_CPU',summary['source_cpu_seconds'],'CACHE_REUSES',summary['reused_universe_cache_cases'],flush=True)
    for row in rows:print(row,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source-seal-sha256',required=True);args=a.parse_args();main(args.source_seal_sha256)
