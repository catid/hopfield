"""Fixed synthetic end-to-end bounds-cache comparison; no cohort artifacts."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np


def fixtures():
    rows=[]
    for n in [10,12]:
        for kind,weights in [('uniform',[1]*n),('geometric',[1<<i for i in range(n)])]:rows.append(dict(name=f'full{n}_{kind}',n=n,weights=weights,signatures=list(range(1<<n)),depth=8))
    for count in [32,128]:
        n=16;random=np.random.default_rng(202611150000+count).choice(np.arange(1,(1<<n)-1),count,replace=False).tolist();signatures=sorted(set([0,(1<<n)-1]+[1<<i for i in range(n)]+[sum(1<<c for c in range(n) if c&(1<<bit)) for bit in range(4)]+random))
        for kind,weights in [('uniform',[1]*n),('geometric',[1<<i for i in range(n)])]:rows.append(dict(name=f'sparse16_{count}_{kind}',n=n,weights=weights,signatures=signatures,depth=8))
    return rows


def paths(proof):
    if not proof['feasible']:return None
    result={}
    for i,cid in enumerate(proof['candidates']):
        key=proof['root_state'];used=[]
        while proof['states'][key]['reason']!='singleton_leaf':
            node=proof['states'][key];used.append(node['chosen_split']['board_index']);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
        result[str(cid)]=used
    return result


def worker(variant,out):
    out.mkdir(parents=True,exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second synthetic assessment guard')));signal.alarm(120);module=importlib.import_module(dict(memoized='memoized_compiler',trivial_inline='compiler',all_inline='all_inline_compiler')[variant]);budget=module.Budget(2000000,50000000);rows=[];start=time.perf_counter();cpu=time.process_time()
    for fixture in fixtures():
        bs=budget.states;bt=budget.splits;wall=time.perf_counter();ct=time.process_time();catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])];proof=module.compile_tree(list(range(fixture['n'])),fixture['weights'],catalog,fixture['depth'],budget);compile_cpu=time.process_time()-ct;raw=json.dumps(proof,indent=2).encode()+b'\n';compressed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(compressed)==raw;file=out/(fixture['name']+'.json.gz');file.write_bytes(compressed);rows.append(dict(name=fixture['name'],n=fixture['n'],feasible=proof['feasible'],cost=proof['optimal_cost'],paths=paths(proof),states=budget.states-bs,work_steps=budget.splits-bt,compile_CPU_seconds=compile_cpu,total_CPU_seconds=time.process_time()-ct,wall_seconds=time.perf_counter()-wall,raw_bytes=len(raw),gzip_bytes=len(compressed),raw_sha256=hashlib.sha256(raw).hexdigest(),gzip_sha256=hashlib.sha256(compressed).hexdigest()));del proof,raw,compressed
    result=dict(variant=variant,fixtures=fixtures(),measurements=rows,total_states=budget.states,total_work_steps=budget.splits,total_CPU_seconds=time.process_time()-cpu,total_wall_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cohort_runs=0,actual_oracle_calls=0);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(variant,result['total_states'],result['total_work_steps'],result['total_CPU_seconds'],flush=True)


def main(out):
    out.mkdir(parents=True,exist_ok=False);variants=[]
    for variant in ['memoized','trivial_inline','all_inline']:
        result=subprocess.run([sys.executable,__file__,'--worker',variant,'--out',str(out/variant)],capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr;variants.append(json.loads((out/variant/'summary.json').read_text()))
    for cases in zip(*(variant['measurements'] for variant in variants)):
        assert all(all(case[key]==cases[0][key] for key in ['name','n','feasible','cost','paths']) for case in cases)
    summary=dict(status='passed_fixed_synthetic_end_to_end_assessment',all_cost_feasibility_paths_identical=True,cohort_runs=0,actual_oracle_calls=0,variants=[{key:value for key,value in row.items() if key not in ['fixtures','measurements']} for row in variants],source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['compiler.py','memoized_compiler.py','all_inline_compiler.py','assess_cache_tradeoff.py']});(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();worker(args.worker,args.out) if args.worker else main(args.out)
