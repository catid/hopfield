"""Separate-process synthetic RSS/canonical-reuse assessment, with byte parity."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from compiler import Budget
import policy


def paths(proof):
    if not proof['feasible']:return None
    result={}
    for i,cid in enumerate(proof['candidates']):
        key=proof['root_state'];used=[]
        while proof['states'][key]['reason']!='singleton_leaf':
            node=proof['states'][key];used.append(node['chosen_split']['board_index']);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
        result[str(cid)]=used
    return result


def worker(variant,suite,folder):
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second synthetic wrapper cap')));signal.alarm(120);folder.mkdir(parents=True,exist_ok=False);n=12 if suite=='RAM' else 10;pred=np.array([[bool(sig&(1<<c)) for sig in range(1<<n)] for c in range(n)]);scores=['1']*n if suite=='RAM' else [str(1<<c) for c in range(n)];budget=Budget();start=time.perf_counter();cpu=time.process_time()
    if variant=='unpruned_full':
        source=Path(__file__).resolve().parents[1]/'h3bu/policy.py';spec=importlib.util.spec_from_file_location('baseline_wrapper',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);from unpruned_compiler import compile_tree;module.compile_tree=compile_tree;module.OUT=folder;module.SRC=source.parent;compiler=module.OnlineCompiler(pred,scores,'a'*64,budget,100000)
    else:
        policy.OUT=folder;compiler=policy.OnlineCompiler(pred,scores,'a'*64,budget,100000,compact_cache=variant!='pruned_full',reuse_effective=variant=='pruned_compact_reuse')
    if suite=='RAM':requests=[([c for c in range(n) if c not in omitted],8) for omitted in list(itertools.combinations(range(n),2))[:12]]
    else:requests=[(sorted((offset+i)%n for i in range(count)),depth) for count in range(2,9) for offset in range(3) for depth in range(4,9)]
    proof_hashes={};chosen={};proof_bytes=0
    for candidates,depth in requests:
        row,proof=compiler.get(candidates,depth);key=json.dumps([candidates,depth],separators=(',',':'));path=folder/f'group{row["group"]}/proof.json';data=path.read_bytes();proof_hashes[key]=hashlib.sha256(data).hexdigest();chosen[key]=dict(feasible=proof['feasible'],cost=proof['optimal_cost'],paths=paths(proof));proof_bytes+=len(data);before=(budget.states,budget.splits);assert compiler.get(candidates,depth)[0]==row and before==(budget.states,budget.splits)
    result=dict(variant=variant,suite=suite,requests=len(requests),memo_states=budget.states,work_steps=budget.splits,CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,retained_tree_states=sum(len(item[1]['states']) for item in compiler.cache.values()),retained_bound_states=sum(len(item[1].get('huffman_bounds',{})) for item in compiler.cache.values()),canonical_reuse_hits=sum(row.get('canonical_computation_reuse') is not None for row in compiler.rows),raw_proof_bytes=proof_bytes,proof_sha256_by_requested_inputs=proof_hashes,chosen_paths_by_requested_inputs=chosen,zero_work_identical_cache_hits=True,actual_oracle_calls=0);(folder/'measurement.json').write_text(json.dumps(result,indent=2)+'\n');print({key:value for key,value in result.items() if key not in ['proof_sha256_by_requested_inputs','chosen_paths_by_requested_inputs']})


def main(folder):
    folder.mkdir(parents=True,exist_ok=False);receipts=[]
    for suite,variants in [('RAM',['unpruned_full','pruned_full','pruned_compact']),('reuse',['pruned_compact','pruned_compact_reuse'])]:
        rows=[]
        for variant in variants:
            out=folder/f'{suite}-{variant}';result=subprocess.run([sys.executable,__file__,'--worker',variant,'--suite',suite,'--folder',str(out)],capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr;row=json.loads((out/'measurement.json').read_text());rows.append(row);receipts.append({key:value for key,value in row.items() if key not in ['proof_sha256_by_requested_inputs','chosen_paths_by_requested_inputs']})
        assert all(row['chosen_paths_by_requested_inputs']==rows[0]['chosen_paths_by_requested_inputs'] for row in rows)
        pruned=[row for row in rows if row['variant']!='unpruned_full'];assert all(row['proof_sha256_by_requested_inputs']==pruned[0]['proof_sha256_by_requested_inputs'] for row in pruned)
    summary=dict(status='passed_synthetic_wrapper_assessment',cohort_runs=0,actual_oracle_calls=0,all_chosen_paths_and_costs_identical=True,pruned_full_compact_and_reused_logical_proofs_byte_identical=True,measurements=receipts,source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['compiler.py','unpruned_compiler.py','policy.py','weights.py','ledger.py','common.py','assess_wrapper.py']});(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--suite');parser.add_argument('--folder',type=Path,required=True);args=parser.parse_args();worker(args.worker,args.suite,args.folder) if args.worker else main(args.folder)
