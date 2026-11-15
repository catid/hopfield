"""Bounded synthetic-only exhaustive parity and work accounting; no cohort data."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import signal
import time
from compiler import Budget,compile_tree,ResourceCap
from unpruned_compiler import compile_tree as unpruned
from verify_tree import weighted_certificate


def paths(proof):
    if not proof['feasible']:return None
    result={}
    for i,cid in enumerate(proof['candidates']):
        key=proof['root_state'];used=[]
        while proof['states'][key]['reason']!='singleton_leaf':
            node=proof['states'][key];used.append(node['chosen_split']['board_index']);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
        result[str(cid)]=used
    return result


def inputs():
    for n in [2,3,4]:
        partitions=list(range(1,1<<(n-1)))
        for subset in range(1<<len(partitions)):
            signatures=[partitions[j] for j in range(len(partitions)) if subset&(1<<j)]
            for weights in itertools.product([1,2],repeat=n):
                for reverse in [False,True]:
                    ordered=list(reversed(signatures)) if reverse else signatures
                    for depth in [1,2,3,4]:yield dict(family='exhaustive_n2_to_n4',weights=list(weights),signatures=[0]+ordered+[(1<<n)-1],depth=depth)
    for n in [6,8,10]:
        for family,signatures in [('full_partitions',list(range(1<<n))),('singletons',[0]+[1<<i for i in range(n)]+[(1<<n)-1])]:
            for weights in [[1]*n,[1<<i for i in range(n)]]:
                for depth in [4,5,6,8]:yield dict(family=family,weights=weights,signatures=signatures,depth=depth)


def main(folder):
    folder.mkdir(parents=True,exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second synthetic assessment cap')));signal.alarm(120);budget=Budget(10000000,50000000);start=time.perf_counter();checks=0;rows=[];total=dict(reference_states=0,reference_splits=0,pruned_tree_states=0,pruned_bound_states=0,pruned_partition_attempts=0,pruned_merge_operations=0,reference_CPU_seconds=0.,pruned_CPU_seconds=0.,strict_prunes=0,tied_prunes=0);ledger=folder/'synthetic-parity.jsonl.gz'
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    with ledger.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as stream:
        for case,fixture in enumerate(inputs()):
            n=len(fixture['weights']);catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])];before_states=budget.states;before_splits=budget.splits;cpu=time.process_time();reference=unpruned(list(range(n)),fixture['weights'],catalog,fixture['depth'],budget);ref_cpu=time.process_time()-cpu;ns=budget.states-before_states;nt=budget.splits-before_splits;before_states=budget.states;before_splits=budget.splits;cpu=time.process_time();proof=compile_tree(list(range(n)),fixture['weights'],catalog,fixture['depth'],budget);new_cpu=time.process_time()-cpu;new_ns=budget.states-before_states;new_nt=budget.splits-before_splits;check('all_feasibility_cost_and_lex_paths',proof['feasible']==reference['feasible'] and proof['optimal_cost']==reference['optimal_cost'] and paths(proof)==paths(reference));verified_ns,verified_nt,_=weighted_certificate(proof,catalog,check,'synthetic_');check('all_bound_work_charged',new_ns==verified_ns and new_nt==verified_nt);strict=tied=0
            for node in proof['states'].values():
                for attempt in node['attempts']:
                    if attempt['result']=='huffman_bound_dominated':
                        b=attempt['bound_evaluation'];strict+=int(b['lower_bound'])>int(b['incumbent_cost']);tied+=int(b['lower_bound'])==int(b['incumbent_cost'])
            bounds=len(proof['huffman_bounds']);merges=sum(len(item['merges']) for item in proof['huffman_bounds'].values());row=dict(case=case,family=fixture['family'],n=n,depth=fixture['depth'],reference_states=ns,reference_splits=nt,pruned_tree_states=len(proof['states']),pruned_bound_states=bounds,pruned_partition_attempts=new_nt-merges,pruned_merge_operations=merges,reference_CPU_seconds=ref_cpu,pruned_CPU_seconds=new_cpu,strict_prunes=strict,tied_prunes=tied,all_paths_identical=True);rows.append(row)
            for key in total:total[key]+=row[key]
            stream.write((json.dumps(dict(case=case,input=fixture,reference=reference,pruned=proof,counters=row),sort_keys=True,separators=(',',':'))+'\n').encode())
    signal.alarm(0);summary=dict(source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['compiler.py','unpruned_compiler.py','verify_tree.py','assess_pruning.py','BOUND_LEMMA.md']},status='passed_synthetic_assessment',cohort_runs=0,actual_oracle_calls=0,synthetic_cases=len(rows),exhaustive_small_cases=sum(r['family']=='exhaustive_n2_to_n4' for r in rows),checks=checks,all_feasibility_costs_and_lex_paths_identical=True,all_bound_merge_and_tie_certificates_verified=True,aggregate_memo_states=budget.states,aggregate_work_steps=budget.splits,**total,wall_seconds=time.perf_counter()-start,proof_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),proof_ledger_bytes=ledger.stat().st_size,larger_fixture_rows=[r for r in rows if r['family']!='exhaustive_n2_to_n4']);(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='larger_fixture_rows'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);main(parser.parse_args().folder)
