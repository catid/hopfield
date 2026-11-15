"""Independently audit serialized normalized bound/reuse fixture artifacts."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from audit_trees import TreeAudit
from verify_tree import weighted_certificate


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def paths(proof):
    if not proof['feasible']:return None
    result={}
    for i,cid in enumerate(proof['candidates']):
        key=proof['root_state'];used=[]
        while proof['states'][key]['reason']!='singleton_leaf':
            node=proof['states'][key];used.append(node['chosen_split']['board_index']);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
        result[str(cid)]=used
    return result


def verify(folder,source,pruning_name="pruning",cache_name="cache",read_json=None,hash_file=None):
    folder=Path(folder);source=Path(source);read=read_json if read_json is not None else lambda path:json.loads(path.read_text());sha=hash_file if hash_file is not None else lambda path:hashlib.sha256(path.read_bytes()).hexdigest();canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();checks=0
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    summary=read(folder/pruning_name/'summary.json');reference=load(source.parent/'h3bu/verify_tree.py','unpruned_normalized_independent');count=0;states=steps=0
    with gzip.open(folder/pruning_name/'synthetic-parity.jsonl.gz','rt') as f:
        for line in f:
            record=json.loads(line);fixture=record['input'];catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])];old=record['reference'];new=record['pruned'];ns,nt,_=reference.weighted_certificate(old,catalog,check,'reference_');new_ns,new_nt,_=weighted_certificate(new,catalog,check,'pruned_');check('serialized_parity_and_work',record['case']==count and new['optimal_cost']==old['optimal_cost'] and new['feasible']==old['feasible'] and paths(new)==paths(old) and ns==record['counters']['reference_states'] and nt==record['counters']['reference_splits'] and new_ns==record['counters']['pruned_tree_states']+record['counters']['pruned_bound_states'] and new_nt==record['counters']['pruned_partition_attempts']+record['counters']['pruned_merge_operations']);states+=ns+new_ns;steps+=nt+new_nt;count+=1
    check('all_pruning_fixtures_retained',count==summary['synthetic_cases']==17008 and states==summary['memo_states_added'] and steps==summary['work_steps_added'] and sha(folder/pruning_name/'synthetic-parity.jsonl.gz')==summary['proof_ledger_sha256'])
    cache=read(folder/cache_name/'summary.json');logical_states=logical_steps=reuse=0
    for fixture in cache['fixtures']:
        sub=folder/cache_name/fixture['name'];n=len(fixture['scores']);pred=np.array([[bool(sig&(1<<c)) for sig in range(1<<n)] for c in range(n)]);trees=TreeAudit(sub,source,fixture['groups'],pred,fixture['scores'],'a'*64,check,read,sha,canonical)
        for row in fixture['groups']:trees.get(row['candidates'],cache,row['depth'])
        ns,nt=trees.finish();states+=ns;steps+=nt;logical_states+=trees.logical_states;logical_steps+=trees.logical_steps;reuse+=trees.reuse_hits
    check('all_cache_charges',states-summary['memo_states_added']==cache['memo_states_added'] and steps-summary['work_steps_added']==cache['work_steps_added'])
    return dict(status='passed_independent_synthetic_artifact_audit',checks=checks,synthetic_cases=count,charged_memo_states=states,charged_work_steps=steps,cache_reuse_hits=reuse,cache_logical_states=logical_states,cache_logical_steps=logical_steps,all_ordering_pruning_bounds_and_ties_verified=True,no_unreferenced_bound_entries=True,all_reuse_source_hashes_full_proof_bodies_requested_depths_verified=True,cohort_runs=0,actual_oracle_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();print(json.dumps(verify(args.folder,Path(__file__).resolve().parent)))
