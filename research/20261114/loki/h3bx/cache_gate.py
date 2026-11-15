"""Independent requested-depth, canonical-reuse and compact-cache proof checks."""
import hashlib
import numpy as np
from common import save,read,sha,now
from ledger import canonical
from verify_tree import weighted_certificate
import policy


def run(root,budget):
    root.mkdir(parents=True,exist_ok=False);old=policy.OUT;before_states=budget.states;before_steps=budget.splits;rows=[];checks=[0]
    def check(name,value):assert value,name;checks[0]+=1
    try:
        for name,scores in [('four',['1','2','3','4']),('seven',[str(1<<i) for i in range(7)]),('large_integer_six',['1']+[str((1<<1100)+1)]*5)]:
            n=len(scores);folder=root/name;folder.mkdir();policy.OUT=folder;pred=np.array([[bool(sig&(1<<c)) for sig in range(1<<n)] for c in range(n)]);compiler=policy.OnlineCompiler(pred,scores,'a'*64,budget,100000);proofs={};inputs=[]
            for depth in range(4,9):
                bs=budget.states;bt=budget.splits;row,cached=compiler.get(list(range(n)),depth);group=row['group'];directory=folder/f'group{group}';proof=read(directory/'proof.json');catalog=read(directory/'catalog.json')['signatures'];ns,nt,_=weighted_certificate(proof,catalog,check,'cache_');check('complete_logical_requested_proof',proof['depth_limit']==depth and row['depth']==depth and row['proof_sha256']==sha(directory/'proof.json') and cached['depth_limit']==depth and cached['optimal_cost']==proof['optimal_cost'] and cached['leaf_depths']==proof['leaf_depths'] and 'huffman_bounds' not in cached and cached['certificate_payload_in_memory'] is False and len(cached['states'])<=2*n-1 and all('attempts' not in node for node in cached['states'].values()));reuse=row['canonical_computation_reuse']
                if reuse is None:check('charged_original_computation',row['memo_states']==ns==budget.states-bs and row['work_steps']==nt==budget.splits-bt)
                else:
                    source=compiler.rows[reuse['group']];original=proofs[reuse['group']];check('source_hash_and_complete_body_reuse',reuse==dict(group=source['group'],source_requested_depth=source['depth'],effective_root_depth=min(depth,n-1),proof_sha256=source['proof_sha256'],group_result_sha256=sha(folder/f'group{source["group"]}/result.json')) and source['group']<group and source['canonical_computation_reuse'] is None and {k:v for k,v in original.items() if k!='depth_limit'}=={k:v for k,v in proof.items() if k!='depth_limit'} and row['memo_states']==row['work_steps']==budget.states-bs==budget.splits-bt==0)
                for i,cid in enumerate(range(n)):
                    key=cached['root_state'];used=[]
                    while cached['states'][key]['reason']!='singleton_leaf':
                        node=cached['states'][key];index=node['chosen_split']['board_index'];check('cached_query_matches_complete_proof',node['chosen_split']==proof['states'][key]['chosen_split'] and index not in used);used.append(index);key=node['positive_state'] if pred[cid,index] else node['negative_state']
                    check('cached_candidate_leaf',cached['states'][key]['mask']==1<<i and len(used)==proof['leaf_depths'][str(cid)]<=depth)
                bs=budget.states;bt=budget.splits;again=compiler.get(list(range(n)),depth);check('identical_requested_hit_zero_work',again[0]==row and again[1]==cached and budget.states==bs and budget.splits==bt);proofs[group]=proof;inputs.append(read(directory/'compiler-input.json'))
            check('requested_depths_distinct',len({hashlib.sha256(canonical(item)).hexdigest() for item in inputs})==5 and [item['depth'] for item in inputs]==list(range(4,9)) and not pred.flags.writeable);rows.append(dict(name=name,scores=scores,groups=compiler.rows,canonical_reuse_hits=sum(row['canonical_computation_reuse'] is not None for row in compiler.rows),all_requested_logical_proofs_verified=True))
    finally:policy.OUT=old
    result=dict(utc=now(),fixtures=rows,checks=checks[0],all_complete_proofs_and_source_reuse_hashes_verified=True,all_compact_paths_identical=True,all_requested_depth_receipts_distinct=True,memo_states_added=budget.states-before_states,work_steps_added=budget.splits-before_steps,global_memo_states=budget.states,global_work_steps=budget.splits,actual_oracle_calls=0);save(root/'summary.json',result);return result
