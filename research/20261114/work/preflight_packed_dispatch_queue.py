"""H2bs synthetic-only stable packing, true-cardinality and identity gates."""
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import time
import numpy as np
import packed_dispatch_queue as Q


def main():
    started=time.perf_counter();checked=0
    def check(indices,mask):
        nonlocal checked
        batches,slots=Q.stable_batches(indices,mask)
        expected=indices if mask is None else np.concatenate([indices[mask],indices[~mask]])
        actual=np.concatenate([b['indices'] for b in batches]);assert np.array_equal(actual,expected)
        assert len(actual)==len(np.unique(actual))==len(indices)
        assert slots==(len(indices)+31)//32+1
        assert all(1<=len(b['indices'])<=32 for b in batches)
        for batch in batches:
            ix=batch['indices'];assert np.all(np.diff(ix)>0)
            if mask is not None:assert np.all(mask[np.searchsorted(indices,ix)]==(batch['group']=='high'))
        final={};seen=np.zeros(len(indices),bool)
        for batch in batches:
            ix=batch['indices'];Q.scatter(final,seen,indices,ix,{'prediction':ix[:,None].copy(),'finite':np.ones(len(ix),bool)})
        assert seen.all() and np.array_equal(final['prediction'][:,0],indices)
        checked+=1
    for n in range(1,11):
        indices=np.arange(n,dtype=np.int64)
        for flags in itertools.product([False,True],repeat=n):check(indices,np.array(flags,bool))
    for n in [1,2,31,32,33,63,64,65,127,128,129,4095,4096]:
        indices=np.arange(n,dtype=np.int64);check(indices,None)
        for count in sorted({x for x in [0,1,2,31,32,33,63,64,65,n//2,n-1,n] if x<=n}):check(indices,np.arange(n)<count)
    indices=np.arange(4096,dtype=np.int64)
    for count in [0,1,31,32,33,2047,2048,2049,4095,4096]:
        batches,slots=Q.stable_batches(indices,np.arange(4096)<count)
        assert slots==129 and 3*slots==387 and len(batches) in [128,129]
        assert 3*sum(len(b['indices']) for b in batches)==12288
    tokens=np.ones((65,180,180),np.uint8);tokens[:,0,0]=2
    distance_values=np.resize(np.array([181,182,183]),65)
    for i,distance in enumerate(distance_values):tokens[i,179,distance-179]=3
    shards={0:tokens};indices=np.arange(65,dtype=np.int64)
    for mode in ['original_order','packed']:
        queue=Q.make_queue(shards,indices,mode)
        if mode=='packed':assert np.array_equal(queue['distances'],distance_values) and np.array_equal(queue['high'],distance_values>=182)
        for batch in queue['batches']:
            actual=Q.materialize(shards,batch['indices']);assert actual.shape==(len(batch['indices']),180,180) and np.array_equal(actual,tokens[batch['indices']])
    assert list(inspect.signature(Q.make_queue).parameters)==['tokens_by_shard','indices','mode']
    rejected=0
    def reject(callback):
        nonlocal rejected
        try:callback()
        except (ValueError,IndexError):rejected+=1
        else:raise AssertionError('Invalid identity or shape accepted')
    for bad in [[],[1,1],[1,0],[-1],[4096]]:reject(lambda bad=bad:Q.identities(np.array(bad,dtype=np.int64)))
    reject(lambda:Q.stable_batches(indices,np.zeros(65,np.int64)))
    reject(lambda:Q.stable_batches(indices,np.zeros(64,bool)))
    reject(lambda:Q.materialize(shards,np.arange(33,dtype=np.int64)))
    reject(lambda:Q.materialize(shards,np.array([0,0],np.int64)))
    reject(lambda:Q.scatter({},np.zeros(65,bool),indices,np.array([0,0]),{'value':np.ones(2)}))
    reject(lambda:Q.scatter({},np.zeros(65,bool),indices,np.array([66]),{'value':np.ones(1)}))
    seen=np.zeros(65,bool);seen[0]=True
    reject(lambda:Q.scatter({},seen,indices,np.array([0]),{'value':np.ones(1)}))
    bad=tokens.copy();bad[0,0,0]=1
    reject(lambda:Q.make_queue({0:bad},indices,'packed'))
    dest=Path(__file__).resolve().parents[3]/'artifacts/research/20261114/work/bp_first_packed_dispatch'
    dest.mkdir(parents=True,exist_ok=True);result={'status':'passed','synthetic_partition_and_scatter_cases':checked,'invalid_cases_rejected':rejected,'full_cohort_slots_per_model':129,'slots_all_models':387,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str(p.relative_to(dest.parents[4])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(Q.__file__)]}}
    (dest/'queue-synthetic-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
