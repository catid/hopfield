"""H2br CPU-only policy boundary, complete scatter and malformed-input gate."""
from itertools import product
import json
import numpy as np
import study as S
import endpoint_dispatch_policy as P


def fails(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('Malformed dispatcher state accepted')


def fields(indices,lp=False):
    values=np.asarray(indices)
    return {'prediction':np.array([[(i%2)==0,lp] for i in values],bool).reshape(-1,1,2),
            'iterations':values.astype(np.int64)+(100 if lp else 0),
            'stopped':np.ones(len(values),bool),'finite':np.ones(len(values),bool),
            'own_margin':values.astype(np.float64)+(0.5 if lp else 0.),
            'native_cap':np.zeros(len(values),bool),'block_cap':np.zeros(len(values),bool)}


def main():
    cases=0
    for n in range(1,6):
        for choices in product(range(3),repeat=n):
            #0directLP,1acceptedBP,2geometry-rescuedBP. This enumerates all disjoint final-source possibilities.
            bi=np.array([i for i,c in enumerate(choices) if c],np.int64);valid=np.array([choices[i]==1 for i in bi],bool)
            li=P.lp_indices(n,bi,valid);expected=np.array([i for i,c in enumerate(choices) if c!=1],np.int64);assert np.array_equal(li,expected)
            out=P.assemble(n,fields(bi),fields(li,True),bi,li)
            for i,choice in enumerate(choices):
                source=fields([i],choice!=1)
                for name in P.FIELDS:assert np.array_equal(out[name][i],source[name][0])
                assert out['chosen_model'][i]==(choice!=1)
            cases+=1
    tokens=[];expected=[]
    for distance in [1,181,182,183,358]:
        goal=(0,1) if distance==1 else (distance-179,179)
        for reverse in [False,True]:
            t=np.ones((180,180),np.uint8);t[0,0]=3 if reverse else 2;t[goal]=2 if reverse else 3;tokens.append(t);expected.append(distance)
    batch=np.stack(tokens);distance,high,bi,low=P.partition(batch)
    assert distance.tolist()==expected and high.tolist()==[d>=182 for d in expected]
    assert np.array_equal(bi,np.flatnonzero(high)) and np.array_equal(low,np.flatnonzero(~high))
    permutation=np.arange(len(batch))[::-1];assert np.array_equal(P.partition(batch[permutation])[0],distance[permutation])
    bad=batch.copy();bad[0,0,0]=1;fails(lambda:P.partition(bad))
    bad=batch.copy();bad[0,1,1]=2;fails(lambda:P.partition(bad))
    bad=batch.copy();bad[0,1,1]=4;fails(lambda:P.partition(bad))
    fails(lambda:P.partition(batch[:0]));fails(lambda:P.partition(batch[:,:179]));fails(lambda:P.lp_indices(3,np.array([1,1]),np.ones(2,bool)))
    fails(lambda:P.lp_indices(3,np.array([0]),np.ones(2,bool)));fails(lambda:P.assemble(3,fields([0]),fields([1]),np.array([0]),np.array([1])))
    out={'status':'passed','synthetic_only':True,'complete_source_partition_scatter_cases':cases,'token_boundary_orientation_cases':len(tokens),'invalid_cases':8,'fresh_inputs_drawn':0,'model_outcomes_read':False,'source_sha256':{name:S.sha(S.HERE/name) for name in ['endpoint_dispatch_policy.py','preflight_endpoint_dispatch_policy.py']}}
    S.save(S.OUT/'bp_first_endpoint_dispatch/policy-synthetic-gate.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
