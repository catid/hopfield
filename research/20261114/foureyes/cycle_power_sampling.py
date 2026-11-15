"""Uniform proper colorings of a cyclic path power, by exact conditioning."""
from functools import lru_cache
import itertools
import numpy as np
FAMILIES=['power2','power3']


def graph_order(seed,n=81):return np.random.default_rng(seed).permutation(n).astype(np.int64)


def graph_edges(order,family,q=9):
    n=len(order);k=int(family[-1]);assert n>2*k and np.array_equal(np.sort(order),np.arange(n))
    adjacency=np.zeros((n,n),bool)
    for i in range(n):
        for step in range(1,k+1):a,b=order[i],order[(i+step)%n];adjacency[a,b]=adjacency[b,a]=True
    return adjacency[np.triu_indices(n,1)]


@lru_cache(None)
def allowed_table(q,k):
    assert q>k;table=np.full((q**k,q-k),-1,np.int64)
    for previous in itertools.permutations(range(q),k):
        state=0
        for digit in previous:state=q*state+digit
        table[state]=[d for d in range(q) if d not in previous]
    return table


def sample(order,family,seed,rows=2400,q=9,candidate_cap=10000,batch=512):
    n=len(order);k=int(family[-1]);assert n>2*k and q>k
    rng=np.random.default_rng(seed);table=allowed_table(q,k);accepted=[];indices=[];generated=0;kept=0
    wrap=[(left,n-distance+left) for distance in range(1,k+1) for left in range(distance)]
    assert len(wrap)==k*(k+1)//2
    while kept<rows and generated<candidate_cap:
        size=min(batch,candidate_cap-generated);path=np.empty((size,n),np.int64)
        path[:,:k]=rng.permuted(np.broadcast_to(np.arange(q),(size,q)),axis=1)[:,:k]
        for pos in range(k,n):
            choices=rng.integers(0,q-k,size,dtype=np.int64);state=np.zeros(size,np.int64)
            for prior in range(pos-k,pos):state=state*q+path[:,prior]
            path[:,pos]=table[state,choices]
        valid=np.ones(size,bool)
        for left,right in wrap:valid&=path[:,left]!=path[:,right]
        good=np.flatnonzero(valid);accepted.append(path[good]);indices.extend((generated+good).tolist());generated+=size;kept+=len(good)
    if kept<rows:raise RuntimeError('proper_coloring_candidate_cap')
    canonical=np.concatenate(accepted)[:rows];result=np.empty_like(canonical,dtype=np.uint8);result[:,order]=canonical
    return result,dict(generated_candidates=generated,retained_rows=rows,accepted_candidate_indices=indices[:rows],candidate_cap=candidate_cap,batch=batch)
