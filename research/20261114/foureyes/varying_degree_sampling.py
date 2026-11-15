"""Exact proper-coloring samplers for two specified regular graph families."""
import numpy as np
FAMILIES=['cycle81','cliques9']


def graph_order(seed,n=81):return np.random.default_rng(seed).permutation(n).astype(np.int64)


def graph_edges(order,family,q=9):
    n=len(order);assert np.array_equal(np.sort(order),np.arange(n))
    adjacency=np.zeros((n,n),bool)
    if family.startswith('cycle'):
        for i in range(n):a,b=order[i],order[(i+1)%n];adjacency[a,b]=adjacency[b,a]=True
    elif family.startswith('cliques'):
        assert n%q==0
        for start in range(0,n,q):
            block=order[start:start+q];adjacency[np.ix_(block,block)]=True
        np.fill_diagonal(adjacency,False)
    else:raise ValueError(family)
    return adjacency[np.triu_indices(n,1)]


def sample(order,family,seed,rows=2400,q=9,candidate_cap=10000,batch=512):
    n=len(order);rng=np.random.default_rng(seed);accepted=[];generated=0;accepted_indices=[]
    if family.startswith('cycle'):
        while sum(len(a) for a in accepted)<rows and generated<candidate_cap:
            size=min(batch,candidate_cap-generated)
            first=rng.integers(0,q,size,dtype=np.int64);increments=rng.integers(1,q,(size,n-1),dtype=np.int64)
            path=np.empty((size,n),np.int64);path[:,0]=first;path[:,1:]=(first[:,None]+np.cumsum(increments,axis=1))%q
            keep=np.flatnonzero(path[:,-1]!=path[:,0]);accepted.append(path[keep]);accepted_indices.extend((generated+keep).tolist());generated+=size
        if sum(len(a) for a in accepted)<rows:raise RuntimeError('proper_coloring_candidate_cap')
        canonical=np.concatenate(accepted)[:rows];chosen=np.array(accepted_indices[:rows],np.int64)
    elif family.startswith('cliques'):
        assert n%q==0;canonical=np.empty((rows,n),np.int64)
        for start in range(0,n,q):canonical[:,start:start+q]=rng.permuted(np.broadcast_to(np.arange(q),(rows,q)),axis=1)
        generated=rows;chosen=np.arange(rows,dtype=np.int64)
    else:raise ValueError(family)
    result=np.empty_like(canonical,dtype=np.uint8);result[:,order]=canonical
    return result,dict(generated_candidates=generated,retained_rows=rows,accepted_candidate_indices=chosen.tolist(),candidate_cap=candidate_cap,batch=batch)
