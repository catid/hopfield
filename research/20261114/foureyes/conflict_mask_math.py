"""Observable conflict ranking; no true corruption locations in this module."""
import numpy as np


def checked(noisy, primal_twice):
    assert noisy.ndim==2 and np.issubdtype(noisy.dtype,np.integer)
    assert ((noisy>=0)&(noisy<=8)).all()
    n=noisy.shape[1]
    assert primal_twice.dtype==np.int64 and primal_twice.shape==(n*(n-1)//2,)
    assert np.isin(primal_twice,[0,1,2]).all()
    ii,jj=np.triu_indices(n,1)
    matrix=np.zeros((n,n),np.int64);matrix[ii,jj]=matrix[jj,ii]=primal_twice
    return matrix


def sparse_scores(noisy, primal_twice):
    matrix=checked(noisy,primal_twice)
    scores=np.zeros_like(noisy,dtype=np.int64)
    for i in range(noisy.shape[1]):
        neighbors=np.flatnonzero(matrix[i])
        scores[:,i]=(noisy[:,neighbors]==noisy[:,i,None])@matrix[i,neighbors]
    return scores


def dense_scores(noisy, primal_twice):
    matrix=checked(noisy,primal_twice)
    return ((noisy[:,:,None]==noisy[:,None,:])*matrix[None]).sum(2,dtype=np.int64)


def literal_scores(noisy, primal_twice):
    matrix=checked(noisy,primal_twice)
    return np.array([[sum(int(matrix[i,j]) for j in range(len(row)) if row[i]==row[j])
                      for i in range(len(row))] for row in noisy],np.int64)


def detector(noisy, primal_twice, k=9):
    assert type(k) is int and 0<=k<=noisy.shape[1]
    scores=sparse_scores(noisy,primal_twice)
    indices=np.argsort(-scores,axis=1,kind='stable')[:,:k]
    mask=np.zeros_like(noisy,dtype=bool)
    mask[np.arange(len(noisy))[:,None],indices]=True
    return scores,mask


def independent_detector(noisy, primal_twice, k=9):
    assert type(k) is int and 0<=k<=noisy.shape[1]
    scores=dense_scores(noisy,primal_twice)
    mask=np.zeros_like(noisy,dtype=bool)
    for r,row in enumerate(scores):
        selected=sorted(range(len(row)),key=lambda i:(-int(row[i]),i))[:k]
        mask[r,selected]=True
    return scores,mask


def random_mask(rows, n, seed, k=9):
    assert all(type(v) is int for v in [rows,n,seed,k]) and rows>0 and 0<=k<=n and seed>=0
    rng=np.random.Generator(np.random.PCG64(seed))
    mask=np.zeros((rows,n),bool)
    for row in range(rows):
        mask[row,rng.choice(n,k,replace=False)]=True
    return mask


def independent_random_mask(rows, n, seed, k=9):
    rng=np.random.default_rng(seed)
    indices=[set(map(int,rng.choice(n,k,replace=False))) for _ in range(rows)]
    return np.array([[cell in row for cell in range(n)] for row in indices],bool)
