"""Count threshold from the supplied corruption law, with a null-edge bound.

The estimator receives corrupted labels and original free-cell masks, not clean
labels or graph geometry. Alpha bounds the chance of dropping ANY truly never-
equal pair under independent rows and the stated corruption law. It does not
bound false extra edges or arbitrary learned stopping certificates.
"""
import numpy as np


def null_means(free,corrupted_cells):
    free=np.asarray(free,dtype=bool);m=free.sum(1).astype(np.float64)
    assert free.ndim==2 and free.shape[1]>=2 and np.all(m>=max(2,corrupted_cells))
    a=corrupted_cells/m;b=corrupted_cells*(corrupted_cells-1)/(m*(m-1))
    f=free.astype(np.float64);single=(f*(a/8)[:,None]).sum(0)
    both=f.T@(f*(9*b/64)[:,None])
    mean=single[:,None]+single[None,:]-both
    i,j=np.triu_indices(free.shape[1],1)
    return mean[i,j]


def chernoff_upper(k,n,mean):
    mean=np.asarray(mean,dtype=np.float64);q=k/n;p=mean/n
    if k<=0:return np.ones_like(p)
    if k>n:return np.zeros_like(p)
    result=np.ones_like(p);inside=(p>0)&(p<q)&(p<1)
    d=np.zeros_like(p[inside])
    if q>0:d+=q*np.log(q/p[inside])
    if q<1:d+=(1-q)*np.log((1-q)/(1-p[inside]))
    result[inside]=np.exp(-n*d);result[p==0]=0
    return result


def thresholds(free,corrupted_cells,alpha=.01):
    """Smallest retained integer threshold with per-pair tail<=alpha/allpairs."""
    mean=null_means(free,corrupted_cells);n=len(free);threshold=np.full(len(mean),n,dtype=np.int64)
    pending=np.ones(len(mean),bool)
    for k in range(1,n+2):
        passed=pending&(chernoff_upper(k,n,mean)<=alpha/len(mean))
        threshold[passed]=k-1;pending[passed]=False
        if not pending.any():break
    assert not pending.any()
    return threshold,mean


def fit_counts(counts,free,corrupted_cells,alpha=.01):
    threshold,mean=thresholds(free,corrupted_cells,alpha)
    counts=np.asarray(counts);assert counts.shape==threshold.shape
    return counts<=threshold,threshold,mean
