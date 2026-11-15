"""H2br token-only Manhattan182 dispatch and explicit final-source scatter."""
import numpy as np
THRESHOLD=182
FIELDS=('prediction','iterations','stopped','finite','own_margin','native_cap','block_cap')


def partition(tokens):
    tokens=np.asarray(tokens)
    if tokens.ndim!=3 or tokens.shape[1:]!=(180,180) or not len(tokens) or tokens.dtype.kind not in 'buif' or not np.all(np.isin(tokens,[0,1,2,3])):
        raise ValueError('Expected nonempty batches of180x180 maze tokens')
    flat=tokens.reshape(len(tokens),-1)
    if not np.all((flat==2).sum(1)==1) or not np.all((flat==3).sum(1)==1):
        raise ValueError('Exactly one start and target per input required')
    start=(flat==2).argmax(1);goal=(flat==3).argmax(1)
    distance=np.abs(start//180-goal//180)+np.abs(start%180-goal%180)
    high=distance>=THRESHOLD
    return distance,high,np.flatnonzero(high),np.flatnonzero(~high)


def lp_indices(n,bp_indices,geometry_valid):
    indices=np.asarray(bp_indices);valid=np.asarray(geometry_valid)
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=n) or not np.array_equal(indices,np.unique(indices)):
        raise ValueError('BP indices must be unique, sorted and in range')
    if valid.shape!=indices.shape or valid.dtype!=np.bool_:raise ValueError('One geometry flag per actual BP input required')
    selected=np.ones(n,bool);selected[indices[valid]]=False
    return np.flatnonzero(selected)


def assemble(n,bp,lp,bp_indices,selected_lp):
    bi=np.asarray(bp_indices);li=np.asarray(selected_lp)
    for indices in [bi,li]:
        if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=n) or not np.array_equal(indices,np.unique(indices)):
            raise ValueError('Unique ascending in-range indices required')
    if len(np.union1d(bi,li))!=n:raise ValueError('Every input must have an actual answer producer')
    out={}
    for field in FIELDS:
        b,l=np.asarray(bp[field]),np.asarray(lp[field])
        if b.shape[0]!=len(bi) or l.shape[0]!=len(li) or b.shape[1:]!=l.shape[1:] or b.dtype!=l.dtype:raise ValueError('Actual subset shape/dtype mismatch: '+field)
        answer=np.empty((n,*b.shape[1:]),b.dtype);answer[bi]=b;answer[li]=l;out[field]=answer
    out['chosen_model']=np.zeros(n,np.int8);out['chosen_model'][li]=1
    return out
