"""Prospective H2bs stable offline queues; no outcome or label argument exists."""
import math
import numpy as np
from endpoint_dispatch_policy import partition


def identities(indices):
    value=np.asarray(indices)
    if value.ndim!=1 or value.dtype.kind not in 'iu' or not len(value) or np.any(value<0) or np.any(value>=4096) or not np.array_equal(value,np.unique(value)):
        raise ValueError('Unique ascending source identities required')
    return value.astype(np.int64,copy=False)


def stable_batches(indices,high=None):
    indices=identities(indices)
    if high is None:groups=[('original',indices)]
    else:
        high=np.asarray(high)
        if high.dtype!=np.bool_ or high.shape!=indices.shape:raise ValueError('One boolean partition flag per identity required')
        groups=[('high',indices[high]),('low',indices[~high])]
    batches=[{'group':label,'indices':group[offset:offset+32].copy()} for label,group in groups for offset in range(0,len(group),32)]
    slots=math.ceil(len(indices)/32)+1
    assert len(batches)<=slots
    actual=np.concatenate([b['indices'] for b in batches]);assert np.array_equal(np.sort(actual),indices)
    assert all(1<=len(b['indices'])<=32 for b in batches)
    return batches,slots


def make_queue(tokens_by_shard,indices,mode):
    """Call inside the charged per-model host preparation interval."""
    indices=identities(indices)
    if mode not in ['original_order','packed']:raise ValueError('Unknown ordering')
    groups=[];distances=None;high=None
    if mode=='packed':
        distances=np.empty(len(indices),np.int64)
        for shard in np.unique(indices//256):
            positions=np.flatnonzero(indices//256==shard);local=indices[positions]%256
            distance,_,_,_=partition(tokens_by_shard[int(shard)][local])
            distances[positions]=distance
        high=distances>=182
    batches,slots=stable_batches(indices,high)
    return {'mode':mode,'indices':indices.copy(),'distances':distances,'high':high,'batches':batches,'slots_per_model':slots,'zero_work_slots':slots-len(batches)}


def materialize(tokens_by_shard,indices):
    """Call inside the charged actual-batch materialization interval."""
    indices=np.asarray(indices)
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or not 1<=len(indices)<=32 or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=4096):
        raise ValueError('One actual unpadded batch of unique source identities required')
    return np.stack([tokens_by_shard[int(i)//256][int(i)%256] for i in indices])


def scatter(final,seen,original_indices,batch_indices,answer):
    """Charged final output materialization in ascending original identity order."""
    positions=np.searchsorted(original_indices,batch_indices)
    if len(np.unique(batch_indices))!=len(batch_indices) or np.any(positions>=len(original_indices)) or not np.array_equal(original_indices[positions],batch_indices) or np.any(seen[positions]):
        raise ValueError('Unknown or duplicate scatter identity')
    if not final:
        for name,value in answer.items():final[name]=np.empty((len(original_indices),*value.shape[1:]),value.dtype)
    if set(final)!=set(answer):raise ValueError('Changed answer fields')
    for name,value in answer.items():
        if len(value)!=len(batch_indices) or final[name].shape[1:]!=value.shape[1:] or final[name].dtype!=value.dtype:raise ValueError('Answer shape or dtype mismatch')
        final[name][positions]=value
    seen[positions]=True
    return positions
