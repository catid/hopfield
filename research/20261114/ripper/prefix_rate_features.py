"""H1ax prefix features: only rows strictly before each checkpoint contribute."""
import numpy as np


def prefix_counts(given,values,prefixes):
    assert given.ndim==2 and given.shape[1]==81 and given.dtype==bool and values.shape==given.shape
    assert values.dtype==np.int8 and prefixes==sorted(set(prefixes)) and prefixes[0]>0 and prefixes[-1]<=len(given)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);pair=np.triu_indices(81,1);snapshots={}
    for row in range(prefixes[-1]):
        cells=np.flatnonzero(given[row]);assert len(cells)==25
        labels=values[row,cells];assert ((labels>=0)&(labels<9)).all()
        view=np.ix_(cells,cells);joint[view]+=1;equal[view]+=labels[:,None]==labels[None,:]
        if row+1 in prefixes:snapshots[row+1]=(joint[pair].copy(),equal[pair].copy())
    return snapshots
