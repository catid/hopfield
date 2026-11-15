"""Acquisition policy boundary: past queried counts only; no current-row access."""
import numpy as np


def choose_mask(arm, joint, equal, k, tie_order):
    assert arm in ('uniform','balanced','candidate') and joint.shape==equal.shape==(81,81)
    assert np.array_equal(np.sort(tie_order),np.arange(81)) and k==25
    if arm=='uniform':
        selected=tie_order[:k]
    else:
        weight=1.0/(1.0+joint)
        if arm=='candidate': weight*=equal==0
        np.fill_diagonal(weight,0.)
        degree=weight.sum(1)
        first=int(tie_order[np.argmax(degree[tie_order])])
        selected=[first]; scores=weight[:,first].copy(); scores[first]=-np.inf
        for _ in range(1,k):
            cell=int(tie_order[np.argmax(scores[tie_order])]); selected.append(cell)
            scores+=weight[:,cell]; scores[selected]=-np.inf
    mask=np.zeros(81,bool);mask[selected]=True
    assert mask.sum()==k
    return mask


def learn_queried(joint,equal,mask,queried):
    cells=np.flatnonzero(mask)
    assert queried.shape==(25,) and np.all((queried>=0)&(queried<9))
    block=np.ix_(cells,cells)
    joint[block]+=1
    equal[block]+=(queried[:,None]==queried[None,:]).astype(np.int32)
