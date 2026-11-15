"""Fixed exponential acquisition ablation; candidate and mixture controls unchanged."""
import numpy as np
from spike_policy import choose_mask as original_choose_mask, learn_queried


def predictive_weights(joint,equal):
    # Fixed exponential exposure discount, zero after any witnessed equality.
    u=np.power(8.0/9.0,joint.astype(np.float64))
    weight=np.where(equal==0,u,0.0)
    np.fill_diagonal(weight,0.)
    return weight


def choose_mask(arm,joint,equal,k,tie_order):
    if arm in ('candidate','spike'):
        return original_choose_mask(arm,joint,equal,k,tie_order)
    assert arm=='exponential' and joint.shape==equal.shape==(81,81) and k==25
    assert np.array_equal(np.sort(tie_order),np.arange(81))
    weight=predictive_weights(joint,equal)
    first=int(tie_order[np.argmax(weight.sum(1)[tie_order])])
    selected=[first];scores=weight[:,first].copy();scores[first]=-np.inf
    for _ in range(1,k):
        cell=int(tie_order[np.argmax(scores[tie_order])]);selected.append(cell)
        scores+=weight[:,cell];scores[selected]=-np.inf
    mask=np.zeros(81,bool);mask[selected]=True
    assert mask.sum()==k
    return mask
