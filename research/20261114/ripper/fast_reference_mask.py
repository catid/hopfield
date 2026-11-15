"""Same left-to-right floating sums as the independent replay, less repeated work."""
import numpy as np


def reference_mask(arm,joint,excluded,order):
    weights=(8./9.)**joint.astype(np.float64);weights[excluded]=0.;weights[np.diag_indices(81)]=0.
    chosen=[];accumulator=None
    for step in range(25):
        score=weights.sum(1) if not chosen else accumulator.copy()
        score[chosen]=-np.inf;best=score.max()
        cell=next(int(c) for c in order if score[c]==best);chosen.append(cell)
        # Reuse exactly the old left-to-right sum. No reassociation or reduction.
        if accumulator is None:accumulator=np.zeros(81)+weights[:,cell]
        else:accumulator=accumulator+weights[:,cell]
    mask=np.zeros(81,bool);mask[chosen]=True;return mask
