"""Opposite-fold-only conflict masking; no corruption labels or geometry."""
import numpy as np
from conflict_mask_math import detector, independent_detector


def fold_costs(noisy):
    assert noisy.ndim==2 and len(noisy)>0 and len(noisy)%2==0
    assert np.issubdtype(noisy.dtype,np.integer) and ((noisy>=0)&(noisy<=8)).all()
    middle=len(noisy)//2;ii,jj=np.triu_indices(noisy.shape[1],1)
    return [(rows[:,ii]==rows[:,jj]).sum(0,dtype=np.int64)
            for rows in [noisy[:middle],noisy[middle:]]]


def crossfit_mask(noisy, first_half_primal, second_half_primal):
    assert noisy.ndim==2 and len(noisy)>0 and len(noisy)%2==0
    middle=len(noisy)//2
    first_scores,first_mask=detector(noisy[:middle],second_half_primal)
    second_scores,second_mask=detector(noisy[middle:],first_half_primal)
    return np.concatenate([first_scores,second_scores]),np.concatenate([first_mask,second_mask])


def independent_fold_costs(noisy):
    assert noisy.ndim==2 and len(noisy)>0 and len(noisy)%2==0
    middle=len(noisy)//2
    result=[]
    for lo,hi in [(0,middle),(middle,len(noisy))]:
        result.append(np.array([np.count_nonzero(noisy[lo:hi,i]==noisy[lo:hi,j])
            for i in range(noisy.shape[1]) for j in range(i+1,noisy.shape[1])],np.int64))
    return result


def independent_crossfit_mask(noisy, first_half_primal, second_half_primal):
    middle=len(noisy)//2;scores=[];masks=[]
    for r,row in enumerate(noisy):
        primal=second_half_primal if r<middle else first_half_primal
        score,mask=independent_detector(row[None],primal)
        scores.append(score[0]);masks.append(mask[0])
    return np.array(scores,np.int64),np.array(masks,bool)
