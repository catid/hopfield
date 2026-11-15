"""Predictable exposure-weighted soft feedback; no calibration or exclusions."""
import hashlib
import numpy as np


def choose(joint,equal,order,permutation=None):
    feedback=equal if permutation is None else equal[np.ix_(permutation,permutation)]
    weight=np.power(8./9.,joint.astype(np.float64))/(1.+feedback)
    np.fill_diagonal(weight,0.)
    first=int(order[np.argmax(weight.sum(1)[order])]);selected=[first];score=weight[:,first].copy();score[first]=-np.inf
    for _ in range(24):
        cell=int(order[np.argmax(score[order])]);selected.append(cell);score+=weight[:,cell];score[selected]=-np.inf
    mask=np.zeros(81,bool);mask[selected]=True;return mask


def collect(clean,uniforms,permutations,tie_seed,feedback_permutation=None,checkpoint=450):
    n=len(clean);assert clean.shape==(n,81) and uniforms.shape==(n,) and permutations.shape==(n,81)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);given=np.zeros((n,81),bool);values=np.full((n,81),-1,np.int8)
    ties=np.random.default_rng(tie_seed);events=[];prefix=None;ii,jj=np.triu_indices(81,1)
    for row in range(n):
        # No current source/latent access occurs until this mask and prior-state digest exist.
        digest=hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()
        joint.setflags(write=False);equal.setflags(write=False)
        mask=choose(joint,equal,ties.permutation(81),feedback_permutation)
        assert digest==hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()
        events.append(dict(row=row,past_state_sha256=digest,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),mask_finalized_before_current_labels_and_latents=True))
        joint.setflags(write=True);equal.setflags(write=True)
        observed=clean[row,permutations[row]] if uniforms[row]<9/160 else clean[row]
        given[row]=mask;values[row,mask]=observed[mask]
        cells=np.flatnonzero(mask);block=np.ix_(cells,cells);labels=values[row,cells]
        joint[block]+=1;equal[block]+=labels[:,None]==labels[None,:]
        if row+1==checkpoint:prefix=(joint[ii,jj].copy(),equal[ii,jj].copy())
    return dict(given=given,values=values,joint_count=joint[ii,jj],equal_count=equal[ii,jj],prefix_joint_count=prefix[0],prefix_equal_count=prefix[1]),events
