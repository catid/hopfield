"""H1be: stop policy feedback updates after a fixed prefix; retain actual counts."""
import hashlib
import numpy as np
from soft_feedback_policy import choose


def collect(clean,uniforms,permutations,tie_seed,freeze_at=300,checkpoint=450):
    n=len(clean);assert 0<freeze_at<checkpoint<=n and clean.shape==(n,81) and uniforms.shape==(n,) and permutations.shape==(n,81)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);given=np.zeros((n,81),bool);values=np.full((n,81),-1,np.int8)
    ties=np.random.default_rng(tie_seed);events=[];prefix=None;frozen=None;ii,jj=np.triu_indices(81,1)
    for row in range(n):
        if row==freeze_at:frozen=equal.copy();frozen.setflags(write=False)
        feedback=equal if row<freeze_at else frozen
        # The state and mask are finalized before indexing the current source or latent.
        digest=hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest();feedback_digest=hashlib.sha256(feedback.tobytes()).hexdigest()
        joint.setflags(write=False);equal.setflags(write=False)
        mask=choose(joint,feedback,ties.permutation(81))
        assert digest==hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest() and feedback_digest==hashlib.sha256(feedback.tobytes()).hexdigest()
        events.append(dict(row=row,past_state_sha256=digest,policy_feedback_sha256=feedback_digest,feedback_frozen=row>=freeze_at,
                           mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),mask_finalized_before_current_labels_and_latents=True))
        joint.setflags(write=True);equal.setflags(write=True)
        observed=clean[row,permutations[row]] if uniforms[row]<9/160 else clean[row]
        given[row]=mask;values[row,mask]=observed[mask]
        cells=np.flatnonzero(mask);block=np.ix_(cells,cells);labels=values[row,cells]
        joint[block]+=1;equal[block]+=labels[:,None]==labels[None,:]
        if row+1==checkpoint:prefix=(joint[ii,jj].copy(),equal[ii,jj].copy())
    return dict(given=given,values=values,joint_count=joint[ii,jj],equal_count=equal[ii,jj],
                feedback_equal_count=frozen[ii,jj].copy(),prefix_joint_count=prefix[0],prefix_equal_count=prefix[1]),events
