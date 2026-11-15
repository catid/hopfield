"""H1bf source-ID-only donors and label-independent recipient acquisition."""
import hashlib
import numpy as np
from soft_feedback_policy import choose
from prefix_rate_features import prefix_counts


def donor_map(source_rows,prefix=300):
    assert source_rows.ndim==2 and 0<prefix<=source_rows.shape[1]
    n=len(source_rows);full=[set(map(int,row)) for row in source_rows];short=[set(map(int,row[:prefix])) for row in source_rows]
    allowed=[[d for d in range(n) if d!=r and full[r].isdisjoint(short[d])] for r in range(n)]
    assigned={}
    def augment(recipient,seen):
        for donor in allowed[recipient]:
            if donor in seen:continue
            seen.add(donor)
            if donor not in assigned or augment(assigned[donor],seen):assigned[donor]=recipient;return True
        return False
    for recipient in range(n):
        if not augment(recipient,set()):return None
    result=np.empty(n,np.int64)
    for donor,recipient in assigned.items():result[recipient]=donor
    assert set(map(int,result))==set(range(n)) and all(int(d) in allowed[r] for r,d in enumerate(result))
    return result


def masks(feedback,tie_seed,rows=900):
    assert feedback.shape==(81,81) and np.array_equal(feedback,feedback.T) and (feedback>=0).all()
    feedback=feedback.copy();feedback.setflags(write=False);digest=hashlib.sha256(feedback.tobytes()).hexdigest()
    rng=np.random.default_rng(tie_seed);given=np.zeros((rows,81),bool);joint=np.zeros((81,81),np.int32);events=[]
    for row in range(rows):
        prior=hashlib.sha256(joint.tobytes()).hexdigest();joint.setflags(write=False)
        mask=choose(joint,feedback,rng.permutation(81))
        assert prior==hashlib.sha256(joint.tobytes()).hexdigest() and digest==hashlib.sha256(feedback.tobytes()).hexdigest()
        joint.setflags(write=True);given[row]=mask;cells=np.flatnonzero(mask);joint[np.ix_(cells,cells)]+=1
        events.append(dict(row=row,past_joint_sha256=prior,feedback_sha256=digest,mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
    return given,events


def observe(given,clean,noise):
    n=len(given);assert clean.shape==given.shape and noise['uniforms'].shape==(n,) and noise['permutations'].shape==(n,81)
    values=np.full((n,81),-1,np.int8)
    for row in range(n):
        board=clean[row,noise['permutations'][row]] if noise['uniforms'][row]<9/160 else clean[row]
        values[row,given[row]]=board[given[row]]
    m,c=prefix_counts(given,values,[n])[n]
    return dict(given=given,values=values,joint_count=m,equal_count=c)


def acquire(feedback,tie_seed,clean,noise):
    given,events=masks(feedback,tie_seed,len(clean))
    return observe(given,clean,noise),events
