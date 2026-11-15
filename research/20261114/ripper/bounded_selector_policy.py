"""Past-only bounded LR tilt of exposure weights; crossing law unchanged."""
import math
import numpy as np
from calibrated_policy import calibration,crossing_thresholds,learn_queried
from noisy_policy import choose_mask as exposure_mask

TILT_CAP=16.
LOG_TILT_CAP=math.log(TILT_CAP)


def bounded_weights(joint,equal,excluded,log_a,log_b):
    assert joint.shape==equal.shape==excluded.shape==(81,81)
    assert np.all((equal>=0)&(equal<=joint))
    assert math.isfinite(log_a) and math.isfinite(log_b) and log_a>=0 and log_b<=0
    # Frozen FP64 order: m*logB + c*(logA-logB), divide by2, clip, exp;
    # multiply by the unchanged exposure base (8/9)^m.
    log_lr=joint.astype(np.float64)*log_b+equal.astype(np.float64)*(log_a-log_b)
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        base=np.power(8./9.,joint.astype(np.float64))
        tilt=np.exp(np.clip(log_lr/2.,0.,LOG_TILT_CAP))
        weight=base*tilt
    weight[excluded]=0.;np.fill_diagonal(weight,0.)
    assert np.all(np.isfinite(weight)) and np.all(weight>=0)
    return weight


def choose_mask(arm,joint,equal,excluded,k,tie_order,log_a,log_b):
    assert arm in ('exposure_exact','bounded_exact','exposure_learned','bounded_learned') and k==25
    assert np.array_equal(np.sort(tie_order),np.arange(81))
    if arm.startswith('exposure_'):return exposure_mask('exponential_e',joint,excluded,k,tie_order)
    weight=bounded_weights(joint,equal,excluded,log_a,log_b)
    first=int(tie_order[np.argmax(weight.sum(1)[tie_order])]);chosen=[first]
    score=weight[:,first].copy();score[first]=-np.inf
    for _ in range(1,k):
        cell=int(tie_order[np.argmax(score[tie_order])]);chosen.append(cell)
        score+=weight[:,cell];score[chosen]=-np.inf
    mask=np.zeros(81,bool);mask[chosen]=True;return mask
