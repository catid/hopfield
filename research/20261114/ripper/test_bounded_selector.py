"""Independent weights,ties,read-only history and extreme-count checks."""
from fractions import Fraction
import hashlib
import inspect
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from bounded_selector_policy import choose_mask,bounded_weights,calibration
from noisy_policy import choose_mask as exposure


def reference(joint,equal,excluded,order,la,lb):
    log_lr=joint.astype(float)*lb+equal.astype(float)*(la-lb)
    weights=np.power(8./9.,joint.astype(float))*np.exp(np.minimum(np.maximum(log_lr/2.,0.),math.log(16.)))
    weights[excluded]=0.;weights[np.diag_indices(81)]=0.;selected=[]
    for _ in range(25):
        score=weights.sum(1) if not selected else np.zeros(81)
        for c in selected:score=score+weights[:,c]
        score[selected]=-np.inf;best=score.max();selected.append(next(int(c) for c in order if score[c]==best))
    mask=np.zeros(81,bool);mask[selected]=True;return mask


def main():
    start=time.monotonic();rng=np.random.default_rng(20363000);checks=0;extreme=0
    assert list(inspect.signature(choose_mask).parameters)==['arm','joint','equal','excluded','k','tie_order','log_a','log_b']
    for eta in [Fraction(1,20),Fraction(7,100),Fraction(8,9)]:
        q0,a,b,fallback=calibration(eta);la,lb=math.log(float(a)),math.log(float(b))
        for index in range(128):
            joint=rng.integers(0,2221,(81,81),dtype=np.int64);joint=(joint+joint.T)//2
            equal=np.minimum(joint,rng.integers(0,501,(81,81)));equal=(equal+equal.T)//2
            excluded=rng.random((81,81))<.35;excluded=excluded|excluded.T;order=rng.permutation(81)
            if index%32==0:joint[:]=0;equal[:]=0;excluded[:]=False
            if index%32==1:excluded[:]=True
            if index%32==2:joint[:]=2000000000;equal[:]=2000000000;extreme+=1
            if index%32==3:joint[:]=2000000000;equal[:]=0;extreme+=1
            for x in [joint,equal,excluded]:x.setflags(write=False)
            before=hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
            with np.errstate(under='ignore'):
                expected=reference(joint,equal,excluded,order,la,lb)
                for arm in ['bounded_exact','bounded_learned']:
                    assert np.array_equal(choose_mask(arm,joint,equal,excluded,25,order,la,lb),expected);checks+=1
                for arm in ['exposure_exact','exposure_learned']:
                    assert np.array_equal(choose_mask(arm,joint,equal,excluded,25,order,la,lb),exposure('exponential_e',joint,excluded,25,order));checks+=1
            w=bounded_weights(joint,equal,excluded,la,lb);assert np.all(np.isfinite(w)) and w.max()<=16*(1+1e-14)
            base=np.power(8./9.,joint.astype(float));active=~excluded&~np.eye(81,dtype=bool)
            assert np.all(w[active]>=base[active]) and np.all(w[active]<=16*base[active]*(1+1e-14))
            assert before==hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()
            if index%32 in [0,1]:assert np.array_equal(np.flatnonzero(expected),np.sort(order[:25]))
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'bounded_selector_policy.py'),
        'independent_mask_comparisons':checks,'extreme_count_cases':extreme,'all_tied_and_excluded_cases':24,'readonly_history_cases':384,
        'selector_signature_past_only':True,'finite_weight_cap':16.,'multiplicative_tilt_bounds':[1.,16.],'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/bounded_selector_preflight.json',r);print(r)


if __name__=='__main__':main()
