"""Fixed equal mixture e-process, exact rational crossings, exposure masks."""
from fractions import Fraction
from functools import lru_cache
import math
import numpy as np
from calibrated_policy import crossing_thresholds as fixed_thresholds
from noisy_policy import choose_mask as exposure_mask
from online_policy import learn_queried

BOUND=648000
ALTERNATIVES=(Fraction(1,9),Fraction(1,6))


@lru_cache(None)
def components(eta):
    assert isinstance(eta,Fraction) and 0<eta<=Fraction(8,9)
    q0=eta/4-9*eta*eta/64
    if q0>=ALTERNATIVES[0]:return q0,((Fraction(1),Fraction(1)),)*2,True
    assert all(q0<q<1 for q in ALTERNATIVES)
    return q0,tuple((q/q0,(1-q)/(1-q0)) for q in ALTERNATIVES),False


def exact_mixture_crosses(pairs,m,c):
    if c<0:return False
    if c>m:return True
    parts=[]
    for a,b in pairs:
        parts.append((pow(a.numerator,c)*pow(b.numerator,m-c),
                      pow(a.denominator,c)*pow(b.denominator,m-c)))
    (n0,d0),(n1,d1)=parts
    return n0*d1+n1*d0>=2*BOUND*d0*d1


@lru_cache(None)
def crossing_thresholds(eta,max_joint,mixture=False):
    if not mixture:
        result=fixed_thresholds(eta,max_joint);result.setflags(write=False);return result
    q0,pairs,fallback=components(eta)
    if fallback:
        result=np.arange(1,max_joint+2,dtype=np.int32);result.setflags(write=False);return result
    logs=[(math.log(float(a)),math.log(float(b))) for a,b in pairs]
    thresholds=[]
    for m in range(max_joint+1):
        # If either component reaches 2*BOUND then the equal mixture crosses.
        # This floating estimate only proposes a count; exact integers certify it.
        c=max(0,min(m+1,min(math.ceil((math.log(2*BOUND)-m*lb)/(la-lb)) for la,lb in logs)))
        while c<=m and not exact_mixture_crosses(pairs,m,c):c+=1
        while c>0 and exact_mixture_crosses(pairs,m,c-1):c-=1
        assert c==m+1 or exact_mixture_crosses(pairs,m,c)
        assert c==0 or not exact_mixture_crosses(pairs,m,c-1)
        thresholds.append(c)
    result=np.array(thresholds,np.int32);result.setflags(write=False);return result


def log_evidence(joint,equal,eta,mixture=False):
    assert joint.shape==equal.shape and np.all((equal>=0)&(equal<=joint))
    q0,pairs,fallback=components(eta)
    if fallback:return np.zeros(joint.shape,np.float64)
    logs=[]
    for a,b in pairs[:2 if mixture else 1]:
        la,lb=math.log(float(a)),math.log(float(b))
        logs.append(joint.astype(np.float64)*lb+equal.astype(np.float64)*(la-lb))
    return np.logaddexp(logs[0],logs[1])-math.log(2.) if mixture else logs[0]


def choose_mask(arm,joint,excluded,k,tie_order):
    assert arm in ('fixed_exact','mixture_exact','fixed_learned','mixture_learned')
    return exposure_mask('exponential_e',joint,excluded,k,tie_order)
