"""Frozen exposure rule and half-budget e-process with a supplied noise bound."""
from fractions import Fraction
import math
import numpy as np
from noisy_policy import choose_mask as frozen_mask
from online_policy import learn_queried

BOUND=648000
Q1=Fraction(1,9)


def calibration(eta):
    assert 0<eta<=Fraction(8,9)
    q0=eta/4-9*eta*eta/64
    if q0>=Q1:return q0,Fraction(1),Fraction(1),True
    return q0,Q1/q0,(1-Q1)/(1-q0),False


def crossing_thresholds(eta,max_joint):
    q0,a,b,fallback=calibration(eta)
    if fallback:return np.arange(1,max_joint+2,dtype=np.int32)
    la=math.log(float(a));lb=math.log(float(b))
    def crosses(m,c):
        if c<0:return False
        if c>m:return True
        return pow(a.numerator,c)*pow(b.numerator,m-c)>=BOUND*pow(a.denominator,c)*pow(b.denominator,m-c)
    out=[]
    for m in range(max_joint+1):
        c=max(0,min(m+1,math.ceil((math.log(BOUND)-m*lb)/(la-lb))))
        while c<=m and not crosses(m,c):c+=1
        while c>0 and crosses(m,c-1):c-=1
        assert c==m+1 or crosses(m,c)
        assert c==0 or not crosses(m,c-1)
        out.append(c)
    return np.array(out,np.int32)


def choose_mask(arm,joint,excluded,k,tie_order):
    assert arm in ('exact','learned','fixed_upper')
    return frozen_mask('exponential_e',joint,excluded,k,tie_order)
