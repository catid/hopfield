"""Fixed exponential masks with separate exact and conservative calibrations."""
from fractions import Fraction
import math
import numpy as np
from noisy_policy import choose_mask as frozen_mask
from online_policy import learn_queried

ASSUMED={'exact':Fraction(1,100),'upper':Fraction(1,50),'under':Fraction(1,200)}
Q1=Fraction(1,9)
BOUND=324000


def calibration(arm):
    eta=ASSUMED[arm];q0=eta/4-9*eta*eta/64
    return q0,Q1/q0,(1-Q1)/(1-q0)


def crossing_thresholds(arm,max_joint):
    q0,a,b=calibration(arm);la=math.log(float(a));lb=math.log(float(b))
    def crosses(m,c):
        if c<0:return False
        if c>m:return True
        return pow(a.numerator,c)*pow(b.numerator,m-c)>=BOUND*pow(a.denominator,c)*pow(b.denominator,m-c)
    result=[]
    for m in range(max_joint+1):
        c=max(0,min(m+1,math.ceil((math.log(BOUND)-m*lb)/(la-lb))))
        while c<=m and not crosses(m,c):c+=1
        while c>0 and crosses(m,c-1):c-=1
        assert c==m+1 or crosses(m,c)
        assert c==0 or not crosses(m,c-1)
        result.append(c)
    return np.array(result,np.int32)


def choose_mask(arm,joint,excluded,k,tie_order):
    assert arm in ASSUMED
    return frozen_mask('exponential_e',joint,excluded,k,tie_order)
