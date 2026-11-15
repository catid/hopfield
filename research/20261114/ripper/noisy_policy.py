"""Pure fixed-rate masks and exact e-process crossing thresholds."""
from fractions import Fraction
import math
import numpy as np
from online_policy import learn_queried

ETA=Fraction(1,100)
Q0=ETA/4-9*ETA*ETA/64
Q1=Fraction(1,9)
BOUND=324000
A=Q1/Q0
B=(1-Q1)/(1-Q0)
LOG_A=math.log(float(A))
LOG_B=math.log(float(B))


def crossing_thresholds(max_joint):
    """Smallest equality count c making the exact LR >= BOUND, for each m."""
    result=[]
    def crosses(m,c):
        if c<0:return False
        if c>m:return True
        left=pow(A.numerator,c)*pow(B.numerator,m-c)
        right=BOUND*pow(A.denominator,c)*pow(B.denominator,m-c)
        return left>=right
    for m in range(max_joint+1):
        c=max(0,min(m+1,math.ceil((math.log(BOUND)-m*LOG_B)/(LOG_A-LOG_B))))
        while c<=m and not crosses(m,c):c+=1
        while c>0 and crosses(m,c-1):c-=1
        assert c==m+1 or crosses(m,c)
        assert c==0 or not crosses(m,c-1)
        result.append(c)
    return np.array(result,np.int32)


def choose_mask(arm,joint,excluded,k,tie_order):
    assert joint.shape==excluded.shape==(81,81) and k==25
    assert np.array_equal(np.sort(tie_order),np.arange(81))
    if arm=='uniform_e':
        mask=np.zeros(81,bool);mask[tie_order[:25]]=True;return mask
    assert arm in ('exponential_e','exponential_naive')
    weight=np.where(excluded,0.,np.power(8./9.,joint.astype(np.float64)))
    np.fill_diagonal(weight,0.)
    first=int(tie_order[np.argmax(weight.sum(1)[tie_order])]);chosen=[first]
    score=weight[:,first].copy();score[first]=-np.inf
    for _ in range(1,k):
        cell=int(tie_order[np.argmax(score[tie_order])]);chosen.append(cell)
        score+=weight[:,cell];score[chosen]=-np.inf
    mask=np.zeros(81,bool);mask[chosen]=True;return mask
