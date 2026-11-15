"""Exact first-crossing thresholds for three frozen equality-null rates."""
from fractions import Fraction
import math
import numpy as np

def thresholds(q0,max_joint=1110):
    assert isinstance(q0,Fraction)and 0<q0<Fraction(1,9)
    q1=Fraction(1,9);a=q1/q0;b=(1-q1)/(1-q0);bound=648000
    la,lb=math.log(float(a)),math.log(float(b))
    def above(m,c):
        if c<0:return False
        if c>m:return True
        return pow(a.numerator,c)*pow(b.numerator,m-c)>=bound*pow(a.denominator,c)*pow(b.denominator,m-c)
    result=[]
    for m in range(max_joint+1):
        c=max(0,min(m+1,math.ceil((math.log(bound)-m*lb)/(la-lb))))
        while c<=m and not above(m,c):c+=1
        while c>0 and above(m,c-1):c-=1
        assert c==m+1 or above(m,c)
        assert c==0 or not above(m,c-1)
        result.append(c)
    return np.asarray(result,np.int32)
