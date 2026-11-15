"""Unused-by-H1ag exact recurrence candidate for future mixture tables."""
from fractions import Fraction
import numpy as np


def thresholds(eta,max_joint):
    assert isinstance(eta,Fraction) and 0<eta<=Fraction(8,9)
    null=eta/4-9*eta*eta/64
    if null>=Fraction(1,9):return np.arange(1,max_joint+2,dtype=np.int32)
    ratios=[(alt/null,(1-alt)/(1-null)) for alt in (Fraction(1,9),Fraction(1,6))]
    assert all(a>1 and 0<b<1 for a,b in ratios)
    numer=[1,1];denom=[1,1];c=0;out=[]
    def crosses(ns,ds):return ns[0]*ds[1]+ns[1]*ds[0]>=1296000*ds[0]*ds[1]
    for m in range(max_joint+1):
        if m:
            for k,(a,b) in enumerate(ratios):numer[k]*=b.numerator;denom[k]*=b.denominator
        # Keep unreduced products: replacing one B factor by A is exact division.
        while c<m and not crosses(numer,denom):
            for k,(a,b) in enumerate(ratios):
                assert numer[k]%b.numerator==denom[k]%b.denominator==0
                numer[k]=numer[k]//b.numerator*a.numerator
                denom[k]=denom[k]//b.denominator*a.denominator
            c+=1
        if crosses(numer,denom):
            if c:
                previous_n=[];previous_d=[]
                for k,(a,b) in enumerate(ratios):
                    assert numer[k]%a.numerator==denom[k]%a.denominator==0
                    previous_n.append(numer[k]//a.numerator*b.numerator)
                    previous_d.append(denom[k]//a.denominator*b.denominator)
                assert not crosses(previous_n,previous_d)
            out.append(c)
        else:
            assert c==m;out.append(m+1)
    return np.array(out,np.int32)
