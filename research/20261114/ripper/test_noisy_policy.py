"""Exact null probabilities, martingale identity, integer crossings, pure masks."""
from decimal import Decimal,localcontext
from fractions import Fraction as F
import inspect
import numpy as np
from run import ROOT,HERE,save,sha,now
from noisy_policy import Q0,Q1,A,B,BOUND,crossing_thresholds,choose_mask
from exponential_policy import choose_mask as original


def main():
    probability_checks=0
    for a in range(9):
        for b in range(9):
            if a==b:continue
            probability=sum((F(99,100) if x==a else F(1,800))*(F(99,100) if x==b else F(1,800)) for x in range(9))
            assert probability==Q0;probability_checks+=1
    assert Q0*A+(1-Q0)*B==1 and Q1==F(1,9) and Q0==F(1591,640000)
    thresholds=crossing_thresholds(3000)
    with localcontext() as context:
        context.prec=70
        la=(Decimal(A.numerator)/A.denominator).ln();lb=(Decimal(B.numerator)/B.denominator).ln();boundary=Decimal(BOUND).ln()
        for m,c in enumerate(thresholds):
            c=int(c)
            if c<=m:assert c*la+(m-c)*lb>=boundary
            if c>0:assert (c-1)*la+(m-c+1)*lb<boundary
    assert list(inspect.signature(choose_mask).parameters)==['arm','joint','excluded','k','tie_order']
    rng=np.random.default_rng(20265009);control_checks=0
    for _ in range(64):
        joint=rng.integers(0,3001,(81,81),dtype=np.int32);joint=(joint+joint.T)//2
        equal=rng.integers(0,2,(81,81),dtype=np.int32);equal=np.maximum(equal,equal.T)
        excluded=equal>0;order=rng.permutation(81)
        joint.setflags(write=False);excluded.setflags(write=False)
        mask=choose_mask('exponential_naive',joint,excluded,25,order)
        assert np.array_equal(mask,original('exponential',joint,equal,25,order))
        assert np.array_equal(mask,choose_mask('exponential_e',joint,excluded,25,order))
        assert np.array_equal(mask,choose_mask('exponential_naive',joint,excluded,25,order))
        assert np.array_equal(np.flatnonzero(choose_mask('uniform_e',joint,excluded,25,order)),np.sort(order[:25]))
        control_checks+=4
    report={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'noisy_policy.py'),
        'exact_null_probability_checks':probability_checks,'exact_martingale_expectation':True,
        'independent_decimal_crossing_thresholds':3001,'decimal_precision':70,'pure_control_checks':control_checks,
        'thresholds':thresholds.tolist(),'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/noisy_policy_preflight.json',report)
    print({k:v for k,v in report.items() if k!='thresholds'})


if __name__=='__main__':main()
