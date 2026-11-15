"""Exact sign, shuffled placement, read-only and boundary preflight."""
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import inspect
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from confidence_weight_policy import strong_pair,weights


def main():
    start=time.monotonic();rng=np.random.default_rng(20403000);checks=0;cases=0
    assert list(inspect.signature(weights).parameters)==['support','joint','equal','eta','pair_permutation']
    for eta in [Fraction(1,20),Fraction(64095,1000000),Fraction(8,9)]:
        q0=eta/4-9*eta**2/64;alt=Fraction(1,9)
        for _ in range(128):
            m=int(rng.integers(0,2221));c=int(rng.integers(0,m+1))
            exact=(alt/q0)**c*((1-alt)/(1-q0))**(m-c)
            assert strong_pair(eta,m,c)==(exact<=1);checks+=1
        assert strong_pair(eta,0,0)
        for index in range(16):
            support=rng.random(3240)<.3;m=rng.integers(0,401,3240);c=np.array([rng.integers(0,int(x)+1) for x in m]);order=rng.permutation(3240)
            if index==0:support[:]=False
            if index==1:support[:]=True;m[:]=0;c[:]=0
            for x in (support,m,c,order):x.setflags(write=False)
            before=hashlib.sha256(support.tobytes()+m.tobytes()+c.tobytes()+order.tobytes()).hexdigest();out=weights(support,m,c,eta,order)
            expected=out['uniform'].copy()
            if eta<Fraction(8,9):
                with localcontext() as ctx:
                    ctx.prec=80;a=alt/q0;b=(1-alt)/(1-q0);la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln()
                    for k in np.flatnonzero(support):expected[k]=16 if int(c[k])*la+(int(m[k])-int(c[k]))*lb<=0 else 1
            assert np.array_equal(out['evidence'],expected)
            retained=[int(k) for k in order if support[k]];expected=out['uniform'].copy();expected[retained[:int((out['evidence']==1).sum())]]=1
            assert np.array_equal(out['shuffled'],expected)
            assert before==hashlib.sha256(support.tobytes()+m.tobytes()+c.tobytes()+order.tobytes()).hexdigest();cases+=1
    report={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'confidence_weight_policy.py'),
        'independent_fraction_sign_checks':checks,'independent_decimal_weight_and_shuffle_cases':cases,'readonly_cases':cases,
        'equality_at_LR1_is_strong':True,'fallback_all_strong':True,'weight_numerators_over16':[0,1,16],'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/confidence_weight_preflight.json',report);print(report)


if __name__=='__main__':main()
