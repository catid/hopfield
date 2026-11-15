"""Independent Decimal and Fraction crossings, fallback, and mask contract."""
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import inspect
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from mixture_law_policy import components,crossing_thresholds,log_evidence,choose_mask
from calibrated_policy import crossing_thresholds as fixed_thresholds
from fast_reference_mask import reference_mask


def main():
    start=time.monotonic();rng=np.random.default_rng(20383000);endpoints=0;fraction_checks=0;log_checks=0
    assert list(inspect.signature(choose_mask).parameters)==['arm','joint','excluded','k','tie_order']
    for eta in [Fraction(1,100),Fraction(1,20),Fraction(75419,1000000),Fraction(3,20),Fraction(8,9)]:
        q0=eta/4-9*eta**2/64;table=crossing_thresholds(eta,2220,True)
        assert np.array_equal(crossing_thresholds(eta,2220,False),fixed_thresholds(eta,2220))
        assert not table.flags.writeable
        if eta==Fraction(8,9):
            assert np.array_equal(table,np.arange(1,2222))
            m=np.arange(2221);assert not log_evidence(m,m,eta,True).any();continue
        with localcontext() as ctx:
            ctx.prec=80;logs=[]
            for alt in [Fraction(1,9),Fraction(1,6)]:
                a=alt/q0;b=(1-alt)/(1-q0)
                logs.append(((Decimal(a.numerator)/a.denominator).ln(),(Decimal(b.numerator)/b.denominator).ln()))
            for m,c in enumerate(table):
                c=int(c)
                def value(k):return sum((k*la+(m-k)*lb).exp() for la,lb in logs)/2
                if c<=m:assert value(c)>=648000
                if c>0:assert value(c-1)<648000
                endpoints+=1
                if m%37==0:
                    for k in sorted(set([max(0,min(m,c)),max(0,c-1)])):
                        exact=sum((alt/q0)**k*((1-alt)/(1-q0))**(m-k) for alt in [Fraction(1,9),Fraction(1,6)])/2
                        assert (exact>=648000)==(k>=c);fraction_checks+=1
                        observed=float(log_evidence(np.array([m]),np.array([k]),eta,True)[0])
                        assert abs(observed-float(value(k).ln()))<2e-11;log_checks+=1
    masks=0
    for trial in range(384):
        joint=rng.integers(0,2221,(81,81));joint=(joint+joint.T)//2
        excluded=rng.random((81,81))<.35;excluded|=excluded.T;order=rng.permutation(81)
        if trial%32==0:joint[:]=0;excluded[:]=False
        if trial%32==1:excluded[:]=True
        joint.setflags(write=False);excluded.setflags(write=False)
        before=hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
        expected=reference_mask('exposure_exact',joint,excluded,order)
        for arm in ['fixed_exact','mixture_exact','fixed_learned','mixture_learned']:
            assert np.array_equal(choose_mask(arm,joint,excluded,25,order),expected);masks+=1
        assert before==hashlib.sha256(joint.tobytes()+excluded.tobytes()).hexdigest()
    report={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'mixture_law_policy.py'),
        'decimal80_mixture_thresholds':endpoints,'independent_fraction_crossings':fraction_checks,'decimal80_log_values':log_checks,
        'fixed_threshold_parity_rates':5,'fallback_no_pruning_verified':True,'independent_mask_comparisons':masks,
        'readonly_history_cases':384,'selector_signature_past_only':True,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/mixture_law_preflight.json',report);print(report)


if __name__=='__main__':main()
