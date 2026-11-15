"""Independent finite-sample coverage and exact/Decimal endpoint gates."""
from fractions import Fraction as F
from decimal import Decimal,localcontext
import math
import time
import numpy as np
from calibration_estimator import fit_count,disagreement_probability,exact_cdf_comparison
from calibrated_policy import calibration,crossing_thresholds,choose_mask
from noisy_policy import choose_mask as original
from run import ROOT,HERE,sha,save,now


def main():
    start=time.perf_counter();cases=[]
    for k in [0,1,5,50,100,150,180,190,199,200,210,250,400,1000,5000,8500,8800,8888,8900,9000,9999,10000]:
        r=fit_count(k);eta=F(r['eta_upper_numerator'],r['eta_upper_denominator'])
        assert eta<=F(8,9)
        if not r['no_pruning_fallback']:
            assert Decimal(r['endpoint_cdf_decimal80'])<=Decimal('.005')
            assert Decimal(r['previous_grid_cdf_decimal80'])>Decimal('.005')
            assert exact_cdf_comparison(k,10000,disagreement_probability(eta))<=0
        else:assert eta==F(8,9)
        cases.append(r)
    probability_checks=0
    for eta in [F(0),F(1,100),F(1,10),F(1,2),F(22,25),F(8,9)]:
        # Literal enumeration of two independently corrupted labels for clean digit zero.
        marg=[1-eta]+[eta/8]*8
        exact=sum(marg[i]*marg[j] for i in range(9) for j in range(9) if i!=j)
        assert exact==disagreement_probability(eta);probability_checks+=1
        # Enumerate all outcomes of a small fixed-n calibration experiment exactly.
        n=20;d=disagreement_probability(eta);failure=F(0)
        for k in range(n+1):
            r=fit_count(k,n,1000);bound=F(r['eta_upper_numerator'],r['eta_upper_denominator'])
            if bound<eta:failure+=math.comb(n,k)*d**k*(1-d)**(n-k)
        assert failure<=F(1,200);probability_checks+=1
    threshold_checks=0
    for eta in [F(1,100),F(1,50),F(11973,1000000),F(12579,1000000),F(8,9)]:
        table=crossing_thresholds(eta,1500);q0,a,b,fallback=calibration(eta)
        if fallback:assert np.array_equal(table,np.arange(1,1502));threshold_checks+=1501;continue
        with localcontext() as ctx:
            ctx.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();boundary=Decimal(648000).ln()
            for m,c in enumerate(table):
                c=int(c)
                if c<=m:assert c*la+(m-c)*lb>=boundary
                if c>0:assert (c-1)*la+(m-c+1)*lb<boundary
                threshold_checks+=1
    rng=np.random.default_rng(20276001);mask_checks=0
    for _ in range(64):
        joint=rng.integers(0,1501,(81,81),dtype=np.int32);joint=(joint+joint.T)//2
        excluded=rng.random((81,81))<.5;excluded=excluded|excluded.T;order=rng.permutation(81)
        joint.setflags(write=False);excluded.setflags(write=False);expected=original('exponential_e',joint,excluded,25,order)
        for arm in ['exact','learned','fixed_upper']:
            assert np.array_equal(choose_mask(arm,joint,excluded,25,order),expected);mask_checks+=1
    report={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'estimator_sha256':sha(HERE/'calibration_estimator.py'),
        'policy_sha256':sha(HERE/'calibrated_policy.py'),'known_rate_probability_and_coverage_checks':probability_checks,
        'calibration_endpoint_cases':cases,'calibration_endpoint_checks':len(cases),'integer_and_decimal_threshold_checks':threshold_checks,
        'unchanged_readonly_mask_checks':mask_checks,'fallback_tested':True,'elapsed_s':time.perf_counter()-start,'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/calibrated_preflight.json',report)
    print({k:v for k,v in report.items() if k!='calibration_endpoint_cases'})


if __name__=='__main__':main()
