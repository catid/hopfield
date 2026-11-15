"""New M1000 interface endpoints and N2300 crossing tables, before outcomes."""
from decimal import Decimal,localcontext
from fractions import Fraction as F
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from equal_budget_estimator import fit_indicators
from calibration_estimator import fit_count
from equal_budget_policy import choose_mask,calibration,crossing_thresholds
from noisy_policy import choose_mask as original


def main():
    start=time.monotonic();cases=[]
    for k in [0,1,5,10,15,20,25,40,100,500,850,880,888,890,999,1000]:
        r=fit_indicators([1]*k+[0]*(1000-k),1000);assert r==fit_count(k,1000)
        if not r['no_pruning_fallback']:
            assert Decimal(r['endpoint_cdf_decimal80'])<=Decimal('.005')<Decimal(r['previous_grid_cdf_decimal80'])
        else:assert F(r['eta_upper_numerator'],r['eta_upper_denominator'])==F(8,9)
        cases.append(r)
    assert fit_indicators([1]*199+[0]*9801,10000)==fit_count(199,10000)
    checks=0
    for eta in [F(1,100),F(1,50),F(11973,1000000),F(8,9)]:
        table=crossing_thresholds(eta,2300);q0,a,b,fallback=calibration(eta)
        with localcontext() as context:
            context.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
            for m,c in enumerate(table):
                c=int(c)
                if c<=m:assert c*la+(m-c)*lb>=bound
                if c>0:assert (c-1)*la+(m-c+1)*lb<bound
                checks+=1
    rng=np.random.default_rng(20303000);masks=0
    for _ in range(32):
        joint=rng.integers(0,2301,(81,81),dtype=np.int32);joint=(joint+joint.T)//2
        excluded=rng.random((81,81))<.5;excluded=excluded|excluded.T;order=rng.permutation(81)
        joint.setflags(write=False);excluded.setflags(write=False);expected=original('exponential_e',joint,excluded,25,order)
        for arm in ['exact','fixed_upper','learned_small','learned_large']:assert np.array_equal(choose_mask(arm,joint,excluded,25,order),expected);masks+=1
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'equal_budget_policy.py'),
       'estimator_sha256':sha(HERE/'equal_budget_estimator.py'),'base_estimator_sha256':sha(HERE/'calibration_estimator.py'),
       'base_policy_sha256':sha(HERE/'calibrated_policy.py'),'endpoint_cases':cases,'endpoint_checks':17,'decimal_threshold_checks':checks,
       'unchanged_readonly_mask_checks':masks,'fallback_checked':True,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/equal_budget_preflight.json',r);print({k:v for k,v in r.items() if k!='endpoint_cases'})


if __name__=='__main__':main()
