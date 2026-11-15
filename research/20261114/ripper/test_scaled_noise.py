"""No-outcome gates at declared1%/5%rates and conservative calibrations."""
from decimal import Decimal,localcontext
from fractions import Fraction as F
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from rate_transfer_estimator import fit_indicators
from calibration_estimator import fit_count
from scaled_noise_policy import choose_mask,calibration,crossing_thresholds
from noisy_policy import choose_mask as original


def main():
    start=time.monotonic();cases=[];checks=0
    for k in [0,1,20,50,95,100,150,999,1000]:
        r=fit_indicators([1]*k+[0]*(1000-k));assert r==fit_count(k,1000);cases.append(r)
        if not r['no_pruning_fallback']:assert Decimal(r['endpoint_cdf_decimal80'])<=Decimal('.005')<Decimal(r['previous_grid_cdf_decimal80'])
    for eta in [F(1,100),F(1,20),F(3,50),F(2,25),F(8,9)]:
        t=crossing_thresholds(eta,4440);q0,a,b,fallback=calibration(eta)
        with localcontext() as context:
            context.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
            for m,c in enumerate(t):
                c=int(c)
                if c<=m:assert c*la+(m-c)*lb>=bound
                if c>0:assert (c-1)*la+(m-c+1)*lb<bound
                checks+=1
    rng=np.random.default_rng(20332000);masks=0
    for _ in range(32):
        j=rng.integers(0,4441,(81,81),dtype=np.int32);j=(j+j.T)//2;e=rng.random((81,81))<.5;e=e|e.T;order=rng.permutation(81)
        j.setflags(write=False);e.setflags(write=False);expected=original('exponential_e',j,e,25,order)
        for a in ['exact_short','learned_short','exact_long','learned_long']:assert np.array_equal(choose_mask(a,j,e,25,order),expected);masks+=1
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'scaled_noise_policy.py'),
       'estimator_sha256':sha(HERE/'rate_transfer_estimator.py'),'base_policy_sha256':sha(HERE/'calibrated_policy.py'),'base_estimator_sha256':sha(HERE/'calibration_estimator.py'),
       'endpoint_cases':cases,'endpoint_checks':9,'decimal_threshold_checks':checks,'unchanged_readonly_mask_checks':masks,'elapsed_s':time.monotonic()-start}
    save(ROOT/'artifacts/research/20261114/ripper/scaled_noise_preflight.json',r);print({k:v for k,v in r.items() if k!='endpoint_cases'})


if __name__=='__main__':main()
