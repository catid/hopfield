"""Check supermartingale expectations, thresholds and unchanged mask policy."""
from decimal import Decimal,localcontext
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save,now
from noise_bound_policy import ASSUMED,BOUND,calibration,crossing_thresholds,choose_mask
from noisy_policy import choose_mask as prior_mask,crossing_thresholds as prior_thresholds


def main():
    true_eta=Fraction(1,100);true_q=true_eta/4-9*true_eta**2/64
    thresholds={};expectations={};decimal_checks=0
    for arm in ASSUMED:
        q0,a,b=calibration(arm);expectation=true_q*a+(1-true_q)*b
        if arm=='exact':assert expectation==1
        elif arm=='upper':assert expectation<1
        else:assert expectation>1
        expectations[arm]={'numerator':expectation.numerator,'denominator':expectation.denominator,'value':float(expectation)}
        thresholds[arm]=crossing_thresholds(arm,1500)
        with localcontext() as ctx:
            ctx.prec=70;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();boundary=Decimal(BOUND).ln()
            for m,c in enumerate(thresholds[arm]):
                c=int(c)
                if c<=m:assert c*la+(m-c)*lb>=boundary
                if c>0:assert (c-1)*la+(m-c+1)*lb<boundary
                decimal_checks+=1
    assert np.array_equal(thresholds['exact'],prior_thresholds(1500))
    rng=np.random.default_rng(20271000);checks=0
    for _ in range(64):
        joint=rng.integers(0,1501,(81,81),dtype=np.int32);joint=(joint+joint.T)//2
        excluded=rng.random((81,81))<.5;excluded=excluded|excluded.T;order=rng.permutation(81)
        joint.setflags(write=False);excluded.setflags(write=False)
        expected=prior_mask('exponential_e',joint,excluded,25,order)
        for arm in ASSUMED:
            assert np.array_equal(choose_mask(arm,joint,excluded,25,order),expected);checks+=1
    result={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'noise_bound_policy.py'),
        'original_noisy_policy_sha256':sha(HERE/'noisy_policy.py'),'conditional_null_expectations':expectations,
        'decimal_threshold_checks':decimal_checks,'decimal_precision':70,'unchanged_mask_checks':checks,
        'exact_control_thresholds_unchanged':True,'thresholds':{k:v.tolist() for k,v in thresholds.items()},'numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/noise_bound_preflight.json',result)
    print({k:v for k,v in result.items() if k!='thresholds'})


if __name__=='__main__':main()
