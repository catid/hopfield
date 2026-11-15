"""Verify identical resamples and measure the audit-only optimization."""
import time
import numpy as np
from bootstrap_two_way import resample_means
from run import ROOT,HERE,save,sha,now


def main():
    diff=np.random.default_rng(818).integers(-1,2,(64,10000),dtype=np.int8)
    seed=20261301;draws=256;rng=np.random.default_rng(seed)
    t=time.perf_counter()
    expected=np.array([100*diff[rng.integers(0,64,64)][:,rng.integers(0,10000,10000)].mean() for _ in range(draws)])
    old=time.perf_counter()-t;t=time.perf_counter();actual=resample_means(diff,seed,draws);new=time.perf_counter()-t
    assert np.allclose(actual,expected,rtol=0,atol=1e-13)
    assert np.array_equal(np.rint(actual*6400),np.rint(expected*6400))
    for value in [-1,0,1]:assert np.all(resample_means(np.full((3,7),value),seed,17,batch=4)==100*value)
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'helper_sha256':sha(HERE/'bootstrap_two_way.py'),
        'matched_resamples':draws,'max_absolute_difference_pp':float(np.max(np.abs(actual-expected))),
        'exact_integer_totals_identical':True,'boundary_checks':51,'reference_seconds':old,'optimized_seconds':new,
        'speedup':old/new,'numpy':np.__version__,'scope':'Audit-only reformulation, same RNG row/puzzle draws and exact integer totals; no training or search changes.'}
    save(ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json',r);print(r)


if __name__=='__main__':main()
