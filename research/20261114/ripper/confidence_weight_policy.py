"""Exact postfit LR sign and label-independent placement control."""
from fractions import Fraction
from functools import lru_cache
import numpy as np
from calibrated_policy import calibration


@lru_cache(None)
def strong_pair(eta,m,c):
    assert isinstance(eta,Fraction) and 0<=c<=m
    _,a,b,fallback=calibration(eta)
    if fallback:return True
    numerator=pow(a.numerator,c)*pow(b.numerator,m-c)
    denominator=pow(a.denominator,c)*pow(b.denominator,m-c)
    return numerator<=denominator


def weights(support,joint,equal,eta,pair_permutation):
    assert support.shape==joint.shape==equal.shape==(3240,) and support.dtype==bool
    assert np.all((equal>=0)&(equal<=joint))
    assert np.array_equal(np.sort(pair_permutation),np.arange(3240))
    uniform=np.where(support,16,0).astype(np.uint8)
    evidence=uniform.copy()
    for k in np.flatnonzero(support):
        if not strong_pair(eta,int(joint[k]),int(equal[k])):evidence[k]=1
    # Only the number of weak weights is read. Their observed pair identities
    # never affect the independent full-pair ordering used by this control.
    weak_count=int((evidence==1).sum())
    ordered_retained=pair_permutation[support[pair_permutation]]
    shuffled=uniform.copy();shuffled[ordered_retained[:weak_count]]=1
    for value in (uniform,evidence,shuffled):assert np.array_equal(value>0,support)
    assert np.array_equal(np.sort(evidence),np.sort(shuffled))
    return {'uniform':uniform,'evidence':evidence,'shuffled':shuffled}
