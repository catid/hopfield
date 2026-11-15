"""Standalone exact-formula, tie, pure-interface and unchanged-control gates."""
from fractions import Fraction as F
import hashlib
import inspect
import numpy as np
from run import ROOT,HERE,save,sha,now
from exponential_policy import choose_mask,predictive_weights
from spike_policy import choose_mask as control


def main():
    rng=np.random.default_rng(20261206);zero=np.zeros((81,81),np.int32)
    assert list(inspect.signature(choose_mask).parameters)==['arm','joint','equal','k','tie_order']
    values=[]
    for m in range(301):
        joint=np.full((81,81),m,np.int32);equal=zero.copy();weight=predictive_weights(joint,equal)
        u=F(8,9)**m;exact=u
        assert np.isclose(weight[0,1],float(exact),rtol=3e-14,atol=0)
        equal[0,1]=equal[1,0]=1;assert predictive_weights(joint,equal)[0,1]==0
        values.append(weight[0,1])
    assert values[0]==1 and np.all(np.diff(values)<0) and min(values)>0
    initial=0;control_checks=0;pure_checks=0
    for _ in range(64):
        order=rng.permutation(81);mask=choose_mask('exponential',zero,zero,25,order)
        assert np.array_equal(np.flatnonzero(mask),np.sort(order[:25]));initial+=1
        joint=rng.integers(0,301,(81,81),dtype=np.int32);joint=(joint+joint.T)//2
        equal=(rng.random((81,81))<.4).astype(np.int32);equal=np.maximum(equal,equal.T)
        before=hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()
        joint.setflags(write=False);equal.setflags(write=False)
        for arm in ('candidate','spike'):
            assert np.array_equal(choose_mask(arm,joint,equal,25,order),control(arm,joint,equal,25,order));control_checks+=1
        a=choose_mask('exponential',joint,equal,25,order);b=choose_mask('exponential',joint,equal,25,order)
        assert np.array_equal(a,b) and a.sum()==25
        assert hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()==before;pure_checks+=1
    result={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'exponential_policy_sha256':sha(HERE/'exponential_policy.py'),'mixture_policy_sha256':sha(HERE/'spike_policy.py'),
        'original_policy_sha256':sha(HERE/'online_policy.py'),'exact_rational_score_checks':301,
        'relative_tolerance':3e-14,'initial_tie_checks':initial,'unchanged_control_checks':control_checks,
        'read_only_purity_and_determinism_checks':pure_checks,
        'normalization':'Exponential weights start exactly at one, preserving initial ties; fixed decay8/9, zero after any equality.'}
    path=ROOT/'artifacts/research/20261114/ripper/exponential_policy_preflight.json';save(path,result)
    print(result)


if __name__=='__main__':main()
