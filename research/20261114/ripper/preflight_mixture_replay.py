"""Synthetic-only independent checks before freezing H1at."""
from fractions import Fraction
import math
import time
import numpy as np
from mixture_replay_policy import exact_crossing,mixture_log,thresholds,counts_step
from audit_mixture_replay import independent_logs
from mixture_replay_study import HERE,sha,save


def main():
    started=time.monotonic();rng=np.random.default_rng(2026111501);checks=0;max_error=0.
    null=Fraction(9,1600);qs=[Fraction(1,36),Fraction(1,18),Fraction(1,9),Fraction(2,9)]
    for m in range(45):
        for c in range(m+1):
            lr=[(q/null)**c*((1-q)/(1-null))**(m-c) for q in qs]
            exact=sum(lr)/4
            assert exact_crossing(m,c)==(exact>=648000)
            assert exact_crossing(m,c,False)==(lr[2]>=648000)
            observed=float(mixture_log(m,c));expected=math.log(float(exact))
            assert abs(observed-expected)<1e-12
            max_error=max(max_error,abs(observed-expected));checks+=1
    cuts=thresholds(44)
    for m in range(45):
        for arm in range(2):
            direct=[c for c in range(m+1) if exact_crossing(m,c,arm==1)]
            assert cuts[arm,m]==(direct[0] if direct else m+1)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint)
    ii,jj=np.triu_indices(81,1);m=np.zeros(3240,np.int64);c=np.zeros_like(m)
    for row in range(11):
        mask=np.zeros(81,bool);mask[rng.choice(81,25,replace=False)]=True
        values=np.full(81,-1,np.int8);values[mask]=rng.integers(0,9,25)
        counts_step(joint,equal,mask,values);queried=mask[ii]&mask[jj]
        m+=queried;c+=queried&(values[ii]==values[jj])
        assert np.array_equal(joint[ii,jj],m) and np.array_equal(equal[ii,jj],c)
        _,logs=independent_logs(m,c);assert np.allclose(logs,mixture_log(m,c),rtol=0,atol=1e-12)
    # A likelihood can recede after crossing; first-crossing decisions stay removed.
    sequence=[True]*6+[False]*100;m=c=0;first=None
    for row,same in enumerate(sequence,1):
        m+=1;c+=same
        if first is None and exact_crossing(m,c):first=row
    assert first is not None and not exact_crossing(m,c)
    result=dict(status='validated_synthetic_only',policy_sha256=sha(HERE/'mixture_replay_policy.py'),
                preflight_sha256=sha(__file__),exact_rational_states_checked=checks,thresholds_checked=90,
                synthetic_pair_states_checked=11*3240,maximum_log_error=max_error,
                permanent_crossing_then_recession_checked=True,wall_s=time.monotonic()-started)
    save(HERE/'mixture-replay-preflight.json',result);print(result)


if __name__=='__main__':main()
