"""Synthetic exact rational rounding and LP proof tests before H1bc freeze."""
from pathlib import Path
from fractions import Fraction
import tempfile,time
import numpy as np
from laplace_rate_math import round_ratio_even,rate_costs
from audit_laplace_rate import independent_rate_costs
from degree_prior_math import attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import check_certificate
from soft_feedback_study import HERE,sha,save


def main():
    started=time.monotonic();states=0
    from soft_feedback_policy import collect,choose
    from audit_soft_feedback import reference_replay,reference_choose,suffix_gate
    rng=np.random.default_rng(202611809999);clean=rng.integers(0,9,(95,81),dtype=np.int8)
    noise=dict(uniforms=rng.random(95),permutations=np.stack([rng.permutation(81).astype(np.uint8) for _ in range(95)]))
    permutation=rng.permutation(81);gates=[]
    for perm in [None,permutation]:
        observed,events=collect(clean,noise['uniforms'],noise['permutations'],891,perm,45)
        expected=reference_replay(clean,noise,891,perm,45,observed,events)
        assert all(np.array_equal(expected[k],observed[k]) for k in expected)
        gates.append(suffix_gate(clean,noise,891,perm,45,observed))
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);order=rng.permutation(81)
    assert np.array_equal(choose(joint,equal,order),choose(joint,equal,order,permutation))
    for _ in range(100):
        joint=rng.integers(0,100,(81,81),dtype=np.int32);joint=np.minimum(joint,joint.T)
        equal=rng.integers(0,10,(81,81),dtype=np.int32);equal=np.minimum(np.minimum(equal,equal.T),joint);order=rng.permutation(81)
        for perm in [None,permutation]:assert np.array_equal(choose(joint,equal,order,perm),reference_choose(joint,equal,order,perm))

    for denominator in range(1,113):
        for numerator in range(401):
            assert round_ratio_even(numerator,denominator)==round(Fraction(numerator,denominator));states+=1
    m=np.arange(1111,dtype=np.int64);c=(13*m+7)%(m+1)
    assert np.array_equal(rate_costs(m,c),independent_rate_costs(m,c))
    assert rate_costs(np.array([0,126,126]),np.array([0,0,2])).tolist()==[500000,7812,23438]
    cases=[('triangle_fractional',3,1,[(0,1),(0,2),(1,2)],[1,1,1],False),
           ('square_integral',4,1,[(0,1),(0,3),(1,2),(2,3)],[1,10,10,2],True),
           ('square_tied',4,1,[(0,1),(0,3),(1,2),(2,3)],[0,0,0,0],True)]
    records=[];rejected=0
    with tempfile.TemporaryDirectory(prefix='laplace-rate-synthetic-') as tmp:
        for index,(name,n,degree,pairs,cost,integral) in enumerate(cases):
            edges=np.array(pairs,np.int64);ii,jj=edges.T;cost=np.array(cost,np.int64)
            x,y,meta=solve(n,degree,ii,jj,cost,index,Path(tmp)/f'{name}.log')
            assert meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']
            cert,a=attempt_certificate(n,degree,ii,jj,cost,x,y);assert cert['certified'] and cert['integral']==integral
            t,dy,dz=[a[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
            assert check_certificate(n,degree,ii,jj,cost,t,dy,dz)
            for k in range(3):
                changed=[t.copy(),dy.copy(),dz.copy()];changed[k][0]+=1
                assert not check_certificate(n,degree,ii,jj,cost,*changed);rejected+=1
            changed=x.copy();changed[0]+=1e-12;assert not attempt_certificate(n,degree,ii,jj,cost,changed,y)[0]['certified'];rejected+=1
            records.append(dict(name=name,certificate=cert,solver=meta))
    result=dict(status='validated_synthetic_only',rational_rounding_states_checked=states,rate_cost_states_checked=1114,
                unexposed_and_both_tie_parities_checked=True,independent_synthetic_predictable_masks=190,synthetic_policy_states=200,suffix_substitution_gates=gates,rejected_mutated_certificates_or_primals=rejected,cases=records,
                wall_s=time.monotonic()-started,code_sha256={name:sha(HERE/name) for name in
                  ['soft_feedback_policy.py','soft_feedback_study.py','audit_soft_feedback.py','preflight_soft_feedback.py','run_soft_feedback.py',
                   'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']})
    save(HERE/'soft-feedback-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
