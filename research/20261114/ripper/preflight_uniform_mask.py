"""Synthetic exact rational rounding and LP proof tests before H1ba freeze."""
from pathlib import Path
from fractions import Fraction
import tempfile,time
import numpy as np
from laplace_rate_math import round_ratio_even,rate_costs
from audit_laplace_rate import independent_rate_costs
from degree_prior_math import attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import check_certificate
from uniform_mask_study import HERE,sha,save


def main():
    started=time.monotonic();states=0
    from uniform_mask_features import masks
    from audit_uniform_mask import reference_masks
    from prefix_rate_features import prefix_counts
    for seed in [202611790999,401,53]:
        a=masks(seed);b=reference_masks(seed);assert np.array_equal(a,b) and np.all(a.sum(1)==25)
    given=masks(882);rng=np.random.default_rng(721);values=np.where(given,rng.integers(0,9,given.shape),-1).astype(np.int8)
    m,c=prefix_counts(given,values,[900])[900];ii,jj=np.triu_indices(81,1)
    exposed=given[:,ii]&given[:,jj];assert np.array_equal(m,exposed.sum(0)) and np.array_equal(c,(exposed&(values[:,ii]==values[:,jj])).sum(0))
    adaptive=masks(883);reused=given&adaptive;new=given&~adaptive
    assert int(reused.sum())+int(new.sum())==22500 and not (new&reused).any()

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
                unexposed_and_both_tie_parities_checked=True,independent_synthetic_mask_rows=2700,synthetic_pair_states_checked=900*3240,overlap_accounting_checked=True,rejected_mutated_certificates_or_primals=rejected,cases=records,
                wall_s=time.monotonic()-started,code_sha256={name:sha(HERE/name) for name in
                  ['uniform_mask_features.py','uniform_mask_study.py','audit_uniform_mask.py','preflight_uniform_mask.py','run_uniform_mask.py','prefix_rate_features.py',
                   'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']})
    save(HERE/'uniform-mask-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
