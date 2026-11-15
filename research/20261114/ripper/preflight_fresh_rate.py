"""Synthetic exact rational rounding and LP proof tests before H1az freeze."""
from pathlib import Path
from fractions import Fraction
import tempfile,time
import numpy as np
from laplace_rate_math import round_ratio_even,rate_costs
from audit_laplace_rate import independent_rate_costs
from degree_prior_math import attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import check_certificate
from fresh_rate_study import HERE,sha,save


def main():
    started=time.monotonic();states=0
    from permuted_noise_policy import permutation_draw,apply_permutation
    from audit_fresh_rate_acquisition import independent_draw
    producer=np.random.default_rng(202611780000);auditor=np.random.default_rng(202611780000);clean=np.repeat(np.arange(9,dtype=np.int8),9)
    for _ in range(512):
        u,permutation=permutation_draw(producer);v,reference,event=independent_draw(auditor)
        assert u==v and np.array_equal(permutation,reference)
        assert np.array_equal(apply_permutation(clean,u,permutation),clean[reference] if event else clean)
    # Verify new collectors are exactly the inherited recipes after declared N900 edits.
    inherited=(HERE/'permuted_noise_acquire.py').read_text().replace('permuted_noise_common','fresh_rate_common').replace('1110','900').replace('29750','24500')
    inherited=inherited.replace('Position-permuted query acquisition','H1az fresh source position-permuted query acquisition')
    assert inherited==(HERE/'fresh_rate_acquire.py').read_text()
    inherited=(HERE/'audit_permuted_noise_acquisition.py').read_text().replace('permuted_noise_common','fresh_rate_common').replace('permuted_noise_acquire.py','fresh_rate_acquire.py').replace('1110','900').replace('1111','901').replace('29750','24500').replace('27750','22500').replace('3596400','2916000')
    inherited=inherited.replace("    z=np.load(d/'calibration_observations.npz');cal=", "    with np.load(d/'calibration_observations.npz',allow_pickle=False) as archive:z={key:archive[key].copy() for key in archive.files}\n    cal=")
    inherited=inherited.replace('Independent position-permutation law','H1az independent position-permutation law')
    assert inherited==(HERE/'audit_fresh_rate_acquisition.py').read_text()

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
                unexposed_and_both_tie_parities_checked=True,independent_permutation_draws=512,inherited_collector_and_auditor_lineage_verified=True,rejected_mutated_certificates_or_primals=rejected,cases=records,
                wall_s=time.monotonic()-started,code_sha256={name:sha(HERE/name) for name in
                  ['fresh_rate_study.py','fresh_rate_common.py','fresh_rate_acquire.py','audit_fresh_rate_acquisition.py','audit_fresh_rate.py','preflight_fresh_rate.py','run_fresh_rate.py',
                   'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']})
    save(HERE/'fresh-rate-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
