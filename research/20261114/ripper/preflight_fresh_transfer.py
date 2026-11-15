"""Synthetic exact rational rounding and LP proof tests before H1bh freeze."""
from pathlib import Path
from fractions import Fraction
import tempfile,time
import numpy as np
from laplace_rate_math import round_ratio_even,rate_costs
from audit_laplace_rate import independent_rate_costs
from degree_prior_math import attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import check_certificate
from fresh_transfer_study import HERE,sha,save


def main():
    started=time.monotonic();states=0
    from transfer_feedback_policy import donor_map,masks,acquire
    from audit_transfer_feedback import reference_map,reference_masks,complete_label_gate
    rng=np.random.default_rng(202611859999);ids=np.arange(4*100).reshape(4,100)
    assert np.array_equal(donor_map(ids,30),reference_map(ids,30))
    invalid=np.tile(np.arange(100),(4,1));assert donor_map(invalid,30) is None and reference_map(invalid,30) is None
    ids[0,:5]=ids[1,:5];assert np.array_equal(donor_map(ids,30),reference_map(ids,30))
    clean=rng.integers(0,9,(95,81),dtype=np.int8)
    noise=dict(uniforms=rng.random(95),permutations=np.stack([rng.permutation(81).astype(np.uint8) for _ in range(95)]))
    feedback=rng.integers(0,12,(81,81),dtype=np.int32);feedback=np.minimum(feedback,feedback.T);np.fill_diagonal(feedback,0)
    original=feedback.copy();permutation=rng.permutation(81);feedback=feedback[np.ix_(permutation,permutation)]
    ii,jj=np.triu_indices(81,1);assert np.array_equal(np.sort(original[ii,jj]),np.sort(feedback[ii,jj]))
    eligible=np.setdiff1d(np.arange(10000,dtype=np.int64),np.arange(3000,dtype=np.int64),assume_unique=True)
    draws=np.stack([np.random.default_rng(908100+seed).choice(eligible,900,replace=False) for seed in range(32)])
    assert np.intersect1d(draws,np.arange(3000)).size==0 and all(len(np.unique(row))==900 for row in draws) and len(np.unique(draws))<draws.size
    observed,events=acquire(feedback,891,clean,noise)
    assert np.array_equal(reference_masks(feedback,891,95,events),observed['given'])
    gates=[complete_label_gate(feedback,891,clean,noise,observed)]
    ii,jj=np.triu_indices(81,1);exposure=observed['given'][:,ii]&observed['given'][:,jj]
    assert np.array_equal(exposure.sum(0),observed['joint_count'])
    assert np.array_equal((exposure&(observed['values'][:,ii]==observed['values'][:,jj])).sum(0),observed['equal_count'])
    donorwarmup=np.zeros_like(observed['given']);donorwarmup[:30]=rng.random((30,81))<.3
    assert int((donorwarmup|observed['given']).sum())==int(donorwarmup.sum()+observed['given'].sum()-(donorwarmup&observed['given']).sum())
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
                unexposed_and_both_tie_parities_checked=True,independent_synthetic_predictable_masks=95,source_disjoint_derangement_checked=True,invalid_assignment_rejected=True,complete_label_substitution_gates=gates,deployment_union_checked=True,new_source_sampling_checked=True,fixed_feedback_histogram_checked=True,rejected_mutated_certificates_or_primals=rejected,cases=records,
                wall_s=time.monotonic()-started,code_sha256={name:sha(HERE/name) for name in
                  ['transfer_feedback_policy.py','soft_feedback_policy.py','audit_soft_feedback.py','prefix_rate_features.py','fresh_transfer_study.py','audit_fresh_transfer.py','audit_transfer_feedback.py','preflight_fresh_transfer.py','run_fresh_transfer.py',
                   'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']})
    save(HERE/'fresh-transfer-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
