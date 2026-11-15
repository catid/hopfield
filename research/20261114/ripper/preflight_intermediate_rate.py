"""Synthetic-only prefix isolation and exact LP certificate gates for H1ay."""
from pathlib import Path
from fractions import Fraction
import tempfile,time
import numpy as np
from prefix_rate_features import prefix_counts
from audit_intermediate_rate import recount,suffix_gate
from laplace_rate_math import round_ratio_even,rate_costs
from audit_laplace_rate import independent_rate_costs
from degree_prior_math import attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import check_certificate
from intermediate_rate_study import HERE,sha,save


def main():
    start=time.monotonic();rng=np.random.default_rng(2026111509);given=np.zeros((1110,81),bool);values=np.full((1110,81),-1,np.int8)
    for row in range(1110):
        cells=rng.choice(81,25,replace=False);given[row,cells]=True;values[row,cells]=rng.integers(0,9,25,dtype=np.int8)
    checkpoints=[750,900,1000];actual=prefix_counts(given,values,checkpoints);independent=recount(given,values,checkpoints)
    pair={tuple(ij):e for e,ij in enumerate(zip(*np.triu_indices(81,1),strict=True))};joint=np.zeros(3240,np.int64);equal=joint.copy();literal={}
    for row in range(1000):
        cells=np.flatnonzero(given[row])
        for offset,i in enumerate(cells):
            for j in cells[offset+1:]:
                edge=pair[(i,j)];joint[edge]+=1;equal[edge]+=int(values[row,i]==values[row,j])
        if row+1 in checkpoints:literal[row+1]=(joint.copy(),equal.copy())
    for n in checkpoints:
        for got,reference,explicit in zip(actual[n],independent[n],literal[n],strict=True):
            assert np.array_equal(got,reference) and np.array_equal(got,explicit)
        for other in checkpoints:
            if other!=n:
                assert not any(np.shares_memory(a,b) for a in actual[n] for b in actual[other])
    gates=[suffix_gate(given,values,n,independent[n]) for n in checkpoints]
    states=0
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
    with tempfile.TemporaryDirectory(prefix='intermediate-rate-synthetic-') as tmp:
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
    result=dict(status='validated_synthetic_only',literal_pair_updates=1000*300,prefix_rows=checkpoints,snapshot_nonaliasing=True,
                suffix_substitution_gates=gates,rational_rounding_states_checked=states,rate_cost_states_checked=1114,
                unexposed_and_both_tie_parities_checked=True,rejected_mutated_certificates_or_primals=rejected,cases=records,
                wall_s=time.monotonic()-start,code_sha256={name:sha(HERE/name) for name in
                  ['prefix_rate_features.py','intermediate_rate_study.py','audit_intermediate_rate.py','preflight_intermediate_rate.py','run_intermediate_rate.py',
                   'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']})
    save(HERE/'intermediate-rate-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
