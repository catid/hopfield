"""Synthetic-only proof/candidate-repair checks before H1av freeze."""
from pathlib import Path
import tempfile,time
import numpy as np
from degree_prior_math import integer_costs,attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import independent_costs,check_certificate
from exclusion_repair_study import HERE,sha,save,degree_deficit


def main():
    started=time.monotonic();records=[];rejected=0
    cases=[('triangle_fractional',3,1,[(0,1),(0,2),(1,2)],[1,1,1],False),
           ('square_integral',4,1,[(0,1),(0,3),(1,2),(2,3)],[-5,1,1,-4],True),
           ('square_tied',4,1,[(0,1),(0,3),(1,2),(2,3)],[0,0,0,0],True),
           ('hard_missing_preferred_edge',4,1,[(0,2),(0,3),(1,2),(1,3),(2,3)],[10,10,10,10,0],True),
           ('full_repair',4,1,[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],[0,10,10,10,10,0],True)]
    with tempfile.TemporaryDirectory(prefix='exclusion-repair-synthetic-') as tmp:
        for index,(name,n,degree,pairs,values,integral) in enumerate(cases):
            edges=np.asarray(pairs,np.int64);ii,jj=edges.T;cost=np.asarray(values,np.int64)
            x,y,solver=solve(n,degree,ii,jj,cost,index,Path(tmp)/f'{name}.log')
            assert solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']
            certificate,a=attempt_certificate(n,degree,ii,jj,cost,x,y)
            assert certificate['certified'] and certificate['integral']==integral
            t,dy,dz=[a[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
            assert check_certificate(n,degree,ii,jj,cost,t,dy,dz)
            for k in range(3):
                mutation=[t.copy(),dy.copy(),dz.copy()];mutation[k][0]+=1
                assert not check_certificate(n,degree,ii,jj,cost,*mutation);rejected+=1
            altered=x.copy();altered[0]+=1e-12
            assert not attempt_certificate(n,degree,ii,jj,cost,altered,y)[0]['certified'];rejected+=1
            if name=='hard_missing_preferred_edge':assert not(x[-1]==1)
            if name=='full_repair':assert np.array_equal(x,np.array([1,0,0,0,0,1]))
            records.append(dict(name=name,certificate=certificate,solver=solver))
    ii,jj=np.triu_indices(4,1);allowed=(ii!=0)&(jj!=0)
    witness=degree_deficit(4,1,ii,jj,allowed)
    assert witness['vertex']==0 and witness['available_neighbors']==0 and witness['required_degree']==1
    assert degree_deficit(4,1,ii,jj,np.ones(6,bool)) is None
    m=np.arange(1111,dtype=np.int64);c=(m*11+1)%(m+1)
    costs=integer_costs(m,c);independent,margin=independent_costs(m,c);assert np.array_equal(costs,independent)
    result=dict(status='validated_synthetic_only',cases=records,rejected_mutated_certificates_or_primals=rejected,
                degree_deficit_witness_checked=True,decimal_cost_states_checked=1111,minimum_cost_rounding_margin=margin,
                wall_s=time.monotonic()-started,code_sha256={name:sha(HERE/name) for name in
                  ['exclusion_repair_study.py','audit_exclusion_repair.py','preflight_exclusion_repair.py',
                   'degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']})
    save(HERE/'exclusion-repair-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
