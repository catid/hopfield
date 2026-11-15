"""Synthetic LPs, exact proof mutations and cost arithmetic before H1au freeze."""
from pathlib import Path
import tempfile,time
import numpy as np
from degree_prior_math import integer_costs,attempt_certificate
from degree_prior_solver import solve
from audit_degree_prior import check_certificate,independent_costs
from degree_prior_study import HERE,sha,save


def main():
    started=time.monotonic();records=[];mutations=0
    cases=[('triangle_fractional',3,1,[(0,1),(0,2),(1,2)],[1,1,1],False),
           ('square_integral',4,1,[(0,1),(0,3),(1,2),(2,3)],[-5,1,1,-4],True),
           ('square_tied',4,1,[(0,1),(0,3),(1,2),(2,3)],[0,0,0,0],True)]
    ii,jj=np.triu_indices(81,1);distance=np.minimum(jj-ii,81-(jj-ii));base=distance<=10
    extra=np.flatnonzero(~base)[:64];candidate=base.copy();candidate[extra]=True
    pairs=list(zip(ii[candidate].tolist(),jj[candidate].tolist()));cost=np.where(base[candidate],-1000000,1000000)
    cases.append(('synthetic_degree20',81,20,pairs,cost.tolist(),True))
    with tempfile.TemporaryDirectory(prefix='degree-prior-synthetic-') as tmp:
        for number,(name,n,degree,pairs,cost,integral) in enumerate(cases):
            edges=np.asarray(pairs,np.int64);left,right=edges.T;cost=np.array(cost,np.int64)
            x,y,meta=solve(n,degree,left,right,cost,number,Path(tmp)/f'{name}.log')
            assert meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']
            certificate,arrays=attempt_certificate(n,degree,left,right,cost,x,y)
            assert certificate['certified'] and certificate['integral']==integral
            t,dy,dz=[arrays[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
            assert check_certificate(n,degree,left,right,cost,t,dy,dz)
            for which in range(3):
                changed=[t.copy(),dy.copy(),dz.copy()];changed[which][0]+=1
                assert not check_certificate(n,degree,left,right,cost,*changed);mutations+=1
            perturbed=x.copy();perturbed[0]+=1e-12
            rejected,_=attempt_certificate(n,degree,left,right,cost,perturbed,y)
            assert not rejected['certified'] and rejected['reason']=='primal_not_literal_half_integral';mutations+=1
            records.append(dict(name=name,integral=integral,variables=len(cost),solver=meta,certificate=certificate))
    m=np.arange(1111,dtype=np.int64);c=(m*7+3)%(m+1)
    produced=integer_costs(m,c);independent,margin=independent_costs(m,c);assert np.array_equal(produced,independent)
    result=dict(status='validated_synthetic_only',cases=records,rejected_mutated_certificates_or_primal_outputs=mutations,
                decimal_cost_states_checked=1111,minimum_cost_rounding_margin=margin,wall_s=time.monotonic()-started,
                code_sha256={name:sha(HERE/name) for name in ['degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py','preflight_degree_prior.py']})
    save(HERE/'degree-prior-preflight.json',result);print({k:v for k,v in result.items() if k!='cases'})


if __name__=='__main__':main()
