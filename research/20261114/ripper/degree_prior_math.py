"""H1au integer costs and a deliberately restricted exact LP certificate."""
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
import numpy as np

SCALE=1000000


def integer_costs(exposures,equalities):
    assert exposures.shape==equalities.shape and np.all((0<=equalities)&(equalities<=exposures))
    with localcontext() as ctx:
        ctx.prec=80
        negative=(Decimal(12800)/14319).ln()
        increment=(Decimal(1600)/81).ln()-negative
        values=[int((SCALE*(int(m)*negative+int(c)*increment)).to_integral_value(rounding=ROUND_HALF_EVEN))
                for m,c in zip(exposures,equalities,strict=True)]
    return np.array(values,np.int64)


def top_count_support(candidate,costs,number=810):
    indices=np.flatnonzero(candidate);assert len(indices)>=number
    selected=indices[np.lexsort((indices,costs[indices]))[:number]]
    support=np.zeros_like(candidate);support[selected]=True
    return support


def attempt_certificate(n,degree,ii,jj,cost,primal,node_dual):
    empty=dict(primal_twice=np.empty(0,np.int64),node_dual_twice=np.empty(0,np.int64),upper_dual_twice=np.empty(0,np.int64))
    def fail(reason,**kwargs):return dict(certified=False,reason=reason,**kwargs),empty
    if primal.shape!=cost.shape or node_dual.shape!=(n,):return fail('missing_solution_arrays')
    if not np.isfinite(primal).all() or not np.isfinite(node_dual).all():return fail('nonfinite_solution')
    # No primal rounding or tolerance snapping, including for integral solutions.
    if not np.isin(primal,[0.,.5,1.]).all():return fail('primal_not_literal_half_integral')
    twice=primal*2;t=twice.astype(np.int64)
    degrees=np.zeros(n,np.int64);np.add.at(degrees,ii,t);np.add.at(degrees,jj,t)
    if not np.array_equal(degrees,np.full(n,2*degree,np.int64)):return fail('exact_primal_degree_failure')
    rounded=np.rint(node_dual*2)
    if np.max(np.abs(node_dual*2-rounded))>1e-6:return fail('dual_not_near_scale_two')
    if np.max(np.abs(rounded))>=2**48 or np.max(np.abs(cost))>=2**48:return fail('certificate_range_limit')
    y=rounded.astype(np.int64);z=np.minimum(0,2*cost-y[ii]-y[jj]);reduced=2*cost-y[ii]-y[jj]-z
    if np.any(reduced<0) or np.any(z>0):return fail('exact_dual_feasibility_failure')
    if np.any((t>0)&(reduced!=0)) or np.any((t<2)&(z!=0)):return fail('exact_complementarity_failure')
    primal_twice=sum(int(c)*int(x) for c,x in zip(cost,t,strict=True))
    dual_twice=degree*sum(map(int,y))+sum(map(int,z))
    if primal_twice!=dual_twice:return fail('exact_objective_gap')
    certificate=dict(certified=True,reason='exact_scaled_integer_primal_dual',integral=bool(np.isin(t,[0,2]).all()),
                     primal_objective_twice=primal_twice,dual_objective_twice=dual_twice,
                     maximum_dual_reconstruction_error=float(np.max(np.abs(node_dual*2-rounded))),
                     primal_rounding_used=False,dual_scale=2)
    return certificate,dict(primal_twice=t,node_dual_twice=y,upper_dual_twice=z)
