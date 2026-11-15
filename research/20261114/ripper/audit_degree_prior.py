"""Independent exact-integer proof checker and Decimal110 cost reconstruction."""
from decimal import Decimal,localcontext,ROUND_HALF_EVEN
import hashlib
import json
import numpy as np


def check_certificate(n,degree,ii,jj,cost,t,y,z):
    """Verify a supplied scale-two primal/dual witness without any solver call."""
    if any(a.dtype!=np.int64 for a in [cost,t,y,z]):return False
    if t.shape!=cost.shape or z.shape!=cost.shape or y.shape!=(n,):return False
    if not all(int(v) in (0,1,2) for v in t):return False
    counts=[0]*n
    for left,right,amount in zip(ii,jj,t,strict=True):counts[int(left)]+=int(amount);counts[int(right)]+=int(amount)
    if counts!=[2*degree]*n:return False
    lhs=rhs=0
    for left,right,c,x,upper in zip(ii,jj,cost,t,z,strict=True):
        left,right,c,x,upper=map(int,[left,right,c,x,upper]);reduced=2*c-int(y[left])-int(y[right])-upper
        if upper>0 or reduced<0 or (x>0 and reduced!=0) or (x<2 and upper!=0):return False
        lhs+=c*x;rhs+=upper
    rhs+=degree*sum(int(value) for value in y)
    return lhs==rhs


def independent_costs(m,c):
    values=[];minimum_margin=float('inf')
    with localcontext() as ctx:
        ctx.prec=110;q0=Decimal(9)/1600;q1=Decimal(1)/9
        equal=(q1/q0).ln();different=((1-q1)/(1-q0)).ln()
        for exposures,equalities in zip(m,c,strict=True):
            value=Decimal(1000000)*(int(equalities)*equal+(int(exposures)-int(equalities))*different)
            rounded=value.to_integral_value(rounding=ROUND_HALF_EVEN)
            margin=Decimal('.5')-abs(value-rounded);assert margin>Decimal('1e-60')
            minimum_margin=min(minimum_margin,float(margin));values.append(int(rounded))
    return np.array(values,np.int64),minimum_margin


def audit_bank(seed):
    from degree_prior_study import HERE,PARENT,OUT,PROTOCOL,ARMS,sha,save,load_npz,verify
    p=verify(seed);d=OUT/f's{seed}';a=load_npz(d/'model.npz');r=json.loads((d/'receipt.json').read_text())
    assert r['seed']==seed and r['status']=='complete' and r['protocol_sha256']==sha(PROTOCOL)
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    old=load_npz(PARENT/f's{seed}'/'fit.npz');cost,margin=independent_costs(old['joint_count'],old['equal_count'])
    assert np.array_equal(cost,a['cost']) and np.array_equal(a['original_support'],old['support'])
    selected=sorted((int(cost[e]),e) for e in range(3240) if old['support'][e])[:810]
    top=np.zeros(3240,bool);top[[edge for _,edge in selected]]=True;assert np.array_equal(top,a['top810_support'])
    ii,jj=np.triu_indices(81,1);indices=np.flatnonzero(old['support']);assert np.array_equal(indices,a['candidate_indices'])
    certified=check_certificate(81,20,ii[indices],jj[indices],cost[indices],a['primal_twice'],a['node_dual_twice'],a['upper_dual_twice'])
    assert certified==r['certificate']['certified']
    raw=a['primal_values'];dual=a['row_dual_values'];candidate_valid=False;candidate_integral=False;candidate_support=np.zeros(3240,bool)
    if raw.shape==indices.shape and np.isfinite(raw).all() and np.isin(raw,[0.,1.]).all():
        candidate_integral=True;candidate_support[indices]=raw==1
        degrees=np.zeros(81,np.int64)
        for edge in np.flatnonzero(candidate_support):degrees[ii[edge]]+=1;degrees[jj[edge]]+=1
        candidate_valid=bool(np.all(degrees==20))
    if certified:
        assert r['solver']['solver_optimal'] and r['solver']['value_valid'] and r['solver']['dual_valid']
        assert np.array_equal(raw*2,a['primal_twice'])
        assert dual.shape==(81,) and np.all(np.abs(dual*2-a['node_dual_twice'])<=1e-6)
        obj=sum(int(c)*int(x) for c,x in zip(cost[indices],a['primal_twice'],strict=True))
        assert obj==r['certificate']['primal_objective_twice']==r['certificate']['dual_objective_twice']
        assert r['certificate']['integral']==bool(np.isin(a['primal_twice'],[0,2]).all())
        assert abs(r['solver']['solver_objective']-obj/2)<1e-5
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    def stats(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,
                    support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    rows=[dict(seed=seed,arm=ARMS[0],**stats(old['support'])),dict(seed=seed,arm=ARMS[1],**stats(top))]
    candidate=stats(candidate_support) if candidate_valid else None
    if certified and r['certificate']['integral']:
        assert candidate_valid
        category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact'
    elif certified:category='resolved_failure_fractional'
    else:category='unresolved'
    assert category==r['classification']
    solver=r['solver'];options=solver['solver_options']
    assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
    for key,value in p['solver_options'].items():
        if key=='random_seed_base':assert options['random_seed']==value+seed
        else:assert options[key]==value,(key,options[key],value)
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),
                receipt_sha256=sha(d/'receipt.json'),model_sha256=sha(d/'model.npz'),rows=rows,
                classification=category,certificate_verified=certified,candidate_integral=candidate_integral,
                candidate_feasible=candidate_valid,integral_candidate_graph=candidate,
                fractional_variables=int((raw==.5).sum()) if raw.shape==indices.shape else None,
                minimum_cost_rounding_margin=margin,certificate_reason=r['certificate']['reason'],solver=solver)
    save(d/'audit.json',result);return result
