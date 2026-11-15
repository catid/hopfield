"""Independent Fraction costs and exact new/archived LP proof verification."""
from fractions import Fraction
import hashlib,json
import numpy as np
from audit_degree_prior import check_certificate,independent_costs


def independent_rate_costs(m,c):
    assert m.shape==c.shape and np.all((0<=c)&(c<=m))
    return np.array([round(Fraction(1000000*(int(equal)+1),int(exposed)+2))
                     for exposed,equal in zip(m,c,strict=True)],np.int64)


def audit_bank(seed):
    from laplace_rate_study import HERE,PARENT,BASELINE,OUT,PROTOCOL,ARMS,sha,save,load_npz,verify
    p=verify(seed);d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL)
    assert r['parent_fit_sha256']==sha(PARENT/f's{seed}'/'fit.npz') and r['baseline_model_sha256']==sha(BASELINE/f's{seed}'/'model.npz')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    old=load_npz(PARENT/f's{seed}'/'fit.npz');baseline=load_npz(BASELINE/f's{seed}'/'model.npz')
    for name in ['joint_count','equal_count']:assert np.array_equal(z[name],old[name]) and z[name].dtype==old[name].dtype
    assert np.array_equal(z['original_support'],old['support'])
    cost=independent_rate_costs(z['joint_count'],z['equal_count']);assert np.array_equal(cost,z['cost'])
    assert cost.dtype==np.int64 and ((cost>=0)&(cost<=1000000)).all()
    top=np.zeros(3240,bool);top[[edge for _,edge in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
    assert np.array_equal(top,z['top810_support'])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    for name in ['cost','unrestricted_primal_values','unrestricted_row_dual_values','unrestricted_primal_twice','unrestricted_node_dual_twice','unrestricted_upper_dual_twice']:
        assert np.array_equal(z['baseline_'+name],baseline[name])
    original_cost,_=independent_costs(z['joint_count'],z['equal_count']);assert np.array_equal(original_cost,z['baseline_cost'])
    bt,by,bupper=[z['baseline_unrestricted_'+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert check_certificate(81,20,ii,jj,original_cost,bt,by,bupper) and np.isin(bt,[0,2]).all()
    assert np.array_equal(z['baseline_unrestricted_primal_values']*2,bt)
    assert np.all(np.abs(z['baseline_unrestricted_row_dual_values']*2-by)<=1e-6)
    baseline_support=bt==2;assert np.array_equal(baseline_support,truth)
    t,y,upper=[z[name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
    cert=check_certificate(81,20,ii,jj,cost,t,y,upper);assert cert==r['certificate']['certified']
    raw=z['primal_values'];candidate=None
    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,
                    support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    if raw.shape==(3240,) and np.isfinite(raw).all() and np.isin(raw,[0.,1.]).all():
        support=raw==1;degree=np.bincount(np.concatenate([ii[support],jj[support]]),minlength=81)
        if (degree==20).all():candidate=statistics(support)
    if cert:
        assert r['solver']['solver_optimal'] and r['solver']['value_valid'] and r['solver']['dual_valid']
        assert np.array_equal(raw*2,t) and np.all(np.abs(z['row_dual_values']*2-y)<=1e-6)
        objective=sum(int(c)*int(x) for c,x in zip(cost,t,strict=True))
        assert objective==r['certificate']['primal_objective_twice']==r['certificate']['dual_objective_twice']
        assert abs(r['solver']['solver_objective']-objective/2)<1e-5
        integral=bool(np.isin(t,[0,2]).all());assert integral==r['certificate']['integral']
        if integral:
            assert candidate is not None
            category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact'
        else:category='resolved_failure_fractional'
    else:category='unresolved'
    assert category==r['classification']
    solver=r['solver'];assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
    for name,value in p['solver_options'].items():
        if name=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
        else:assert solver['solver_options'][name]==value
    rows=[dict(seed=seed,arm=arm,**statistics(support)) for arm,support in
          zip(ARMS[:3],[old['support'],baseline_support,top],strict=True)]
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),
                model_sha256=sha(d/'model.npz'),parent_fit_sha256=r['parent_fit_sha256'],baseline_model_sha256=r['baseline_model_sha256'],
                rows=rows,classification=category,certificate_verified=cert,integral_candidate_graph=candidate,
                fractional_variables=int((raw==.5).sum()) if raw.shape==(3240,) else None,
                archived_optimality_certificate_verified=True,certificate_reason=r['certificate']['reason'],solver=solver)
    save(d/'audit.json',result);return result
