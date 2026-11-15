"""H1av exact proofs, fixed observations and hard-restriction diagnostics."""
import hashlib,json
import numpy as np
from audit_degree_prior import check_certificate,independent_costs


def audit_arm(arm,z,metadata,candidate,cost,truth,seed,p):
    ii,jj=np.triu_indices(81,1);indices=np.flatnonzero(candidate)
    assert np.array_equal(z['candidate_indices'],indices)
    degree=[0]*81
    for edge in indices:degree[int(ii[edge])]+=1;degree[int(jj[edge])]+=1
    deficient=[cell for cell,value in enumerate(degree) if value<20];deficit=metadata['infeasibility_certificate']
    cert=check_certificate(81,20,ii[indices],jj[indices],cost[indices],z['primal_twice'],z['node_dual_twice'],z['upper_dual_twice'])
    assert cert==metadata['certificate']['certified']
    raw=z['primal_values'];integral_candidate=None
    if raw.shape==indices.shape and np.isfinite(raw).all() and np.isin(raw,[0.,1.]).all():
        support=np.zeros(3240,bool);support[indices]=raw==1;degrees=np.zeros(81,np.int64)
        np.add.at(degrees,ii[support],1);np.add.at(degrees,jj[support],1)
        if (degrees==20).all():
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            integral_candidate=dict(exact_graph=missing==extra==0,missing_true_edges=missing,extra_edges=extra,
                                    support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    solver=metadata['solver']
    if deficit is not None:
        assert deficient and deficit['type']=='integer_degree_deficit' and deficit['vertex']==deficient[0]
        assert deficit['available_neighbors']==degree[deficient[0]]<deficit['required_degree']==20
        assert solver is None and not cert and all(len(z[k])==0 for k in ['primal_values','row_dual_values','primal_twice','node_dual_twice','upper_dual_twice'])
        category='resolved_failure_infeasible'
    else:
        assert not deficient and solver is not None
        assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
        for key,value in p['solver_options'].items():
            if key=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
            else:assert solver['solver_options'][key]==value
        if cert:
            assert solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']
            assert np.array_equal(raw*2,z['primal_twice']) and np.all(np.abs(z['row_dual_values']*2-z['node_dual_twice'])<=1e-6)
            obj=sum(int(c)*int(x) for c,x in zip(cost[indices],z['primal_twice'],strict=True))
            assert obj==metadata['certificate']['primal_objective_twice']==metadata['certificate']['dual_objective_twice']
            assert abs(solver['solver_objective']-obj/2)<1e-5
            integral=bool(np.isin(z['primal_twice'],[0,2]).all());assert integral==metadata['certificate']['integral']
            if integral:
                assert integral_candidate is not None
                category='success' if integral_candidate['exact_graph'] else 'resolved_failure_nonexact'
            else:category='resolved_failure_fractional'
        else:category='unresolved'
    assert category==metadata['classification']
    omitted=int((truth&~candidate).sum())
    if omitted:assert category!='success'
    return dict(arm=arm,classification=category,certificate_verified=cert,feasibility_certificate=deficit,
                integral_candidate_graph=integral_candidate,candidate_pairs=len(indices),omitted_true_edges=omitted,
                exact_truth_recovery_impossible_due_to_candidate_restriction=omitted>0,solver=solver,
                fractional_variables=int((raw==.5).sum()) if raw.shape==indices.shape else None,
                certificate_reason=metadata['certificate']['reason'])


def audit_bank(seed):
    from exclusion_repair_study import HERE,PARENT,OUT,PROTOCOL,sha,save,load_npz,verify
    p=verify(seed);d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL)
    assert r['parent_fit_sha256']==sha(PARENT/f's{seed}'/'fit.npz')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    old=load_npz(PARENT/f's{seed}'/'fit.npz')
    for name in ['joint_count','equal_count']:assert z[name].dtype==old[name].dtype and np.array_equal(z[name],old[name])
    assert np.array_equal(z['original_support'],old['support'])
    cost,margin=independent_costs(z['joint_count'],z['equal_count']);assert np.array_equal(cost,z['cost'])
    chosen=sorted((int(cost[e]),e) for e in range(3240))[:810];top=np.zeros(3240,bool);top[[e for _,e in chosen]]=True
    assert np.array_equal(top,z['top810_support'])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    rows=[]
    for arm,support in [('original_support',old['support']),('top810_unrestricted',top)]:
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        rows.append(dict(seed=seed,arm=arm,retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,
                         exact_graph=missing==extra==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest()))
    arms={}
    for arm,candidate in [('hard',old['support']),('unrestricted',np.ones(3240,bool))]:
        arrays={name.removeprefix(arm+'_'):value for name,value in z.items() if name.startswith(arm+'_')}
        metadata=json.loads((d/f'{arm}-fit.json').read_text());assert metadata['arm']==arm
        arms[arm]=audit_arm(arm,arrays,metadata,candidate,cost,truth,seed,p)
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),
                model_sha256=sha(d/'model.npz'),parent_fit_sha256=r['parent_fit_sha256'],rows=rows,arms=arms,
                original_missing_true_edges=int((truth&~old['support']).sum()),minimum_cost_rounding_margin=margin)
    save(d/'audit.json',result);return result
