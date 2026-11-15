"""H1ax independent pair-vector recount, suffix gates and exact LP proofs."""
import hashlib,json
import numpy as np
from audit_degree_prior import check_certificate
from audit_laplace_rate import independent_rate_costs


def recount(given,values,prefixes):
    assert given.shape==values.shape and given.shape[1]==81 and given.dtype==bool and values.dtype==np.int8
    assert np.all(given.sum(axis=1)==25) and np.all(values[~given]==-1) and np.all((values[given]>=0)&(values[given]<9))
    ii,jj=np.triu_indices(81,1);joint=np.zeros(3240,np.int64);equal=joint.copy();result={}
    for row in range(max(prefixes)):
        queried=given[row,ii]&given[row,jj]
        joint+=queried;equal+=queried&(values[row,ii]==values[row,jj])
        if row+1 in prefixes:result[row+1]=(joint.copy(),equal.copy())
    return result


def suffix_gate(given,values,n,expected):
    from prefix_rate_features import prefix_counts
    changed_given=given.copy();changed_values=values.copy()
    changed_given[n:]=np.roll(given[n:],17,axis=1)
    rolled=np.roll(values[n:],17,axis=1)
    changed_values[n:]=np.where(changed_given[n:],(rolled+3)%9,-1)
    assert not np.array_equal(changed_given[n:],given[n:]) and not np.array_equal(changed_values[n:],values[n:])
    assert np.array_equal(changed_given[:n],given[:n]) and np.array_equal(changed_values[:n],values[:n])
    actual=prefix_counts(changed_given,changed_values,[n])[n]
    for got,want in zip(actual,expected,strict=True):assert np.array_equal(got,want)
    assert np.array_equal(independent_rate_costs(*actual),independent_rate_costs(*expected))
    return dict(rows=n,status='validated',changed_suffix_rows=len(given)-n,
                original_suffix_sha256=hashlib.sha256(given[n:].tobytes()+values[n:].tobytes()).hexdigest(),
                substituted_suffix_sha256=hashlib.sha256(changed_given[n:].tobytes()+changed_values[n:].tobytes()).hexdigest())


def audit_bank(seed,observed=None):
    from prefix_rate_study import HERE,PARENT,BASELINE,OUT,PROTOCOL,sha,save,load_npz,verify,observations
    p=verify(seed);d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL)
    assert r['producer_sha256']==sha(HERE/'prefix_rate_study.py')
    assert r['observations_sha256']==sha(PARENT/f's{seed}'/'observations.npz') and r['baseline_model_sha256']==sha(BASELINE/f's{seed}'/'model.npz')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    given,values=observations(seed) if observed is None else observed
    # This independent post-fit audit may inspect all rows to reverify the archived comparator.
    counts=recount(given,values,p['prefixes']+[1110]);baseline=load_npz(BASELINE/f's{seed}'/'model.npz')
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    rows=[];audits={};gates=[]

    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,
                    support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())

    for n in p['prefixes']+[1110]:
        prefix='baseline_' if n==1110 else f'n{n}_'
        m,c=counts[n]
        assert np.array_equal(z[prefix+'joint_count'],m) and np.array_equal(z[prefix+'equal_count'],c)
        cost=independent_rate_costs(m,c);assert np.array_equal(z[prefix+'cost'],cost)
        top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
        assert np.array_equal(z[prefix+'top810_support'],top)
        rows.append(dict(seed=seed,rows=n,arm='top810',**statistics(top)))
        if n==1110:
            for name in baseline:
                if prefix+name in z:assert np.array_equal(z[prefix+name],baseline[name])
            record=json.loads((BASELINE/f's{seed}'/'receipt.json').read_text())
        else:
            record=json.loads((d/f'n{n}-fit.json').read_text());assert record['rows']==n
            gates.append(suffix_gate(given,values,n,counts[n]))
        t,y,upper=[z[prefix+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
        cert=check_certificate(81,20,ii,jj,cost,t,y,upper);assert cert==record['certificate']['certified']
        raw=z[prefix+'primal_values'];candidate=None
        if raw.shape==(3240,) and np.isfinite(raw).all() and np.isin(raw,[0.,1.]).all():
            support=raw==1;degree=np.bincount(np.concatenate([ii[support],jj[support]]),minlength=81)
            if (degree==20).all():candidate=statistics(support)
        if cert:
            assert record['solver']['solver_optimal'] and record['solver']['value_valid'] and record['solver']['dual_valid']
            assert np.array_equal(raw*2,t) and np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
            objective=sum(int(w)*int(x) for w,x in zip(cost,t,strict=True))
            assert objective==record['certificate']['primal_objective_twice']==record['certificate']['dual_objective_twice']
            assert abs(record['solver']['solver_objective']-objective/2)<1e-5
            integral=bool(np.isin(t,[0,2]).all());assert integral==record['certificate']['integral']
            if integral:
                assert candidate is not None
                category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact'
            else:category='resolved_failure_fractional'
        else:category='unresolved'
        assert category==record['classification']
        solver=record['solver'];assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
        for name,value in p['solver_options'].items():
            if name=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
            else:assert solver['solver_options'][name]==value
        if candidate is not None:rows.append(dict(seed=seed,rows=n,arm='degree20',certified=cert,**candidate))
        audits[str(n)]=dict(classification=category,certificate_verified=cert,integral_candidate_graph=candidate,
                           fractional_variables=int((raw==.5).sum()) if raw.shape==(3240,) else None,
                           certificate_reason=record['certificate']['reason'],solver=solver)
        if n==1110:assert cert and candidate is not None and category=='success'
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),
                model_sha256=sha(d/'model.npz'),observations_sha256=r['observations_sha256'],baseline_model_sha256=r['baseline_model_sha256'],
                rows=rows,prefix_audits=audits,suffix_substitution_gates=gates,independent_pair_states_checked=1110*3240,
                archived_optimality_certificate_verified=True)
    save(d/'audit.json',result);return result
