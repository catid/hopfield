"""Independent exposure-balanced masks, common-noise replay, overlap accounting and LP proofs."""
from fractions import Fraction
import hashlib,json
import numpy as np
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate


def reference_masks(seed,rows=900):
    from fast_reference_mask import reference_mask
    rng=np.random.default_rng(seed);joint=np.zeros((81,81),np.int32);excluded=np.zeros((81,81),bool);answer=[]
    for _ in range(rows):
        mask=reference_mask('exposure_exact',joint,excluded,rng.permutation(81));answer.append(mask)
        # Independent outer-product exposure update.
        joint+=mask[:,None]&mask[None,:]
    assert not excluded.any()
    return np.asarray(answer,bool)


def exposure_statistics(counts):
    values=list(map(int,counts));total=sum(values);squares=sum(v*v for v in values)
    return dict(minimum=min(values),maximum=max(values),sum=total,sum_of_squares=squares,dispersion=len(values)*squares-total*total)


def audit_bank(seed):
    from balanced_mask_study import ROOT,HERE,PARENT,UNIFORM,OUT,PROTOCOL,sha,save,load_npz,verify
    p=verify(seed);d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz');observed=load_npz(d/'observations.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL) and r['producer_sha256']==sha(HERE/'balanced_mask_study.py')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    assert r['mask_sha256_before_labels']==sha(d/'masks.npz') and r['mask_seed']==p['mask_seed_base']+seed
    given=reference_masks(p['mask_seed_base']+seed);assert np.array_equal(given,observed['given']) and np.array_equal(given,load_npz(d/'masks.npz')['given'])
    assert given.shape==(900,81) and (given.sum(1)==25).all() and observed['values'].dtype==np.int8 and np.all(observed['values'][~given]==-1)
    parent_p=json.loads((HERE/'protocol-fresh-rate.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text());entry=manifest['inputs'][seed]
    assert entry['seed']==seed and r['source_sha256']==entry['training_sha256']==sha(ROOT/entry['training_path'])
    assert r['noise_sha256']==sha(PARENT/f's{seed}'/'potential_noise.npz') and r['baseline_sha256']==sha(PARENT/f's{seed}'/'recovery.npz')
    clean=load_npz(ROOT/entry['training_path'])['solutions'];noise=load_npz(PARENT/f's{seed}'/'potential_noise.npz')
    nrng=np.random.default_rng(parent_p['streams']['training_noise']+seed);ii,jj=np.triu_indices(81,1);m=np.zeros(3240,np.int64);c=np.zeros_like(m)
    for row in range(900):
        uniform=float(nrng.random());permutation=nrng.permutation(81).astype(np.uint8)
        assert uniform==noise['uniforms'][row] and np.array_equal(permutation,noise['permutations'][row])
        assert set(map(int,permutation))==set(range(81));corrupted=Fraction(uniform)<Fraction(9,160)
        expected=clean[row,permutation] if corrupted else clean[row]
        values=observed['values'][row];assert np.array_equal(values[given[row]],expected[given[row]])
        queried=given[row,ii]&given[row,jj];m+=queried;c+=queried&(values[ii]==values[jj])
    assert np.array_equal(m,z['joint_count']) and np.array_equal(c,z['equal_count']) and int(m.sum())==270000
    cost=independent_rate_costs(m,c);assert np.array_equal(cost,z['cost']);top=np.zeros(3240,bool)
    top[[edge for _,edge in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,z['top810_support'])
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    baseline=load_npz(PARENT/f's{seed}'/'recovery.npz')
    for name,value in baseline.items():assert np.array_equal(z['baseline_'+name],value)
    baseline_cost=independent_rate_costs(z['baseline_joint_count'],z['baseline_equal_count']);assert np.array_equal(baseline_cost,z['baseline_cost'])
    bt,by,bupper=[z['baseline_'+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert check_certificate(81,20,ii,jj,baseline_cost,bt,by,bupper) and np.isin(bt,[0,2]).all()
    assert np.array_equal(z['baseline_primal_values']*2,bt) and np.all(np.abs(z['baseline_row_dual_values']*2-by)<=1e-6) and np.array_equal(bt==2,truth)
    rows=[dict(seed=seed,arm=arm,**statistics(support)) for arm,support in [('adaptive_degree20',bt==2),('balanced_top810',top)]]
    uniform=load_npz(UNIFORM/f's{seed}'/'model.npz');assert r['uniform_baseline_sha256']==sha(UNIFORM/f's{seed}'/'model.npz')
    for name,value in uniform.items():
        if not name.startswith('baseline_'):assert np.array_equal(z['uniformbaseline_'+name],value)
    ucost=independent_rate_costs(z['uniformbaseline_joint_count'],z['uniformbaseline_equal_count']);assert np.array_equal(ucost,z['uniformbaseline_cost'])
    ut,uy,uu=[z['uniformbaseline_'+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert check_certificate(81,20,ii,jj,ucost,ut,uy,uu)
    assert np.array_equal(z['uniformbaseline_primal_values']*2,ut) and np.all(np.abs(z['uniformbaseline_row_dual_values']*2-uy)<=1e-6)
    if np.isin(ut,[0,2]).all():
        graph=ut==2;ucategory='success' if np.array_equal(graph,truth) else 'resolved_failure_nonexact'
        rows.append(dict(seed=seed,arm='archived_uniform_degree20',certified=True,**statistics(graph)))
    else:ucategory='resolved_failure_fractional'
    assert ucategory==json.loads((UNIFORM/f's{seed}'/'audit.json').read_text())['classification']
    t,y,upper=[z[name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
    certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==r['certificate']['certified'];raw=z['primal_values'];candidate=None
    if raw.shape==(3240,) and np.isfinite(raw).all() and np.isin(raw,[0.,1.]).all():
        support=raw==1;degree=np.bincount(np.r_[ii[support],jj[support]],minlength=81)
        if np.all(degree==20):candidate=statistics(support)
    if certified:
        assert r['solver']['solver_optimal'] and r['solver']['value_valid'] and r['solver']['dual_valid']
        assert np.array_equal(raw*2,t) and np.all(np.abs(z['row_dual_values']*2-y)<=1e-6)
        objective=sum(int(w)*int(x) for w,x in zip(cost,t,strict=True))
        assert objective==r['certificate']['primal_objective_twice']==r['certificate']['dual_objective_twice']
        assert abs(r['solver']['solver_objective']-objective/2)<1e-5
        integral=bool(np.isin(t,[0,2]).all());assert integral==r['certificate']['integral']
        if integral:
            assert candidate is not None;category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact'
            rows.append(dict(seed=seed,arm='balanced_degree20',certified=True,**candidate))
        else:category='resolved_failure_fractional'
    else:category='unresolved'
    assert category==r['classification'];solver=r['solver'];assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
    for name,value in p['solver_options'].items():
        if name=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
        else:assert solver['solver_options'][name]==value
    with np.load(PARENT/f's{seed}'/'observations.npz',allow_pickle=False) as old:adaptive=old['given'].copy();oldvalues=old['values'].copy()
    assert adaptive.shape==(900,81) and adaptive.dtype==bool and np.all(adaptive.sum(1)==25)
    with np.load(UNIFORM/f's{seed}'/'observations.npz',allow_pickle=False) as old:uniformgiven=old['given'].copy();uniformvalues=old['values'].copy()
    assert uniformgiven.shape==(900,81) and uniformgiven.dtype==bool and np.all(uniformgiven.sum(1)==25)
    previous=adaptive|uniformgiven;reused=given&previous;new=given&~previous
    assert np.array_equal(observed['values'][given&adaptive],oldvalues[given&adaptive])
    assert np.array_equal(observed['values'][given&uniformgiven],uniformvalues[given&uniformgiven])
    prior_union=24500+int((uniformgiven&~adaptive).sum())
    accounting=dict(reused_queries=int(reused.sum()),newly_exposed_queries=int(new.sum()),balanced_training_queries=int(given.sum()),new_calibration_queries=0,
                    prior_cumulative_union_queries=prior_union,combined_union_queries=prior_union+int(new.sum()))
    assert accounting==r['query_accounting'] and accounting['reused_queries']+accounting['newly_exposed_queries']==22500
    diagnostics={arm:exposure_statistics(counts) for arm,counts in [('balanced',m),('uniform',z['uniformbaseline_joint_count']),('adaptive',z['baseline_joint_count'])]}
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),model_sha256=sha(d/'model.npz'),
                source_sha256=r['source_sha256'],noise_sha256=r['noise_sha256'],baseline_sha256=r['baseline_sha256'],uniform_baseline_sha256=r['uniform_baseline_sha256'],uniform_classification=ucategory,exposure_diagnostics=diagnostics,rows=rows,classification=category,
                certificate_verified=certified,integral_candidate_graph=candidate,fractional_variables=int((raw==.5).sum()) if raw.shape==(3240,) else None,
                archived_optimality_certificates_verified=True,independent_masks_replayed=900,common_noise_rows_replayed=900,query_accounting=accounting,solver=solver)
    save(d/'audit.json',result);return result
