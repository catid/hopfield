"""Independent sequential soft feedback, suffix isolation and exact LP audit."""
from fractions import Fraction
import hashlib,json
import numpy as np
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate


def reference_choose(joint,equal,order,permutation=None):
    feedback=equal if permutation is None else equal[permutation[:,None],permutation[None,:]]
    weights=(8./9.)**joint.astype(np.float64)/(1.+feedback);weights[np.diag_indices(81)]=0.
    selected=[];accumulator=None
    for step in range(25):
        score=weights.sum(1) if step==0 else accumulator.copy();score[selected]=-np.inf;best=score.max()
        cell=next(int(v) for v in order if score[v]==best);selected.append(cell)
        accumulator=weights[:,cell].copy() if accumulator is None else accumulator+weights[:,cell]
    return np.isin(np.arange(81),selected)


def reference_replay(clean,noise,tie_seed,permutation,checkpoint,observed=None,events=None):
    n=len(clean);joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);ii,jj=np.triu_indices(81,1);ties=np.random.default_rng(tie_seed)
    given=np.zeros((n,81),bool);values=np.full((n,81),-1,np.int8);prefix=None
    for row in range(n):
        mask=reference_choose(joint,equal,ties.permutation(81),permutation)
        if events is not None:
            event=events[row];assert event['row']==row and event['mask_finalized_before_current_labels_and_latents'] is True
            assert event['past_state_sha256']==hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest() and event['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
        if observed is not None:assert np.array_equal(mask,observed['given'][row])
        corrupted=Fraction(float(noise['uniforms'][row]))<Fraction(9,160);board=clean[row,noise['permutations'][row]] if corrupted else clean[row]
        given[row]=mask;values[row,mask]=board[mask]
        if observed is not None:assert np.array_equal(values[row],observed['values'][row])
        queried=mask[:,None]&mask[None,:];joint+=queried;equal+=queried&(values[row,:,None]==values[row,None,:])
        if row+1==checkpoint:prefix=(joint[ii,jj].copy(),equal[ii,jj].copy())
    return dict(given=given,values=values,joint_count=joint[ii,jj],equal_count=equal[ii,jj],prefix_joint_count=prefix[0],prefix_equal_count=prefix[1])


def suffix_gate(clean,noise,tie_seed,permutation,checkpoint,expected):
    from soft_feedback_policy import collect
    changed=clean.copy();changed[checkpoint:]=(np.roll(clean[checkpoint:],19,axis=1)+2)%9
    uniforms=noise['uniforms'].copy();uniforms[checkpoint:]=1-uniforms[checkpoint:]
    permutations=noise['permutations'].copy();permutations[checkpoint:]=np.roll(permutations[checkpoint:],11,axis=1)
    assert not np.array_equal(changed[checkpoint:],clean[checkpoint:])
    actual,_=collect(changed,uniforms,permutations,tie_seed,permutation,checkpoint)
    assert np.array_equal(actual['given'][:checkpoint],expected['given'][:checkpoint]) and np.array_equal(actual['values'][:checkpoint],expected['values'][:checkpoint])
    for key in ['prefix_joint_count','prefix_equal_count']:assert np.array_equal(actual[key],expected[key])
    before=hashlib.sha256(clean[checkpoint:].tobytes()+noise['uniforms'][checkpoint:].tobytes()+noise['permutations'][checkpoint:].tobytes()).hexdigest()
    after=hashlib.sha256(changed[checkpoint:].tobytes()+uniforms[checkpoint:].tobytes()+permutations[checkpoint:].tobytes()).hexdigest();assert before!=after
    return dict(status='validated',checkpoint=checkpoint,prefix_masks_and_labels_unchanged=True,prefix_counts_unchanged=True,original_suffix_sha256=before,substituted_suffix_sha256=after)


def audit_bank(seed,raw=None):
    from soft_feedback_study import ROOT,HERE,PARENT,UNIFORM,BALANCED,OUT,PROTOCOL,sha,save,load_npz,verify
    p=verify(seed);d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL) and r['producer_sha256']==sha(HERE/'soft_feedback_study.py')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    entry=json.loads((PARENT/'input_manifest.json').read_text())['inputs'][seed];assert r['source_sha256']==entry['training_sha256']==sha(ROOT/entry['training_path'])
    assert r['noise_sha256']==sha(PARENT/f's{seed}'/'potential_noise.npz')
    clean,noise=(load_npz(ROOT/entry['training_path'])['solutions'],load_npz(PARENT/f's{seed}'/'potential_noise.npz')) if raw is None else raw
    parent_p=json.loads((HERE/'protocol-fresh-rate.json').read_text());nrng=np.random.default_rng(parent_p['streams']['training_noise']+seed)
    for row in range(900):
        uniform=nrng.random();permutation=nrng.permutation(81).astype(np.uint8)
        assert uniform==noise['uniforms'][row] and np.array_equal(permutation,noise['permutations'][row])
    feedback_perm=np.random.default_rng(p['feedback_permutation_seed_base']+seed).permutation(81)
    assert np.array_equal(feedback_perm,p['feedback_permutations'][seed]) and set(map(int,feedback_perm))==set(range(81))
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    rows=[];arm_audits={};observations={};gates=[]
    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    for arm in ['aligned','permuted','adaptive','balanced']:
        prefix=arm+'_';record=None
        if arm in ['aligned','permuted']:
            observed=load_npz(d/f'{arm}-observations.npz');events=json.loads((d/f'{arm}-events.json').read_text());assert len(events)==900
            perm=feedback_perm if arm=='permuted' else None
            replay=reference_replay(clean,noise,p['tie_seed_base']+seed,perm,p['suffix_checkpoint'],observed,events)
            for name,value in replay.items():assert np.array_equal(value,observed[name])
            assert np.array_equal(observed['given'],z[arm+'_mask'])
            for key in ['joint_count','equal_count']:assert np.array_equal(observed[key],z[prefix+key])
            observations[arm]=observed;gates.append(dict(arm=arm,**suffix_gate(clean,noise,p['tie_seed_base']+seed,perm,p['suffix_checkpoint'],observed)))
            record=json.loads((d/f'{arm}-fit.json').read_text())
        else:
            path=PARENT if arm=='adaptive' else BALANCED;name='recovery.npz' if arm=='adaptive' else 'model.npz';old=load_npz(path/f's{seed}'/name)
            for key in ['joint_count','equal_count','cost','primal_values','row_dual_values','primal_twice','node_dual_twice','upper_dual_twice']:assert np.array_equal(z[prefix+key],old[key])
            record=json.loads((path/f's{seed}'/('recovery.json' if arm=='adaptive' else 'receipt.json')).read_text())
        m,c=z[prefix+'joint_count'],z[prefix+'equal_count'];assert int(m.sum())==270000
        cost=independent_rate_costs(m,c);assert np.array_equal(cost,z[prefix+'cost'])
        if arm in ['aligned','permuted']:
            top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
            assert np.array_equal(top,z[prefix+'top810_support']);rows.append(dict(seed=seed,arm=arm+'_top810',**statistics(top)))
        t,y,upper=[z[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
        certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==record['certificate']['certified'];rawx=z[prefix+'primal_values'];candidate=None
        if rawx.shape==(3240,) and np.isfinite(rawx).all() and np.isin(rawx,[0.,1.]).all():
            graph=rawx==1;degree=np.bincount(np.r_[ii[graph],jj[graph]],minlength=81)
            if np.all(degree==20):candidate=statistics(graph)
        if certified:
            assert record['solver']['solver_optimal'] and record['solver']['value_valid'] and record['solver']['dual_valid']
            assert np.array_equal(rawx*2,t) and np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
            objective=sum(int(w)*int(x) for w,x in zip(cost,t,strict=True));assert objective==record['certificate']['primal_objective_twice']==record['certificate']['dual_objective_twice']
            assert abs(record['solver']['solver_objective']-objective/2)<1e-5
            integral=bool(np.isin(t,[0,2]).all());assert integral==record['certificate']['integral']
            if integral:
                assert candidate is not None;category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact';rows.append(dict(seed=seed,arm=arm,certified=True,**candidate))
            else:category='resolved_failure_fractional'
        else:category='unresolved'
        assert category==record['classification'];solver=record['solver'];assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
        for name,value in p['solver_options'].items():
            if name=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
            else:assert solver['solver_options'][name]==value
        arm_audits[arm]=dict(classification=category,certificate_verified=certified,integral_candidate_graph=candidate,fractional_variables=int((rawx==.5).sum()) if rawx.shape==(3240,) else None,solver=solver)
        if arm in ['adaptive','balanced']:assert certified and candidate is not None
    previous=np.zeros((900,81),bool)
    for path in [PARENT,UNIFORM,BALANCED]:
        with np.load(path/f's{seed}'/'observations.npz',allow_pickle=False) as old:given=old['given'].copy();values=old['values'].copy()
        previous|=given
        for arm in ['aligned','permuted']:
            overlap=given&observations[arm]['given'];assert np.array_equal(values[overlap],observations[arm]['values'][overlap])
    assert np.array_equal(previous,z['previous_mask_union']);aligned=observations['aligned']['given'];permuted=observations['permuted']['given'];shared=aligned&permuted
    assert np.array_equal(observations['aligned']['values'][shared],observations['permuted']['values'][shared])
    accounting=dict(prior_union_queries=int(previous.sum())+2000,new_union_queries=int(((aligned|permuted)&~previous).sum()),aligned_reused=int((aligned&previous).sum()),
                    aligned_new_relative_prior=int((aligned&~previous).sum()),permuted_reused=int((permuted&previous).sum()),permuted_new_relative_prior=int((permuted&~previous).sum()),
                    new_shared_between_arms=int((shared&~previous).sum()),new_calibration_queries=0)
    accounting['combined_union_queries']=accounting['prior_union_queries']+accounting['new_union_queries']
    assert accounting==r['query_accounting'] and accounting['new_union_queries']==accounting['aligned_new_relative_prior']+accounting['permuted_new_relative_prior']-accounting['new_shared_between_arms']
    for arm in ['aligned','permuted']:assert accounting[arm+'_reused']+accounting[arm+'_new_relative_prior']==22500
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),model_sha256=sha(d/'model.npz'),
                rows=rows,arm_audits=arm_audits,query_accounting=accounting,suffix_substitution_gates=gates,independent_predictable_masks_replayed=1800,common_noise_rows_replayed=900)
    save(d/'audit.json',result);return result
