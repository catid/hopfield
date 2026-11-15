"""Independent transferred-feedback source, mask, label and exact-proof audit."""
from fractions import Fraction
import hashlib,json
import numpy as np
from audit_soft_feedback import reference_choose
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate


def reference_map(ids,prefix):
    assignment=[-1]*len(ids)
    def visit(recipient,seen):
        for donor in range(len(ids)):
            if donor==recipient or donor in seen or np.intersect1d(ids[recipient],ids[donor,:prefix]).size:continue
            seen.add(donor)
            if assignment[donor]<0 or visit(assignment[donor],seen):assignment[donor]=recipient;return True
        return False
    for r in range(len(ids)):
        if not visit(r,set()):return None
    return np.array([assignment.index(r) for r in range(len(ids))],np.int64)


def reference_masks(feedback,tie_seed,rows,events=None):
    joint=np.zeros((81,81),np.int32);ties=np.random.default_rng(tie_seed);given=np.zeros((rows,81),bool)
    for row in range(rows):
        mask=reference_choose(joint,feedback,ties.permutation(81));given[row]=mask
        if events is not None:
            e=events[row];assert e['row']==row and e['past_joint_sha256']==hashlib.sha256(joint.tobytes()).hexdigest()
            assert e['feedback_sha256']==hashlib.sha256(feedback.tobytes()).hexdigest() and e['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
        joint+=mask[:,None]&mask[None,:]
    return given


def complete_label_gate(feedback,tie_seed,clean,noise,expected):
    from transfer_feedback_policy import acquire
    changed=(clean+(np.arange(81)%8+1)[None,:])%9;changed=changed.astype(np.int8)
    assert (changed!=clean).all()
    actual,_=acquire(feedback,tie_seed,changed,noise);mask=expected['given']
    assert np.array_equal(actual['given'],mask) and np.array_equal(actual['joint_count'],expected['joint_count'])
    assert (actual['values'][mask]!=expected['values'][mask]).all() and not np.array_equal(actual['equal_count'],expected['equal_count'])
    return dict(status='validated',all_recipient_source_labels_replaced=True,all_queried_values_changed=True,every_mask_unchanged=True,joint_counts_unchanged=True,actual_equality_counts_changed=True,
                original_source_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),changed_source_sha256=hashlib.sha256(changed.tobytes()).hexdigest())


def audit_bank(seed,raw=None):
    from transfer_feedback_study import ROOT,HERE,PARENT,SNAPSHOT,OUT,PROTOCOL,sha,save,load_npz,verify
    p=verify(seed);d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz');donor=p['donor_map'][seed]
    assert r['status']=='complete' and r['seed']==seed and r['donor']==donor and r['protocol_sha256']==sha(PROTOCOL) and r['producer_sha256']==sha(HERE/'transfer_feedback_study.py')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    assert r['masks_persisted_before_recipient_sources_and_latents'] and r['mask_archive_sha256_before_recipient_labels']==sha(d/'masks.npz')
    ids=load_npz(PARENT/'selected_sources.npz')['source_rows']
    assert np.array_equal(reference_map(ids,300),p['donor_map']) and np.intersect1d(ids[seed],ids[donor,:300]).size==0 and donor!=seed
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    olddonor=load_npz(PARENT/f's{donor}'/'aligned-observations.npz');g=olddonor['given'][:300];v=olddonor['values'][:300]
    donor_counts=((g[:,ii]&g[:,jj])&(v[:,ii]==v[:,jj])).sum(0)
    assert np.array_equal(donor_counts,load_npz(SNAPSHOT/f's{donor}'/'model.npz')['feedback_equal_count'])
    feedback=np.zeros((81,81),np.int32);feedback[ii,jj]=donor_counts;feedback[jj,ii]=donor_counts
    masked=load_npz(d/'masks.npz');events=json.loads((d/'events.json').read_text());assert len(events)==900
    assert np.array_equal(feedback,masked['donor_feedback']) and np.array_equal(feedback,z['donor_feedback'])
    given=reference_masks(feedback,p['tie_seed_base']+seed,900,events)
    assert np.array_equal(given,masked['given']) and np.array_equal(given,z['transfer_mask'])
    entry=json.loads((PARENT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed and r['source_sha256']==entry['completion_sha256']==sha(ROOT/entry['completion_path'])
    assert r['noise_sha256']==sha(PARENT/f's{seed}'/'potential_noise.npz')
    clean,noise=(load_npz(ROOT/entry['completion_path'])['solutions'],load_npz(PARENT/f's{seed}'/'potential_noise.npz')) if raw is None else raw
    values=np.full((900,81),-1,np.int8);nrng=np.random.default_rng(p['noise_seed_base']+seed)
    for row in range(900):
        uniform=nrng.random();permutation=nrng.permutation(81).astype(np.uint8)
        assert uniform==noise['uniforms'][row] and np.array_equal(permutation,noise['permutations'][row])
        board=clean[row,permutation] if Fraction(float(uniform))<Fraction(9,160) else clean[row];values[row,given[row]]=board[given[row]]
    exposure=given[:,ii]&given[:,jj];joint=exposure.sum(0);equal=(exposure&(values[:,ii]==values[:,jj])).sum(0)
    observed=load_npz(d/'observations.npz')
    for key,value in dict(given=given,values=values,joint_count=joint,equal_count=equal).items():assert np.array_equal(observed[key],value)
    assert np.array_equal(joint,z['transfer_joint_count']) and np.array_equal(equal,z['transfer_equal_count'])
    gate=complete_label_gate(feedback,p['tie_seed_base']+seed,clean,noise,observed)
    baseline=load_npz(SNAPSHOT/f's{seed}'/'model.npz')
    for key,value in baseline.items():
        if key.startswith('frozen_') or key in ['aligned_mask','permuted_mask']:assert np.array_equal(z[key],value)
    rows=[];arm_audits={}
    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    for arm in ['transfer','frozen']:
        prefix=arm+'_';record=json.loads((d/'fit.json' if arm=='transfer' else SNAPSHOT/f's{seed}'/'fit.json').read_text())
        m,c=z[prefix+'joint_count'],z[prefix+'equal_count'];assert int(m.sum())==270000
        cost=independent_rate_costs(m,c);assert np.array_equal(cost,z[prefix+'cost'])
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
        if arm=='frozen':assert certified and candidate is not None

    previous=baseline['aligned_mask']|baseline['permuted_mask']|baseline['frozen_mask'];warmup=np.zeros_like(given);warmup[:300]=baseline['aligned_mask'][:300]
    assert np.array_equal(warmup,z['warmup_mask'])
    for arm,path,file in [('aligned',PARENT,'aligned-observations.npz'),('permuted',PARENT,'permuted-observations.npz'),('frozen',SNAPSHOT,'observations.npz')]:
        old=load_npz(path/f's{seed}'/file);overlap=old['given']&given;assert np.array_equal(old['values'][overlap],values[overlap])
    accounting=dict(recipient_queries=int(given.sum()),reused_queries=int((given&previous).sum()),new_queries=int((given&~previous).sum()),prior_h1bd_stream_union=int(previous.sum()),combined_h1bd_stream_union=int((given|previous).sum()),
                    donor_warmup_queries=int(warmup.sum()),deployment_overlap=int((given&warmup).sum()),deployment_union=int((given|warmup).sum()),new_calibration_queries=0)
    assert accounting==r['query_accounting'] and accounting['recipient_queries']==accounting['reused_queries']+accounting['new_queries']==22500
    assert accounting['donor_warmup_queries']==7500 and accounting['deployment_union']==30000-accounting['deployment_overlap']
    assert accounting['combined_h1bd_stream_union']==accounting['prior_h1bd_stream_union']+accounting['new_queries']
    result=dict(status='validated',seed=seed,donor=donor,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),model_sha256=sha(d/'model.npz'),
                rows=rows,arm_audits=arm_audits,query_accounting=accounting,complete_label_substitution_gate=gate,independent_masks_replayed=900,common_noise_rows_replayed=900,
                donor_prefix_rows=300,donor_recipient_source_overlap=0,donor_feedback_reconstructed_from_actual_observed_labels=True,all_masks_archived_before_recipient_labels=True)
    save(d/'audit.json',result);return result
