"""Independent H4ad paired corruption, unrestricted PL and graph audits."""
import argparse
import hashlib
import json
import os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[_key]='1'
import numpy as np
from heterogeneous_noise_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,PARENT,CONDITIONS,ARMS,sha,save,verify
from audit_pooled_source_study import numpy_objective


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {name:z[name].copy() for name in z.files}


def bytes_sha(array):return hashlib.sha256(array.tobytes()).hexdigest()


def numpy_raw_loss_gradient(w,b,labels,free):
    """Explicit directed pair gathers; the reverse direction uses W transpose."""
    w=np.asarray(w);b=np.asarray(b);labels=np.asarray(labels);free=np.asarray(free)
    assert w.shape==(3240,9,9) and b.shape==(81,9) and w.dtype==b.dtype==np.float64
    assert labels.ndim==2 and labels.shape[1]==81 and free.shape==labels.shape and free.dtype==bool
    assert np.isfinite(w).all() and np.isfinite(b).all() and ((labels>=0)&(labels<9)).all() and free.any()
    ii,jj=np.triu_indices(81,1);fields=np.broadcast_to(b,(len(labels),81,9)).copy()
    for edge,(left,right) in enumerate(zip(ii,jj)):
        matrix=w[edge]
        fields[:,left,:]+=matrix[:,labels[:,right]].T
        fields[:,right,:]+=matrix[labels[:,left],:]
    logits=-fields;peak=logits.max(2,keepdims=True);exponentials=np.exp(logits-peak)
    normalizer=exponentials.sum(2,keepdims=True);probability=exponentials/normalizer
    actual=np.take_along_axis(fields,labels[:,:,None].astype(np.int64),2)[:,:,0]
    loss=float((peak[:,:,0]+np.log(normalizer[:,:,0])+actual)[free].sum()/free.sum())
    indicators=np.eye(9,dtype=np.float64)[labels];residual=(indicators-probability)*free[:,:,None]/free.sum()
    dw=np.empty_like(w)
    for edge,(left,right) in enumerate(zip(ii,jj)):
        dw[edge]=residual[:,left,:].T@indicators[:,right,:]+indicators[:,left,:].T@residual[:,right,:]
    db=residual.sum(0)
    assert np.isfinite(loss) and np.isfinite(dw).all() and np.isfinite(db).all()
    return loss,dw,db


def independent_corruption(clean,free,uniforms,offsets,hot,condition):
    assert clean.shape==free.shape==uniforms.shape==offsets.shape and clean.shape[1]==81
    assert free.dtype==bool and (free.sum(1)>=9).all() and ((uniforms>0)&(uniforms<=1)).all()
    assert ((offsets>=1)&(offsets<=8)).all()
    weight=np.ones(81,np.float64)
    if condition=='hot18_weight8':weight[hot]=8
    else:assert condition=='uniform'
    cells=np.empty((len(clean),9),np.int64);noisy=clean.copy();indices=np.arange(81)
    for row in range(len(clean)):
        priority=-np.log(uniforms[row])/weight;priority[~free[row]]=np.inf
        selected=np.lexsort((indices,priority))[:9];cells[row]=selected
        noisy[row,selected]=(clean[row,selected].astype(np.int64)+offsets[row,selected])%9
    return noisy,cells


def sources():
    p=json.loads(PROTOCOL.read_text());verify(p)
    assert p['groups']==[list(range(g*4,g*4+4)) for g in range(16)] and p['conditions']==CONDITIONS and p['arms']==ARMS
    manifest=json.loads((OUT/'inputs.json').read_text());assert manifest['protocol_sha256']==sha(PROTOCOL)
    hot=np.sort(np.random.default_rng(p['hotspot_seed']).choice(81,18,replace=False))
    assert manifest['hot_cells']==hot.tolist() and len(set(hot))==18
    old=json.loads((PARENT/'inputs.json').read_text());prior=json.loads((PARENT/'source-audit.json').read_text())
    assert prior['status']=='validated' and prior['protocol_sha256']==old['protocol_sha256']==sha(HERE/'canonical_confirm.json')
    assert [e['bank'] for e in old['banks']]==list(range(64))
    assert len(manifest['entries'])==32
    expected_ids=[f'g{g:02d}-{condition}' for g in range(16) for condition in CONDITIONS]
    assert [e['job_id'] for e in manifest['entries']]==expected_ids
    evidence={};groups=[]
    for group,banks in enumerate(p['groups']):
        original={name:[] for name in ['solutions','free','source_row','source_bank','uniforms','offsets']}
        for bank in banks:
            e=old['banks'][bank];path=ROOT/e['path'];assert sha(path)==e['sha256']
            assert prior['input_sha256'][e['path']]==e['sha256'];evidence[e['path']]=e['sha256']
            z=load_npz(path);assert z['solutions'].shape==z['free'].shape==(600,81)
            assert z['solutions'].dtype==np.uint8 and z['free'].dtype==bool and z['source_row'].dtype==np.int64
            assert ((z['solutions']>=0)&(z['solutions']<9)).all() and (z['free'].sum(1)>=9).all()
            for name in ['solutions','free','source_row']:original[name].append(z[name])
            original['source_bank'].append(np.full(600,bank,np.int64))
            rng=np.random.default_rng(p['noise_rng_base']+bank)
            original['uniforms'].append(1-rng.random((600,81)))
            original['offsets'].append(rng.integers(1,9,(600,81),dtype=np.int64))
        expected={name:np.concatenate(values) for name,values in original.items()}
        paired=[]
        for condition in CONDITIONS:
            e=manifest['entries'][2*group+CONDITIONS.index(condition)]
            assert e['group']==group and e['condition']==condition and e['banks']==banks
            path=ROOT/e['path'];assert sha(path)==e['sha256'];evidence[e['path']]=e['sha256'];z=load_npz(path)
            assert set(z)==set(expected)|{'noisy','cells'}
            for name,array in expected.items():assert z[name].dtype==array.dtype and np.array_equal(z[name],array),(e['job_id'],name)
            noisy,cells=independent_corruption(z['solutions'],z['free'],z['uniforms'],z['offsets'],hot,condition)
            assert z['noisy'].dtype==np.uint8 and z['cells'].dtype==np.int64
            assert np.array_equal(noisy,z['noisy']) and np.array_equal(cells,z['cells'])
            changed=z['noisy']!=z['solutions'];assert (changed.sum(1)==9).all() and not changed[~z['free']].any()
            assert np.array_equal(np.take_along_axis(z['free'],z['cells'],1),np.ones((2400,9),bool))
            assert all(len(set(row))==9 for row in z['cells'])
            paired.append(dict(condition=condition,input_sha256=e['sha256'],rows=2400,changed_free_cells=21600))
        groups.append(dict(group=group,banks=banks,paired=paired,observation_instances_per_condition=2400,
                           unique_source_rows=int(len(np.unique(expected['source_row']))),
                           repeated_source_row_instances=2400-int(len(np.unique(expected['source_row'])))))
    queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==32 and [job['job_id'] for job in queue]==expected_ids and all(job['status']=='pending' for job in queue)
    for job,e in zip(queue,manifest['entries'],strict=True):assert job['bank']==e['group'] and job['condition']==e['condition']
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'inputs.json'),
                parent_source_audit_sha256=sha(PARENT/'source-audit.json'),hot_cells=hot.tolist(),groups=groups,
                input_sha256=evidence,paired_clean_observation_instances=38400,condition_observation_instances=76800,
                source_banks=64,new_source_rows=0,new_solver_inputs=0,all_repeated_instances_retained=True)
    save(OUT/'source-audit.json',result);print(json.dumps(dict(status='validated',groups=16,paired_entries=32,new_source_rows=0)))


def canonical_support(coefficients):
    ordered=sorted(float(x) for x in coefficients if x>0)
    if not ordered:return np.zeros(3240,bool),None
    n=len(ordered);median=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    cut=median/4
    return (coefficients>0)&(coefficients>=cut),cut


def count_support(noisy):
    ii,jj=np.triu_indices(81,1);counts=np.array([np.count_nonzero(noisy[:,i]==noisy[:,j]) for i,j in zip(ii,jj)],np.int64)
    ordered=sorted(int(x) for x in counts);median=(ordered[1619]+ordered[1620])/2
    distinct=sorted(set(ordered));gaps=[(distinct[k+1]-distinct[k],k) for k in range(len(distinct)-1) if distinct[k+1]<=median]
    index=max(gaps,key=lambda row:row[0])[1] if gaps else None
    cut=(distinct[index]+distinct[index+1])//2 if index is not None else 0
    return counts<=cut,counts,cut


def adamw_support(w):
    diagonal=np.stack([w[:,digit,digit] for digit in range(9)],1).mean(1)
    off=np.stack([w[:,a,b] for a in range(9) for b in range(9) if a!=b],1).mean(1)
    coeff=np.maximum(diagonal-off,0);positive=sorted(float(value) for value in coeff if value>0)
    if len(positive)<2:return np.zeros(3240,bool),coeff,dict(eligible=False,reason='insufficient finite positive coefficients')
    gaps=[b-a for a,b in zip(positive[:-1],positive[1:])];index=max(range(len(gaps)),key=lambda k:gaps[k])
    if gaps[index]<=0:return np.zeros(3240,bool),coeff,dict(eligible=False,reason='no positive coefficient gap')
    cut=(positive[index]+positive[index+1])/2;support=coeff>cut
    return support,coeff,dict(eligible=True,gap_index=index,gap=gaps[index],threshold=cut,
                             retained=int(support.sum()),minimum_threshold_distance=float(np.abs(coeff-cut).min()))


def audit_canonical_fit(j,fit,noisy,free):
    history=fit['history'];penalty=9*(1/30000)
    assert j.shape==(3240,) and j.dtype==np.float64 and np.isfinite(j).all()
    assert fit['updates']==len(history)==256 and fit['raw_lambda']==1/30000 and fit['canonical_penalty']==penalty
    assert fit['dtype']=='torch.float64' and 0<fit['seconds'] and np.isfinite(fit['seconds'])
    previous_loss=previous_objective=float(np.log(9));previous_step=540.;maximum_increase=-float('inf')
    for update,h in enumerate(history,1):
        assert set(h)=={'update','loss','objective','step','backtracks','step_linf','majorizer_gap'}
        assert h['update']==update and type(h['backtracks']) is int and 0<=h['backtracks']<32
        assert all(np.isfinite(value) for value in h.values()) and h['loss']>=0 and h['objective']>=h['loss']-1e-13 and h['step_linf']>=0
        assert h['step']==previous_step*(.5**h['backtracks']) and h['step']>0
        allowance=1e-12*(1+abs(previous_loss));increase=h['objective']-previous_objective
        assert h['majorizer_gap']<=allowance+1e-14 and increase<=allowance+1e-14
        maximum_increase=max(maximum_increase,increase)
        previous_loss=h['loss'];previous_objective=h['objective'];previous_step=h['step']
    loss,gradient=numpy_objective(j,noisy,free)
    residual=np.where(j>0,gradient+penalty,np.where(j<0,gradient-penalty,np.maximum(np.abs(gradient)-penalty,0)))
    objective=loss+penalty*float(np.abs(j).sum());kkt=float(np.abs(residual).max())
    assert abs(loss-history[-1]['loss'])<1e-12 and abs(objective-history[-1]['objective'])<1e-12
    assert np.isfinite(fit['kkt_linf']) and abs(kkt-fit['kkt_linf'])<1e-12
    return gradient,dict(numpy_loss=loss,numpy_objective=objective,numpy_kkt_linf=kkt,
                         maximum_objective_increase=maximum_increase,updates=256,finite_budget_not_optimum_claim=True)


def outcomes():
    import torch
    torch.set_num_threads(1)
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['auditor_sha256']==sha(__file__)
    assert source['input_manifest_sha256']==sha(OUT/'inputs.json')
    for path,digest in source['input_sha256'].items():assert sha(ROOT/path)==digest
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==32 and all(job['status']=='complete' for job in queue),'Incomplete fits remain retained; no filtered outcome summary.'
    assert {job['job_id'] for job in queue}=={e['job_id'] for e in manifest['entries']}
    for gpu in range(4):
        gate=json.loads((OUT/f'gate-gpu{gpu}.json').read_text())
        assert gate['status']=='passed' and gate['gpu']==gpu and gate['protocol_sha256']==sha(PROTOCOL)
        assert [case['condition'] for case in gate['cases']]==CONDITIONS
        for case in gate['cases']:
            e=next(e for e in manifest['entries'] if e['group']==gpu and e['condition']==case['condition'])
            assert case['input_sha256']==e['sha256'] and len(case['canonical'])==3
            for test in case['canonical']:assert 0<=test['loss_error']<1e-12 and 0<=test['gradient_error']<1e-12
            raw=case['raw'];assert raw['rows']==17 and 0<=raw['loss_error']<2e-6
            assert 0<=raw['w_gradient_error']<2e-7 and 0<=raw['b_gradient_error']<2e-7
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    hot=np.zeros(81,bool);hot[manifest['hot_cells']]=True
    categories={'hot_hot':hot[ii]&hot[jj],'hot_cold':hot[ii]^hot[jj],'cold_cold':~hot[ii]&~hot[jj]}
    assert sum(int(value.sum()) for value in categories.values())==3240
    rows=[];fits=[];diagnostics=[];evidence=dict(source['input_sha256']);public_arrays={name:[] for name in
        ['supports','adamw_coefficients','canonical_coefficients','canonical_gradient','counts','free_counts','corruption_counts']}
    for e in manifest['entries']:
        job=next(job for job in queue if job['job_id']==e['job_id']);d=OUT/'fits'/e['job_id']
        data=load_npz(ROOT/e['path']);models=load_npz(d/'models.npz');receipt=json.loads((d/'receipt.json').read_text())
        assert receipt['group']==job['bank']==e['group'] and receipt['condition']==job['condition']==e['condition']
        assert receipt['gpu']==job['gpu'] and 0<=receipt['gpu']<4
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['input_sha256']==e['sha256']
        assert 0<receipt['wall_seconds']<p['budget']['max_group_wall_seconds']
        assert set(receipt['output_sha256'])=={'adamw.pt','models.npz','canonical-fit.json'}
        for name,digest in receipt['output_sha256'].items():assert sha(d/name)==digest;evidence[str((d/name).relative_to(ROOT))]=digest
        evidence[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        assert set(models)=={'adamw_coefficients','adamw_support','canonical_coefficients','canonical_support','counts','restricted_support'}
        for name,array in models.items():
            assert array.shape==(3240,)
            if name.endswith('support'):assert array.dtype==bool
            elif name=='counts':assert array.dtype==np.int64
            else:assert array.dtype==np.float64 and np.isfinite(array).all()
        checkpoint=torch.load(d/'adamw.pt',map_location='cpu',weights_only=True)
        assert set(checkpoint)=={'W','B'} and checkpoint['W'].shape==(3240,9,9) and checkpoint['B'].shape==(81,9)
        for value in checkpoint.values():assert value.dtype==torch.float32 and torch.isfinite(value).all()
        w=checkpoint['W'].numpy().astype(np.float64);adamw,coeff,meta=adamw_support(w)
        assert np.max(np.abs(coeff-models['adamw_coefficients']))<1e-14 and np.array_equal(adamw,models['adamw_support'])
        actual=receipt['adamw'];assert actual['eligible']==meta['eligible']
        for name,value in meta.items():
            if isinstance(value,float):assert abs(actual[name]-value)<1e-12
            else:assert actual[name]==value
        assert actual['augmentation_seed']==p['augmentation_rng_base']+e['group']
        assert len(actual['history'])==32 and all(np.isfinite(x) and x>=0 for x in actual['history'])
        assert abs(actual['history'][0]-float(np.log(9)))<1e-6 and 0<actual['loop_seconds']<=receipt['wall_seconds']
        fit=json.loads((d/'canonical-fit.json').read_text());j=models['canonical_coefficients']
        gradient,fit_audit=audit_canonical_fit(j,fit,data['noisy'],data['free'])
        assert fit['seconds']<=receipt['wall_seconds'] and fit['seconds']+actual['loop_seconds']<=receipt['wall_seconds']+1e-6
        canonical,cut=canonical_support(j);assert np.array_equal(canonical,models['canonical_support'])
        assert receipt['canonical_eligible']==(cut is not None) and receipt['canonical_threshold']==cut
        restricted,counts,count_cut=count_support(data['noisy'])
        assert np.array_equal(counts,models['counts']) and np.array_equal(restricted,models['restricted_support'])
        assert count_cut==receipt['restricted_threshold']
        supports=np.stack([adamw,canonical,restricted]);changed=data['noisy']!=data['solutions']
        free_counts=data['free'].sum(0,dtype=np.int64);corruption_counts=changed.sum(0,dtype=np.int64)
        assert int(corruption_counts.sum())==21600 and np.all(corruption_counts<=free_counts)
        for name,array in [('supports',supports),('adamw_coefficients',models['adamw_coefficients']),('canonical_coefficients',j),
                           ('canonical_gradient',gradient),('counts',counts),('free_counts',free_counts),('corruption_counts',corruption_counts)]:public_arrays[name].append(array)
        fits.append(dict(job_id=e['job_id'],group=e['group'],condition=e['condition'],banks=e['banks'],
                         canonical_threshold=cut,restricted_threshold=count_cut,adamw_metadata=actual,
                         wall_seconds=receipt['wall_seconds'],receipt_sha256=sha(d/'receipt.json'),**fit_audit))
        true_max=int(counts[truth].max());nonpeer_min=int(counts[~truth].min())
        diagnostics.append(dict(job_id=e['job_id'],group=e['group'],condition=e['condition'],
                                true_count_min=int(counts[truth].min()),true_count_max=true_max,
                                nonpeer_count_min=nonpeer_min,nonpeer_count_max=int(counts[~truth].max()),
                                any_low_count_scalar_threshold_separates=true_max<nonpeer_min,
                                separation_margin=nonpeer_min-true_max,
                                hot_free_exposures=int(free_counts[hot].sum()),hot_corruptions=int(corruption_counts[hot].sum()),
                                cold_free_exposures=int(free_counts[~hot].sum()),cold_corruptions=int(corruption_counts[~hot].sum())))
        eligibility=[bool(meta['eligible']),cut is not None,True]
        for arm,support,eligible in zip(ARMS,supports,eligibility,strict=True):
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            stratified={name:dict(missing_true=int((mask&truth&~support).sum()),extra=int((mask&~truth&support).sum()),
                                   true_pairs=int((mask&truth).sum()),nonpeer_pairs=int((mask&~truth).sum())) for name,mask in categories.items()}
            assert sum(r['missing_true'] for r in stratified.values())==missing and sum(r['extra'] for r in stratified.values())==extra
            rows.append(dict(group=e['group'],condition=e['condition'],arm=arm,job_id=e['job_id'],banks=e['banks'],
                             eligible=eligible,retained=int(support.sum()),missing_true=missing,extra=extra,all_true_retained=missing==0,
                             exact_graph=missing==extra==0,support_sha256=bytes_sha(support),hot_categories=stratified))
    grouped=[]
    for condition in CONDITIONS:
        for arm in ARMS:
            part=[row for row in rows if row['condition']==condition and row['arm']==arm];assert len(part)==16
            grouped.append(dict(condition=condition,arm=arm,exact_graphs=sum(row['exact_graph'] for row in part),
                                eligible_groups=sum(row['eligible'] for row in part),ineligible_groups=sum(not row['eligible'] for row in part),
                                all_true_retaining_groups=sum(row['all_true_retained'] for row in part),
                                missing_true_total=sum(row['missing_true'] for row in part),extra_total=sum(row['extra'] for row in part)))
    target=next(g for g in grouped if g['condition']=='hot18_weight8' and g['arm']=='restricted')
    uniform=next(g for g in grouped if g['condition']=='uniform' and g['arm']=='restricted')
    success=target['exact_graphs']>=15 and target['missing_true_total']==0
    uniform_success=uniform['exact_graphs']>=15 and uniform['missing_true_total']==0
    primary=dict(decision='supported' if success else 'falsified',exact_graphs=target['exact_graphs'],missing_true_total=target['missing_true_total'],
                 uniform_control_target_met=uniform_success,heterogeneous_only_failure=(not success) and uniform_success,
                 attribution='Conditional on this fixed heat map and paired finite banks; if both targets fail, no heterogeneous-only attribution.')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PUBLIC/'models.npz',**{name:np.stack(values) for name,values in public_arrays.items()},
                        groups=np.array([e['group'] for e in manifest['entries']],np.int64),conditions=np.array([e['condition'] for e in manifest['entries']]),
                        arms=np.array(ARMS),hot_cells=np.array(manifest['hot_cells'],np.int64))
    launch={}
    for phase in ['gate','fit']:
        record=json.loads((OUT/f'{phase}-launch.json').read_text());assert record['status']=='complete' and record['returncodes']==[0,0,0,0]
        assert record['failure'] is None and 0<record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds']
        assert np.isfinite(record['aggregate_cpu_seconds']) and record['aggregate_cpu_seconds']>=0;launch[phase]=record
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),scope=p['scope'],primary=primary,
                grouped=grouped,rows=rows,fit_audits=fits,diagnostics=diagnostics,source_audit=source,input_sha256=evidence,
                models_sha256=sha(PUBLIC/'models.npz'),launch=launch,all_groups_retained=True,source_banks=64,groups=16,conditions=2,
                logical_graphs=96,paired_clean_observation_instances=38400,condition_observation_instances=76800,
                new_source_rows=0,new_solver_inputs=0,inference_gpu_seconds=0,optimizer_scope='Finite BP32 and canonical256 recipes; descriptive controls, no matched-compute or optimum claim.')
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(status='validated',primary=primary,grouped=grouped)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','outcomes']);globals()[ap.parse_args().action]()
