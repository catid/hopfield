"""Independent H4ac source concatenation, conditional PL and graph audit."""
import argparse
import hashlib
import json
import os
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[_name]='1'
import numpy as np
from pooled_source_study import ROOT,HERE,RIPPER,OUT,PUBLIC,PROTOCOL,PARENT,MODELS,sha,save,verify

ARMS=['adamw_vote3','canonical_vote3','canonical_pooled','restricted_pooled']


def bytes_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def numpy_objective(j,noisy,free):
    """Unpenalized conditional loss and pair gradient, independently in FP64.

    Derivative for J_ij is the sum of the two directed conditional residuals.
    This uses label gathers rather than the optimizer's one-hot contractions.
    """
    j=np.asarray(j);labels=np.asarray(noisy);mask=np.asarray(free)
    assert j.dtype==np.float64 and j.shape==(3240,) and np.isfinite(j).all()
    assert labels.ndim==2 and labels.shape[1]==81 and labels.dtype.kind in 'iu'
    assert mask.shape==labels.shape and mask.dtype==bool and mask.any()
    assert ((labels>=0)&(labels<9)).all()
    ii,jj=np.triu_indices(81,1);matrix=np.zeros((81,81),np.float64)
    matrix[ii,jj]=j;matrix[jj,ii]=j
    # Each candidate field counts the weights to neighbors currently bearing
    # that candidate label, regardless of whether those neighbors are free.
    fields=np.stack([(labels==digit).astype(np.float64)@matrix.T for digit in range(9)],axis=2)
    logits=-fields;shift=logits.max(axis=2,keepdims=True)
    exponential=np.exp(logits-shift);normalizer=exponential.sum(axis=2,keepdims=True)
    probabilities=exponential/normalizer
    log_normalizer=shift[:,:,0]+np.log(normalizer[:,:,0])
    actual_field=np.take_along_axis(fields,labels[:,:,None].astype(np.int64),axis=2)[:,:,0]
    denominator=int(mask.sum());loss=float((log_normalizer+actual_field)[mask].sum()/denominator)
    gradient=np.empty(3240,np.float64);rows=np.arange(len(labels))[:,None]
    for start in range(0,3240,128):
        end=min(start+128,3240);left,right=ii[start:end],jj[start:end]
        same=labels[:,left]==labels[:,right]
        at_left=probabilities[rows,left[None,:],labels[:,right]]
        at_right=probabilities[rows,right[None,:],labels[:,left]]
        terms=(same-at_left)*mask[:,left]+(same-at_right)*mask[:,right]
        gradient[start:end]=terms.sum(axis=0)/denominator
    assert np.isfinite(loss)and np.isfinite(gradient).all()
    return loss,gradient


def strong_support(j):
    assert j.shape==(3240,) and j.dtype==np.float64 and np.isfinite(j).all()
    positives=sorted(float(value)for value in j if value>0);n=len(positives)
    assert n>0,'No positive coefficient: pooled threshold is ineligible; retain failure, do not filter group.'
    median=positives[n//2]if n%2 else(positives[n//2-1]+positives[n//2])/2
    cut=median/4
    return np.array([value>0 and value>=cut for value in j],bool),cut


def restricted_support(noisy):
    ii,jj=np.triu_indices(81,1)
    counts=np.array([np.count_nonzero(noisy[:,i]==noisy[:,j])for i,j in zip(ii,jj)],np.int64)
    ordered=sorted(int(x)for x in counts);median=(ordered[1619]+ordered[1620])/2
    distinct=sorted(set(ordered));candidates=[(distinct[k+1]-distinct[k],k)for k in range(len(distinct)-1)if distinct[k+1]<=median]
    if candidates:
        # max chooses the first largest gap, matching the fixed lower-tail rule.
        index=max(candidates,key=lambda row:row[0])[1];cut=(distinct[index]+distinct[index+1])//2
    else:cut=0
    return counts<=cut,counts,cut


def load_public_models():
    with np.load(MODELS,allow_pickle=False)as z:
        result={name:z[name].copy()for name in ['canonical_weak_coefficients','canonical_weak_support','adamw_support','adamw_coefficients']}
    assert result['canonical_weak_coefficients'].shape==(64,3240)and result['canonical_weak_coefficients'].dtype==np.float64
    assert result['adamw_coefficients'].shape==(64,3240)and result['adamw_coefficients'].dtype==np.float64
    for name in ['canonical_weak_support','adamw_support']:
        assert result[name].shape==(64,3240)and result[name].dtype==bool
    assert np.isfinite(result['canonical_weak_coefficients']).all()and np.isfinite(result['adamw_coefficients']).all()
    assert np.array_equal(result['canonical_weak_support'],result['canonical_weak_coefficients']>0)
    return result


def sources():
    p=json.loads(PROTOCOL.read_text());verify(p)
    assert p['groups']==[list(range(4*g,4*g+4))for g in range(16)]and p['arms']==ARMS
    manifest=json.loads((OUT/'inputs.json').read_text());assert manifest['protocol_sha256']==sha(PROTOCOL)
    assert len(manifest['groups'])==16
    old=json.loads((PARENT/'inputs.json').read_text());prior=json.loads((PARENT/'source-audit.json').read_text())
    assert old['protocol_sha256']==prior['protocol_sha256']==sha(HERE/'canonical_confirm.json')and prior['status']=='validated'
    assert [r['bank']for r in old['banks']]==list(range(64))
    strong_parent=json.loads((ROOT/'results/research/foureyes/strong_support_study/summary.json').read_text());assert strong_parent['status']=='validated'
    models=load_public_models();evidence={};raw={};individual=[]
    for bank,e in enumerate(old['banks']):
        path=ROOT/e['path'];assert e['bank']==bank and sha(path)==e['sha256'];evidence[e['path']]=sha(path)
        with np.load(path,allow_pickle=False)as z:raw[bank]={name:z[name].copy()for name in ['noisy','free','source_row']}
        assert raw[bank]['noisy'].shape==raw[bank]['free'].shape==(600,81)
        assert raw[bank]['noisy'].dtype==np.uint8 and raw[bank]['free'].dtype==bool
        assert raw[bank]['source_row'].shape==(600,)and raw[bank]['source_row'].dtype==np.int64
        assert ((raw[bank]['noisy']>=0)&(raw[bank]['noisy']<9)).all()and raw[bank]['free'].any()
        original_model=PARENT/'fits'/f'b{bank:02d}'/'models.npz'
        assert sha(original_model)==prior['input_sha256'][str(original_model.relative_to(ROOT))]
        evidence[str(original_model.relative_to(ROOT))]=sha(original_model)
        with np.load(original_model,allow_pickle=False)as z:
            for name,array in models.items():assert np.array_equal(array[bank],z[name]),(bank,name)
        support,cut=strong_support(models['canonical_weak_coefficients'][bank])
        prior_row=next(r for r in strong_parent['rows']if r['bank']==bank and r['arm']=='strongonly')
        assert bytes_sha(support.astype(np.uint16)*256)==prior_row['weight256_sha256']
        individual.append(dict(bank=bank,threshold=cut,strong_support_sha256=bytes_sha(support),adamw_support_sha256=bytes_sha(models['adamw_support'][bank]),raw_input_sha256=e['sha256'],raw_models_sha256=sha(original_model)))
    group_rows=[]
    for group,e in enumerate(manifest['groups']):
        banks=list(range(4*group,4*group+4));assert e['group']==group and e['banks']==banks
        path=ROOT/e['path'];assert sha(path)==e['sha256'];evidence[e['path']]=sha(path)
        with np.load(path,allow_pickle=False)as z:
            assert set(z.files)=={'noisy','free','source_row','source_bank'}
            for name in ['noisy','free','source_row']:
                expected=np.concatenate([raw[b][name]for b in banks],axis=0)
                assert z[name].dtype==expected.dtype and z[name].shape==expected.shape and np.array_equal(z[name],expected)
            expected_banks=np.array([b for b in banks for _ in range(600)],np.int64)
            assert z['source_bank'].dtype==np.int64 and np.array_equal(z['source_bank'],expected_banks)
            assert len(z['source_row'])==2400
            group_rows.append(dict(group=group,banks=banks,observations=2400,unique_source_rows=int(len(np.unique(z['source_row']))),repeated_source_row_instances=2400-int(len(np.unique(z['source_row']))),source_row_bytes_sha256=bytes_sha(z['source_row']),source_bank_bytes_sha256=bytes_sha(z['source_bank']),pooled_input_sha256=e['sha256']))
    queue=json.loads((OUT/'queue.json').read_text());assert len(queue)==16
    assert {j['job_id']for j in queue}=={f'g{g:02d}'for g in range(16)}and sorted(j['bank']for j in queue)==list(range(16))
    for j in queue:assert j['job_id']==f'g{j["bank"]:02d}'and j['status']=='pending'
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'inputs.json'),parent_source_audit_sha256=sha(PARENT/'source-audit.json'),public_models_sha256=sha(MODELS),groups=group_rows,original_banks=individual,input_sha256=evidence,source_banks=64,pooled_groups=16,observation_instances=38400,all_repeated_instances_retained=True,new_source_rows=0)
    save(OUT/'source-audit.json',result);print(json.dumps(dict(status='validated',groups=16,source_banks=64,observation_instances=38400)))


def outcomes():
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated'and source['protocol_sha256']==sha(PROTOCOL)and source['auditor_sha256']==sha(__file__)
    assert source['input_manifest_sha256']==sha(OUT/'inputs.json')and source['public_models_sha256']==sha(MODELS)
    for path,digest in source['input_sha256'].items():assert sha(ROOT/path)==digest
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==16 and sorted(j['bank']for j in queue)==list(range(16))and all(j['status']=='complete'for j in queue),'Incomplete or failed fits stay retained; no filtered outcome summary.'
    for gpu in range(4):
        gate=json.loads((OUT/f'gate-gpu{gpu}.json').read_text())
        assert gate['status']=='passed'and gate['gpu']==gpu and gate['protocol_sha256']==sha(PROTOCOL)and gate['rows']==2400 and len(gate['cases'])==3
        for case in gate['cases']:assert 0<=case['loss_error']<1e-12 and 0<=case['gradient_error']<1e-12
    original=load_public_models();individual=np.stack([strong_support(j)[0]for j in original['canonical_weak_coefficients']])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    matrices=[];pooled_coefficients=[];pooled_gradients=[];canonical_votes=[];adamw_votes=[];pooled_counts=[];rows=[];fits=[];evidence=dict(source['input_sha256'])
    for group,e in enumerate(manifest['groups']):
        banks=list(range(4*group,4*group+4));assert e['group']==group and e['banks']==banks
        path=ROOT/e['path'];assert sha(path)==e['sha256']
        with np.load(path,allow_pickle=False)as z:noisy=z['noisy'].copy();free=z['free'].copy()
        d=OUT/'fits'/f'g{group:02d}';receipt=json.loads((d/'receipt.json').read_text())
        assert receipt['group']==group and receipt['protocol_sha256']==sha(PROTOCOL)and receipt['input_sha256']==e['sha256']
        job=next(j for j in queue if j['bank']==group);assert job['job_id']==f'g{group:02d}'and job['gpu']==receipt['gpu']and 0<=receipt['gpu']<4
        assert receipt['finite']and 0<receipt['wall_seconds']<p['budget']['max_group_wall_seconds']
        assert set(receipt['output_sha256'])=={'model.npz','fit.json'}
        for name,digest in receipt['output_sha256'].items():assert sha(d/name)==digest;evidence[str((d/name).relative_to(ROOT))]=digest
        evidence[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        with np.load(d/'model.npz',allow_pickle=False)as z:
            assert set(z.files)=={'pooled_coefficients'};j=z['pooled_coefficients'].copy()
        assert j.shape==(3240,)and j.dtype==np.float64 and np.isfinite(j).all()and receipt['positive']==int((j>0).sum())
        fit=json.loads((d/'fit.json').read_text());history=fit['history']
        assert fit['updates']==len(history)==256 and fit['raw_lambda']==1/30000 and fit['canonical_penalty']==9*(1/30000) and fit['dtype']=='torch.float64'
        assert np.isfinite(fit['seconds'])and 0<fit['seconds']<=receipt['wall_seconds']+1e-9
        previous_loss=previous_objective=float(np.log(9));previous_step=540.
        maximum_increase=-float('inf')
        for update,h in enumerate(history,1):
            assert set(h)=={'update','loss','objective','step','backtracks','step_linf','majorizer_gap'}
            assert h['update']==update and type(h['backtracks'])is int and 0<=h['backtracks']<32
            assert all(np.isfinite(v)for v in h.values())and h['loss']>=0 and h['objective']>=h['loss']-1e-13 and h['step_linf']>=0
            assert h['step']==previous_step*(.5**h['backtracks'])and h['step']>0
            allowance=1e-12*(1+abs(previous_loss))
            assert h['majorizer_gap']<=allowance+1e-14 and h['objective']-previous_objective<=allowance+1e-14
            maximum_increase=max(maximum_increase,h['objective']-previous_objective)
            previous_loss=h['loss'];previous_objective=h['objective'];previous_step=h['step']
        loss,gradient=numpy_objective(j,noisy,free);penalty=9*(1/30000)
        residual=np.where(j>0,gradient+penalty,np.where(j<0,gradient-penalty,np.maximum(np.abs(gradient)-penalty,0)))
        kkt=float(np.abs(residual).max());objective=loss+penalty*float(np.abs(j).sum())
        assert abs(loss-history[-1]['loss'])<1e-12 and abs(objective-history[-1]['objective'])<1e-12
        assert np.isfinite(fit['kkt_linf'])and abs(kkt-fit['kkt_linf'])<1e-12
        pooled,cut=strong_support(j);restricted,counts,count_cut=restricted_support(noisy)
        cv=individual[banks].sum(axis=0,dtype=np.uint8);av=original['adamw_support'][banks].sum(axis=0,dtype=np.uint8)
        supports=np.stack([av>=3,cv>=3,pooled,restricted]);assert supports.dtype==bool and supports.shape==(4,3240)
        matrices.append(supports);pooled_coefficients.append(j);pooled_gradients.append(gradient);pooled_counts.append(counts);canonical_votes.append(cv);adamw_votes.append(av)
        fits.append(dict(group=group,banks=banks,receipt_sha256=sha(d/'receipt.json'),model_sha256=sha(d/'model.npz'),fit_sha256=sha(d/'fit.json'),numpy_loss=loss,numpy_objective=objective,numpy_gradient_linf=float(np.abs(gradient).max()),numpy_kkt_linf=kkt,pooled_threshold=cut,restricted_count_threshold=count_cut,updates=256,maximum_objective_increase=maximum_increase,wall_seconds=receipt['wall_seconds'],scope='Finite256 updates; KKT residual reported, no optimum claim.'))
        for arm,support in zip(ARMS,supports):
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            rows.append(dict(group=group,banks=banks,arm=arm,retained=int(support.sum()),missing_true=missing,extra=extra,all_true_retained=missing==0,exact_graph=missing==extra==0,support_sha256=bytes_sha(support)))
    grouped=[]
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm];assert len(selected)==16
        grouped.append(dict(arm=arm,exact_graphs=sum(r['exact_graph']for r in selected),all_true_retaining_groups=sum(r['all_true_retained']for r in selected),missing_true_total=sum(r['missing_true']for r in selected),extra_total=sum(r['extra']for r in selected)))
    primary_group=grouped[1];primary=dict(exact_graphs=primary_group['exact_graphs'],missing_true_total=primary_group['missing_true_total'],decision='supported'if primary_group['exact_graphs']>=15 and primary_group['missing_true_total']==0 else'falsified',scope='Fixed16 disjoint quartets; finite target, no fresh-source or population-confidence claim.')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PUBLIC/'models.npz',groups=np.asarray(p['groups'],np.int64),arms=np.array(ARMS),supports=np.stack(matrices),pooled_coefficients=np.stack(pooled_coefficients),pooled_gradient=np.stack(pooled_gradients),pooled_counts=np.stack(pooled_counts),canonical_vote_counts=np.stack(canonical_votes),adamw_vote_counts=np.stack(adamw_votes))
    launch={}
    for phase in ['gate','fit']:
        record=json.loads((OUT/f'{phase}-launch.json').read_text());assert record['status']=='complete'and record['returncodes']==[0,0,0,0]
        assert record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds'];launch[phase]=record
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=grouped,rows=rows,pooled_fit_audits=fits,source_audit=source,input_sha256=evidence,models_sha256=sha(PUBLIC/'models.npz'),source_banks=64,groups=16,logical_graphs=64,observation_instances=38400,new_source_rows=0,new_solver_inputs=0,inference_gpu_seconds=0,launch=launch,all_groups_retained=True,optimizer_scope='Finite256-update canonical pooled control, not tuned-compute or optimum comparison.')
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(status='validated',primary=primary,grouped=grouped)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['sources','outcomes']);globals()[parser.parse_args().action]()
