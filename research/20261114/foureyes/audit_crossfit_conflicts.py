"""H4ax independent opposite-fold replay, exact certificates, and endpoint bounds."""
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
from crossfit_conflicts import ROOT,HERE,OBS,DETECTOR,PARENT_PROTOCOL,OUT,PUBLIC,PROTOCOL,ARMS,FAMILIES,sha,read,verify,check_seal
from crossfit_conflict_math import independent_fold_costs,independent_crossfit_mask
from corruption_censor_math import independent_score
from audit_conflict_mask import audit_bank as audit_parent
from audit_degree_prior import check_certificate
from audit_objective_gap import reconstruct


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(256))
    assert sum(r['baseline_exact'] for r in rows)==129 and sum(r['parent_detector_exact'] for r in rows)==130
    counts={k:sum(r['classification']==k for r in rows) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']}
    lo,hi=counts['success'],counts['success']+counts['unresolved']
    losses=sum(r['parent_detector_exact'] and r['classification'].startswith('resolved_failure') for r in rows)
    uncertain=sum(r['parent_detector_exact'] and r['classification']=='unresolved' for r in rows)
    primary=dict(exact_recovery=dict(lower=lo,upper=hi,required=162,total=256,
        decision='supported' if lo>=162 else 'falsified' if hi<162 else 'inconclusive'),
        detector_regressions=dict(lower=losses,upper=losses+uncertain,maximum=8,total=130,
        decision='supported' if losses+uncertain<=8 else 'falsified' if losses>8 else 'inconclusive'))
    group=dict(arm='crossfit',**counts,
        gains_vs_detector=sum(not r['parent_detector_exact'] and r['classification']=='success' for r in rows),losses_vs_detector=losses,
        gains_vs_original=sum(not r['baseline_exact'] and r['classification']=='success' for r in rows),
        losses_vs_original=sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in rows),
        masks_constructed=sum(r['mask_constructed'] for r in rows),
        **{key:sum(r[key] for r in rows if r['mask_constructed']) for key in ['true_positive_locations','false_positive_locations','false_negative_locations']},
        families=[dict(family=f,exact=sum(r['family']==f and r['classification']=='success' for r in rows),unresolved=sum(r['family']==f and r['classification']=='unresolved' for r in rows)) for f in FAMILIES])
    count=sum(sum(f['certified'] for f in r['folds'])+r['certified'] for r in rows)
    return dict(primary=primary,grouped=[group],new_certificates_checked=count,new_mutations_rejected=3*count,all256_banks_retained=True)


def audit_fit(a,rec,cost,degree,index,phase,p,truth):
    assert a['cost'].dtype==np.int64 and np.array_equal(a['cost'],cost)
    meta=rec['solver'];ii,jj=np.triu_indices(81,1)
    for key,value in p['solver'].items():
        if key=='highspy':assert meta['highs_version']==value
        elif key=='numpy':assert meta['numpy']==value
        else:assert meta['solver_options']['random_seed' if key=='random_seed_base' else key]==(value+3*index+phase if key=='random_seed_base' else value)
    proof=[a[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    valid=check_certificate(81,degree,ii,jj,cost,*proof);assert valid==rec['certificate']['certified']
    result=dict(certified=valid,classification='unresolved',integral=None,objective_twice=None,true_gap_twice=None,missing_true=None,extra=None)
    if valid:
        assert meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']
        assert a['primal_values'].dtype==a['node_dual_values'].dtype==np.float64
        assert np.isin(a['primal_values'],[0.,.5,1.]).all() and np.array_equal(2*a['primal_values'],proof[0])
        assert np.max(abs(2*a['node_dual_values']-proof[1]))<=1e-6
        objective=sum(int(c)*int(x) for c,x in zip(cost,proof[0]));cert=rec['certificate']
        assert cert['primal_objective_twice']==cert['dual_objective_twice']==objective
        assert abs(meta['solver_objective']-objective/2)<1e-5 and cert['primal_rounding_used'] is False
        integral=bool(np.isin(proof[0],[0,2]).all());assert integral==cert['integral']
        missing=int(np.count_nonzero(truth&(proof[0]!=2))) if integral else None
        extra=int(np.count_nonzero(~truth&(proof[0]==2))) if integral else None
        classification=('success' if missing==extra==0 else 'resolved_failure_nonexact') if integral else 'resolved_failure_fractional'
        gap=2*sum(map(int,cost[truth]))-objective;assert gap>=0
        for which in range(3):
            bad=[v.copy() for v in proof];bad[which][0]+=1
            assert not check_certificate(81,degree,ii,jj,cost,*bad)
        result.update(classification=classification,integral=integral,objective_twice=objective,true_gap_twice=gap,missing_true=missing,extra=extra)
    return result


def audit_bank(index,p,public=False):
    base=PUBLIC if public else OUT;folder=base/f'b{index:03d}';seal=read(base/'fit-seal.json');foldseal=read(base/'fold-seal.json')
    assert seal['protocol_sha256']==foldseal['protocol_sha256']==sha(PROTOCOL)
    arrays=[];receipts=[]
    for arm in ARMS:
        rec=read(folder/f'{arm}.json')
        assert rec['fit_completed_monotonic_ns']<=seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns']
        assert rec['index']==index and rec['arm']==arm and rec['protocol_sha256']==sha(PROTOCOL)
        assert rec['models_sha256']==sha(folder/f'{arm}.npz')
        if not public and rec['executed']:assert rec['log_sha256']==sha(folder/f'{arm}.log')
        with np.load(folder/f'{arm}.npz',allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        receipts.append(rec)
    assert all(r['executed'] is True for r in receipts[:2])
    assert max(r['fit_completed_monotonic_ns'] for r in receipts[:2])<=foldseal['latest_fit_monotonic_ns']<foldseal['sealed_monotonic_ns']<receipts[2]['fit_started_monotonic_ns']
    assert receipts[2]['fold_seal_sha256']==sha(base/'fold-seal.json')
    # All folds/final outcomes are sealed before these ground-truth diagnostics.
    inherited=audit_parent(index,read(PARENT_PROTOCOL),public=True)
    assert inherited==read(DETECTOR/f'b{index:03d}/audit.json')
    old=read(HERE/'corruption-censor.json')['jobs'][index];job=p['jobs'][index]
    assert job=={k:old[k] for k in ['index','inferred_degree','parent_prefix_index']}
    degree=job['inferred_degree'];family=old['family'];group=old['group']
    assert all(r['degree']==degree for r in receipts)
    with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:noisy,truth_mask=z['noisy'],z['mask']
    name='varying_degree' if family in ['cycle81','cliques9'] else 'cycle_power';gi=2*group+int(family in ['cliques9','power3']);ii,jj=np.triu_indices(81,1)
    with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz') as z:
        graph=reconstruct(z['orders'][gi],family);truth=graph[ii,jj]
        assert np.array_equal(truth,z['truths'][gi]) and (graph.sum(0)==degree).all()
    costs=independent_fold_costs(noisy)
    assert np.array_equal(costs[0]+costs[1],independent_score(noisy,np.zeros_like(noisy,bool))['cost'])
    folds=[audit_fit(arrays[k],receipts[k],costs[k],degree,index,k,p,truth) for k in [0,1]]
    eligible=all(f['certified'] for f in folds);assert receipts[2]['executed']==eligible
    final=arrays[2]
    row=dict(index=index,arm='crossfit',family=family,degree=degree,baseline_exact=old['baseline_exact'],
        parent_detector_exact=inherited['rows'][0]['classification']=='success',folds=folds,attempted_LPs=2+int(eligible),mask_constructed=eligible)
    if eligible:
        scores,mask=independent_crossfit_mask(noisy,arrays[0]['primal_twice'],arrays[1]['primal_twice'])
        assert final['scores'].dtype==np.int64 and np.array_equal(final['scores'],scores)
        assert final['mask'].dtype==bool and np.array_equal(final['mask'],mask) and (mask.sum(1)==9).all()
        result=independent_score(noisy,mask)
        for key,value in result.items():assert final[key].dtype==np.int64 and np.array_equal(final[key],value)
        row.update(audit_fit(final,receipts[2],result['cost'],degree,index,2,p,truth))
        tp=int(np.count_nonzero(mask&truth_mask));fp=int(np.count_nonzero(mask&~truth_mask));fn=int(np.count_nonzero(~mask&truth_mask));assert tp+fp==tp+fn==5400
        row.update(true_positive_locations=tp,false_positive_locations=fp,false_negative_locations=fn,
            flagged_per_cell=mask.sum(0,dtype=np.int64).tolist(),flagged_per_row=mask.sum(1,dtype=np.int64).tolist(),
            exposure_min=int(final['exposure'].min()),exposure_max=int(final['exposure'].max()),zero_exposure_pairs=int(np.count_nonzero(final['exposure']==0)),
            true_edges_nonzero_cost=int(np.count_nonzero(final['cost'][truth])),zero_support_missing_true=int(np.count_nonzero(truth&(final['cost']!=0))),zero_support_extra=int(np.count_nonzero(~truth&(final['cost']==0))))
    else:
        assert set(final)=={'unexecuted'} and final['unexecuted'].shape==() and final['unexecuted'].dtype==bool and final['unexecuted'].item()
        assert receipts[2]['reason']=='required_fold_uncertified' and receipts[2]['unresolved_folds']==[k for k,f in enumerate(folds) if not f['certified']]
        assert not receipts[2]['certificate']['certified']
        row.update(certified=False,classification='unresolved',true_positive_locations=None,false_positive_locations=None,false_negative_locations=None)
    files=[f'{a}.{e}' for a in ARMS for e in ['npz','json']]
    return dict(rows=[row],protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),fit_seal_sha256=sha(base/'fit-seal.json'),
        inherited_audit_sha256=sha(DETECTOR/f'b{index:03d}/audit.json'),artifact_sha256={name:sha(folder/name) for name in files})


def public_job(index):return audit_bank(index,read(PROTOCOL),public=True)


def main():
    started=time.perf_counter();p=read(PROTOCOL);verify(p);seal=check_seal(PUBLIC);saved=read(PUBLIC/'summary.json')
    assert saved['protocol_sha256']==sha(PROTOCOL)
    for rel,h in saved['artifact_sha256'].items():assert sha(PUBLIC/rel)==h
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(256)))
    for index,audit in enumerate(audits):
        assert audit==read(PUBLIC/f'b{index:03d}/audit.json')
        order=read(PUBLIC/f'b{index:03d}/audit-order.json')
        assert order['seal_sha256']==sha(PUBLIC/'fit-seal.json') and order['audit_started_monotonic_ns']>seal['sealed_monotonic_ns']
    rows=[r for a in audits for r in a['rows']];assert rows==saved['rows']
    for key,value in aggregate(rows).items():assert saved[key]==value
    assert saved['new_LPs']==sum(r['attempted_LPs'] for r in rows)
    assert saved['inherited_certificates_checked']==1024 and saved['inherited_mutations_rejected']==3072
    rt=read(PUBLIC/'runtime.json');assert rt['status']=='complete' and rt['returncodes']==[0]*24 and rt['error'] is None
    assert rt['wall_seconds']<p['budget']['wall_seconds'] and rt['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
    assert rt['samples'][-1]['completed_LPs']==saved['new_LPs'] and rt['samples'][-1]['audited_banks']==256
    assert all(s['audited_banks']==0 for s in rt['samples'] if s['phase'] in ['hashes','fold_worker','all_folds_sealed','final_worker','all_fits_sealed'])
    print(json.dumps(dict(status='passed',**aggregate(rows),wall_seconds=time.perf_counter()-started,all_fold_and_final_seals_verified=True)))


if __name__=='__main__':main()
