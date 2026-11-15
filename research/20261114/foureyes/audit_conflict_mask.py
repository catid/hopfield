"""Independent dense conflict scores, post-seal diagnostics and exact LP checks."""
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
from conflict_mask import ROOT,HERE,OBS,BASE,PARENT_PROTOCOL,OUT,PUBLIC,PROTOCOL,ARMS,FAMILIES,sha,read,verify,check_seal
from conflict_mask_math import independent_detector,independent_random_mask
from corruption_censor_math import independent_score
from audit_corruption_censor import audit_bank as audit_parent
from audit_degree_prior import check_certificate
from audit_objective_gap import reconstruct


def aggregate(rows):
    assert [(r['index'],r['arm']) for r in rows]==[(i,a) for i in range(256) for a in ARMS]
    groups=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm]
        assert sum(r['baseline_exact'] for r in part)==129
        groups.append(dict(arm=arm,**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
                      true_positive_locations=sum(r['true_positive_locations'] for r in part),
                      false_positive_locations=sum(r['false_positive_locations'] for r in part),
                      false_negative_locations=sum(r['false_negative_locations'] for r in part),
                      flagged_locations=256*600*9,actual_corrupt_locations=256*600*9,
                      gains_vs_baseline=sum(not r['baseline_exact'] and r['classification']=='success' for r in part),
                      losses_vs_baseline=sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in part),
                      families=[dict(family=f,exact=sum(r['family']==f and r['classification']=='success' for r in part),
                                     unresolved=sum(r['family']==f and r['classification']=='unresolved' for r in part)) for f in FAMILIES]))
    detector,random=groups
    lo,hi=detector['success'],detector['success']+detector['unresolved']
    alo=detector['success']-random['success']-random['unresolved'];ahi=detector['success']+detector['unresolved']-random['success']
    primary=dict(detector_exact=dict(lower=lo,upper=hi,required=200,total=256,
                 decision='supported' if lo>=200 else 'falsified' if hi<200 else 'inconclusive'),
                 advantage_over_random=dict(lower=alo,upper=ahi,required=32,total=256,
                 decision='supported' if alo>=32 else 'falsified' if ahi<32 else 'inconclusive'))
    return dict(primary=primary,grouped=groups,new_certificates_checked=sum(r['certified'] for r in rows),
                new_mutations_rejected=3*sum(r['certified'] for r in rows),all512_outputs_retained=True)


def audit_bank(index,p,public=False):
    folder=(PUBLIC if public else OUT)/f'b{index:03d}';base=PUBLIC if public else OUT
    seal=read(base/'fit-seal.json')
    assert seal['protocol_sha256']==sha(PROTOCOL) and seal['fits']==512
    for arm in ARMS:
        rec=read(folder/f'{arm}.json')
        assert rec['fit_completed_monotonic_ns']<=seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns']
    # All scientific fits are complete and sealed before these privileged/truth reads.
    parent=read(PARENT_PROTOCOL);inherited=audit_parent(index,parent,public=True)
    assert inherited==read(OBS/f'b{index:03d}/audit.json') and inherited['row']['classification']=='success'
    job=p['jobs'][index];old_job=parent['jobs'][index]
    assert job=={k:old_job[k] for k in ['index','inferred_degree','parent_prefix_index']}
    degree=job['inferred_degree'];family=old_job['family'];group=old_job['group']
    with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:noisy,truth_mask=z['noisy'],z['mask']
    with np.load(BASE/f"j{job['parent_prefix_index']:03d}/models.npz",allow_pickle=False) as z:primal=z[f'd{degree:02d}_primal_twice']
    scores,detector_mask=independent_detector(noisy,primal)
    with np.load(folder/'detector-scores.npz',allow_pickle=False) as z:
        assert z['scores'].dtype==np.int64 and np.array_equal(z['scores'],scores)
    random=independent_random_mask(600,81,p['random_seed_base']+index)
    name='varying_degree' if family in ['cycle81','cliques9'] else 'cycle_power';gi=2*group+int(family in ['cliques9','power3']);ii,jj=np.triu_indices(81,1)
    with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz') as z:
        graph=reconstruct(z['orders'][gi],family);truth=graph[ii,jj]
        assert np.array_equal(truth,z['truths'][gi]) and (graph.sum(0)==degree).all()
    rows=[]
    for ai,(arm,mask) in enumerate(zip(ARMS,[detector_mask,random])):
        with np.load(folder/f'{arm}.npz',allow_pickle=False) as z:a={key:z[key] for key in z.files}
        assert a['mask'].dtype==bool and np.array_equal(a['mask'],mask) and (mask.sum(1)==9).all()
        for key,value in independent_score(noisy,mask).items():
            assert a[key].dtype==np.int64 and np.array_equal(a[key],value)
        receipt=read(folder/f'{arm}.json')
        assert receipt['index']==index and receipt['arm']==arm and receipt['degree']==degree
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['models_sha256']==sha(folder/f'{arm}.npz')
        if not public:assert receipt['log_sha256']==sha(folder/f'{arm}.log')
        meta=receipt['solver']
        for key,value in p['solver'].items():
            if key=='highspy':assert meta['highs_version']==value
            elif key=='numpy':assert meta['numpy']==value
            else:assert meta['solver_options']['random_seed' if key=='random_seed_base' else key]==(value+2*index+ai if key=='random_seed_base' else value)
        cost=a['cost'];proof=[a[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
        valid=check_certificate(81,degree,ii,jj,cost,*proof);assert valid==receipt['certificate']['certified']
        classification='unresolved';objective=gap=missing=extra=None
        if valid:
            assert meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']
            assert a['primal_values'].dtype==a['node_dual_values'].dtype==np.float64
            assert np.isin(a['primal_values'],[0.,.5,1.]).all() and np.array_equal(2*a['primal_values'],proof[0])
            assert np.max(abs(2*a['node_dual_values']-proof[1]))<=1e-6
            objective=sum(int(c)*int(x) for c,x in zip(cost,proof[0]));cert=receipt['certificate']
            assert cert['primal_objective_twice']==cert['dual_objective_twice']==objective
            assert abs(meta['solver_objective']-objective/2)<1e-5 and cert['primal_rounding_used'] is False
            integral=bool(np.isin(proof[0],[0,2]).all());assert integral==cert['integral']
            if integral:
                missing=int(np.count_nonzero(truth&(proof[0]!=2)));extra=int(np.count_nonzero(~truth&(proof[0]==2)))
                classification='success' if missing==extra==0 else 'resolved_failure_nonexact'
            else:classification='resolved_failure_fractional'
            gap=2*sum(map(int,cost[truth]))-objective;assert gap>=0
            for which in range(3):
                bad=[v.copy() for v in proof];bad[which][0]+=1
                assert not check_certificate(81,degree,ii,jj,cost,*bad)
        tp=int(np.count_nonzero(mask&truth_mask));fp=int(np.count_nonzero(mask&~truth_mask));fn=int(np.count_nonzero(~mask&truth_mask))
        assert tp+fp==tp+fn==5400
        rows.append(dict(index=index,arm=arm,family=family,degree=degree,baseline_exact=old_job['baseline_exact'],classification=classification,certified=valid,
                    true_positive_locations=tp,false_positive_locations=fp,false_negative_locations=fn,
                    flagged_per_cell=mask.sum(0,dtype=np.int64).tolist(),exposure_min=int(a['exposure'].min()),exposure_max=int(a['exposure'].max()),
                    zero_exposure_pairs=int(np.count_nonzero(a['exposure']==0)),true_edges_nonzero_cost=int(np.count_nonzero(cost[truth])),
                    zero_support_missing_true=int(np.count_nonzero(truth&(cost!=0))),zero_support_extra=int(np.count_nonzero(~truth&(cost==0))),
                    objective_twice=objective,true_gap_twice=gap,missing_true=missing,extra=extra))
    files=['detector-scores.npz']+[f'{a}.{e}' for a in ARMS for e in ['npz','json']]
    return dict(rows=rows,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),fit_seal_sha256=sha(base/'fit-seal.json'),
                inherited_audit_sha256=sha(OBS/f'b{index:03d}/audit.json'),artifact_sha256={name:sha(folder/name) for name in files})


def public_job(index):return audit_bank(index,read(PROTOCOL),public=True)


def main():
    start=time.perf_counter();p=read(PROTOCOL);verify(p);seal=check_seal(PUBLIC);saved=read(PUBLIC/'summary.json')
    assert saved['protocol_sha256']==sha(PROTOCOL)
    for name,h in saved['artifact_sha256'].items():assert sha(PUBLIC/name)==h
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(256)))
    for index,audit in enumerate(audits):
        assert audit==read(PUBLIC/f'b{index:03d}/audit.json')
        order=read(PUBLIC/f'b{index:03d}/audit-order.json')
        assert order['seal_sha256']==sha(PUBLIC/'fit-seal.json') and order['audit_started_monotonic_ns']>seal['sealed_monotonic_ns']
    rows=[r for a in audits for r in a['rows']];assert rows==saved['rows']
    for key,value in aggregate(rows).items():assert saved[key]==value
    assert saved['inherited_certificates_checked']==512 and saved['inherited_mutations_rejected']==1536
    rt=read(PUBLIC/'runtime.json')
    assert rt['status']=='complete' and rt['returncodes']==[0]*16 and rt['error'] is None
    assert rt['wall_seconds']<p['budget']['wall_seconds'] and rt['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
    assert rt['samples'][-1]['completed_LPs']==512 and rt['samples'][-1]['audited_banks']==256
    assert all(s['audited_banks']==0 for s in rt['samples'] if s['phase'] in ['hashes','fit_worker','all_fits_sealed'])
    print(json.dumps(dict(status='passed',**aggregate(rows),wall_seconds=time.perf_counter()-start,
                         full_fit_seal_before_diagnostics_verified=True,original_generation_provenance='Inherited H4at complete audit.')))


if __name__=='__main__':main()
