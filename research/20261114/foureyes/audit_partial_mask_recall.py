"""Independent partial-mask replay and exact proof/endpoint checks."""
from concurrent.futures import ProcessPoolExecutor
import json, time
import numpy as np
from partial_mask_recall import ROOT, HERE, PARENT, PARENT_PROTOCOL, OUT, PUBLIC, PROTOCOL, RECALLS, FAMILIES, sha, read, verify
from partial_mask_math import independent_draws, independent_mask
from corruption_censor_math import independent_score
from audit_corruption_censor import audit_bank as audit_parent
from audit_objective_gap import reconstruct
from audit_degree_prior import check_certificate


def aggregate(rows):
    assert len(rows) == 1280 and [(r['index'], r['recall_percent']) for r in rows] == [(i,p) for i in range(256) for p in RECALLS]
    groups = []
    for percent in RECALLS:
        part = [r for r in rows if r['recall_percent'] == percent]
        groups.append(dict(recall_percent=percent,
                      **{k:sum(r['classification'] == k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
                      revealed_indices=sum(r['revealed_indices'] for r in part), supplied_true_indices=256*600*9,
                      gains_vs_zero=sum(not r['baseline_exact'] and r['classification']=='success' for r in part),
                      losses_vs_zero=sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in part),
                      families=[dict(family=f, exact=sum(r['family']==f and r['classification']=='success' for r in part),
                                     unresolved=sum(r['family']==f and r['classification']=='unresolved' for r in part)) for f in FAMILIES]))
    primary = groups[2]
    low, high = primary['success'], primary['success']+primary['unresolved']
    return dict(primary=dict(recall_percent=90, lower=low, upper=high, required=245, total=256,
                decision='supported' if low>=245 else 'falsified' if high<245 else 'inconclusive'), grouped=groups,
                new_certificates_checked=sum(r['certified'] for r in rows), new_mutations_rejected=3*sum(r['certified'] for r in rows),
                all1280_outputs_retained=True)


def audit_bank(index, p, public=False):
    folder = (PUBLIC if public else OUT)/f'b{index:03d}'
    parent_protocol = read(PARENT_PROTOCOL)
    inherited = audit_parent(index, parent_protocol, public=True)
    assert inherited == read(PARENT/f'b{index:03d}/audit.json')
    assert inherited['row']['classification'] == 'success'
    job = p['jobs'][index]
    assert job == parent_protocol['jobs'][index]
    degree, family, group = job['inferred_degree'], job['family'], job['group']
    with np.load(PARENT/f'b{index:03d}/models.npz', allow_pickle=False) as z:
        cells, noisy, full = z['cells'], z['noisy'], z['mask']
    raw = np.load(folder/'draws.npy', allow_pickle=False)
    reference_raw = independent_draws(p['thinning']['seed_base'], index)
    assert raw.dtype == np.uint64 and raw.shape == (600,9) and np.array_equal(raw, reference_raw)
    name = 'varying_degree' if family in ['cycle81','cliques9'] else 'cycle_power'
    gi = 2*group+int(family in ['cliques9','power3'])
    ii,jj = np.triu_indices(81,1)
    with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz') as z:
        graph = reconstruct(z['orders'][gi], family)
        truth = graph[ii,jj]
        assert np.array_equal(truth,z['truths'][gi]) and (graph.sum(0)==degree).all()
    rows, previous = [], np.zeros((600,81),bool)
    for arm, percent in enumerate(RECALLS):
        with np.load(folder/f'p{percent}.npz', allow_pickle=False) as z:
            a = {key:z[key] for key in z.files}
        mask = independent_mask(cells, raw, percent)
        assert a['mask'].dtype==bool and np.array_equal(a['mask'],mask)
        assert not np.any(mask & ~full) and not np.any(previous & ~mask)
        previous = mask
        reference = independent_score(noisy,mask)
        for key,value in reference.items():
            assert a[key].dtype==np.int64 and np.array_equal(a[key],value)
        cost = a['cost']
        receipt = read(folder/f'p{percent}.json')
        assert receipt['index']==index and receipt['recall_percent']==percent and receipt['degree']==degree
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['models_sha256']==sha(folder/f'p{percent}.npz')
        if not public:
            assert receipt['log_sha256']==sha(folder/f'p{percent}.log')
        meta = receipt['solver']
        for key,value in p['solver'].items():
            if key=='highspy':
                assert meta['highs_version']==value
            elif key=='numpy':
                assert meta['numpy']==value
            else:
                assert meta['solver_options']['random_seed' if key=='random_seed_base' else key]==(value+5*index+arm if key=='random_seed_base' else value)
        proof = [a[key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
        valid = check_certificate(81,degree,ii,jj,cost,*proof)
        assert valid==receipt['certificate']['certified']
        classification, gap, objective, missing, extra = 'unresolved', None, None, None, None
        if valid:
            assert meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']
            assert a['primal_values'].dtype==a['node_dual_values'].dtype==np.float64
            assert np.isin(a['primal_values'],[0.,.5,1.]).all() and np.array_equal(2*a['primal_values'],proof[0])
            assert np.max(abs(2*a['node_dual_values']-proof[1]))<=1e-6
            objective=sum(int(c)*int(x) for c,x in zip(cost,proof[0]))
            cert=receipt['certificate']
            assert cert['primal_objective_twice']==cert['dual_objective_twice']==objective
            assert abs(meta['solver_objective']-objective/2)<1e-5 and cert['primal_rounding_used'] is False
            integral=bool(np.isin(proof[0],[0,2]).all())
            assert integral==cert['integral']
            if integral:
                missing=int(np.count_nonzero(truth & (proof[0]!=2)))
                extra=int(np.count_nonzero(~truth & (proof[0]==2)))
                classification='success' if missing==extra==0 else 'resolved_failure_nonexact'
            else:
                classification='resolved_failure_fractional'
            gap=2*sum(map(int,cost[truth]))-objective
            assert gap>=0
            for which in range(3):
                changed=[v.copy() for v in proof]
                changed[which][0]+=1
                assert not check_certificate(81,degree,ii,jj,cost,*changed)
        rows.append(dict(index=index, recall_percent=percent, family=family, degree=degree,
                    baseline_exact=job['baseline_exact'], classification=classification, certified=valid,
                    revealed_indices=int(mask.sum()), supplied_true_indices=int(full.sum()),
                    exposure_min=int(a['exposure'].min()), exposure_max=int(a['exposure'].max()),
                    zero_exposure_pairs=int(np.count_nonzero(a['exposure']==0)), true_edges_nonzero_cost=int(np.count_nonzero(cost[truth])),
                    zero_support_missing_true=int(np.count_nonzero(truth & (cost!=0))), zero_support_extra=int(np.count_nonzero(~truth & (cost==0))),
                    objective_twice=objective, true_gap_twice=gap, missing_true=missing, extra=extra))
    files=['draws.npy']+[f'p{p}.{ext}' for p in RECALLS for ext in ['npz','json']]
    return dict(rows=rows, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__),
                inherited_audit_sha256=sha(PARENT/f'b{index:03d}/audit.json'),
                artifact_sha256={name:sha(folder/name) for name in files})


def public_job(index):
    return audit_bank(index,read(PROTOCOL),public=True)


def main():
    start=time.perf_counter()
    p=read(PROTOCOL)
    verify(p)
    saved=read(PUBLIC/'summary.json')
    assert saved['protocol_sha256']==sha(PROTOCOL)
    for name,digest in saved['artifact_sha256'].items():
        assert sha(PUBLIC/name)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:
        audits=list(pool.map(public_job,range(256)))
    for index,audit in enumerate(audits):
        assert audit==read(PUBLIC/f'b{index:03d}/audit.json')
    rows=[r for a in audits for r in a['rows']]
    assert rows==saved['rows']
    for key,value in aggregate(rows).items():
        assert saved[key]==value
    assert saved['inherited_certificates_checked']==512 and saved['inherited_mutations_rejected']==1536
    runtime=read(PUBLIC/'runtime.json')
    assert runtime['status']=='complete' and runtime['returncodes']==[0]*8 and runtime['error'] is None
    assert runtime['wall_seconds']<p['budget']['wall_seconds'] and runtime['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
    assert runtime['samples'][-1]['completed_LPs']==1280 and runtime['samples'][-1]['audited_banks']==256
    print(json.dumps(dict(status='passed',**aggregate(rows),wall_seconds=time.perf_counter()-start,
               scope='All scores and proofs replayed from public H4at observations. Original source generation remains inherited.')))


if __name__=='__main__':
    main()
