"""H4bb independent pooled-union replay, inherited controls and exact certificates."""
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
from pooled_donor import ROOT,HERE,OBS,PARENT,PARENT_PROTOCOL,OUT,PUBLIC,PROTOCOL,FAMILIES,sha,read,save,verify,check_seal,check_parent_seal
from larger_donor_math import independent_counts,validate_labels,row_digests
from audit_larger_donor import audit_bank as audit_parent
from audit_degree_prior import check_certificate
from audit_objective_gap import reconstruct


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(256)) and sum(r['parent_exact'] for r in rows)==229
    assert all(r['parent_exact']==r['donor_exact'] for r in rows)
    counts={k:sum(r['classification']==k for r in rows) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']}
    lo=counts['success'];hi=lo+counts['unresolved']
    losses=sum(r['parent_exact'] and r['classification'].startswith('resolved_failure') for r in rows)
    uncertain=sum(r['parent_exact'] and r['classification']=='unresolved' for r in rows)
    primary=dict(exact_recovery=dict(lower=lo,upper=hi,required=245,total=256,decision='supported' if lo>=245 else 'falsified' if hi<245 else 'inconclusive'),
        regressions=dict(lower=losses,upper=losses+uncertain,maximum=4,total=229,decision='supported' if losses+uncertain<=4 else 'falsified' if losses>4 else 'inconclusive'))
    group=dict(arm='pooled1800',**counts,gains_vs_parent=sum(not r['parent_exact'] and r['classification']=='success' for r in rows),losses_vs_parent=losses,
        families=[dict(family=f,exact=sum(r['family']==f and r['classification']=='success' for r in rows),unresolved=sum(r['family']==f and r['classification']=='unresolved' for r in rows)) for f in FAMILIES])
    certs=sum(r['certified'] for r in rows)
    return dict(primary=primary,grouped=[group],new_certificates_checked=certs,new_mutations_rejected=3*certs,all256_banks_retained=True)


def audit_fit(a,rec,cost,degree,index,p,truth):
    assert a['cost'].dtype==np.int64 and np.array_equal(a['cost'],cost)
    meta=rec['solver'];ii,jj=np.triu_indices(81,1)
    for key,value in p['solver'].items():
        if key=='highspy':assert meta['highs_version']==value
        elif key=='numpy':assert meta['numpy']==value
        else:assert meta['solver_options']['random_seed' if key=='random_seed_base' else key]==(value+index if key=='random_seed_base' else value)
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
    base=PUBLIC if public else OUT;folder=base/f'b{index:03d}';seal=read(base/'fit-seal.json');rec=read(folder/'pooled.json')
    assert rec['protocol_sha256']==seal['protocol_sha256']==sha(PROTOCOL) and rec['index']==index and rec['arm']=='pooled'
    assert rec['fit_started_monotonic_ns']<=rec['fit_completed_monotonic_ns']<=seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns']
    assert rec['models_sha256']==sha(folder/'pooled.npz')
    if not public:assert rec['log_sha256']==sha(folder/'pooled.log')
    parentp=read(PARENT_PROTOCOL);assert p['jobs'][index]==parentp['jobs'][index]
    archive=ROOT/p['jobs'][index]['archive']['source_path']
    if not public:assert archive.exists() # Mandatory raw provenance in charged scientific audit.
    # Parent auditor reconstructs all2400 original noise rows if raw sources exist.
    inherited=audit_parent(index,parentp,public=True);assert inherited==read(PARENT/f'b{index:03d}/audit.json')
    old=inherited['rows'][0];assert old['classification']!='unresolved' and old['donors'][0]['classification']!='unresolved'
    with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:target=z['noisy']
    with np.load(PARENT/f'b{index:03d}/donor.npz',allow_pickle=False) as z:donor=z['noisy'];donorids=z['source_rows']
    validate_labels(target,(600,81));validate_labels(donor,(1200,81));assert np.array_equal(donorids,np.arange(600,1800))
    # Separate assembly from the producer; every row has an exact ancestor identity.
    rows=np.empty((1800,81),np.int64)
    for row in range(1800):rows[row]=target[row] if row<600 else donor[row-600]
    with np.load(folder/'pooled.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    assert a['source_rows'].dtype==np.int64 and np.array_equal(a['source_rows'],np.arange(1800))
    provenance=dict(index=index,target_models_sha256=sha(OBS/f'b{index:03d}/models.npz'),donor_models_sha256=sha(PARENT/f'b{index:03d}/donor.npz'),
        parent_source_provenance_sha256=sha(PARENT/f'b{index:03d}/source-provenance.json'),original_source_sha256=p['jobs'][index]['archive']['source_sha256'],
        pooled_source_rows=list(range(1800)),pooled_row_sha256=row_digests(rows),original2400_noise_replay_verified_in_charged_audit=True,
        scope='Raw noise replay is inherited if raw source archives are absent in a public checkout; exact1800-row union is always reconstructed.')
    if public:assert provenance==read(folder/'source-union.json')
    else:save(folder/'source-union.json',provenance)
    degree=p['jobs'][index]['inferred_degree'];assert rec['degree']==degree==old['degree']
    meta=read(HERE/'corruption-censor.json')['jobs'][index];family=meta['family'];group=meta['group']
    name='varying_degree' if family in ['cycle81','cliques9'] else 'cycle_power';gi=2*group+int(family in ['cliques9','power3']);ii,jj=np.triu_indices(81,1)
    with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz') as z:
        graph=reconstruct(z['orders'][gi],family);truth=graph[ii,jj];assert np.array_equal(truth,z['truths'][gi]) and (graph.sum(0)==degree).all()
    checked=audit_fit(a,rec,independent_counts(rows),degree,index,p,truth)
    row=dict(index=index,arm='pooled1800',family=family,degree=degree,parent_exact=old['classification']=='success',donor_exact=old['donors'][0]['classification']=='success',**checked)
    files=['pooled.npz','pooled.json','source-union.json']
    return dict(rows=[row],protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),fit_seal_sha256=sha(base/'fit-seal.json'),
        inherited_audit_sha256=sha(PARENT/f'b{index:03d}/audit.json'),artifact_sha256={name:sha(folder/name) for name in files})


def public_job(index):return audit_bank(index,read(PROTOCOL),True)


def main():
    start=time.perf_counter();p=read(PROTOCOL);verify(p,public=True);check_parent_seal(PARENT);seal=check_seal(PUBLIC);saved=read(PUBLIC/'summary.json')
    assert saved['protocol_sha256']==sha(PROTOCOL)
    for rel,h in saved['artifact_sha256'].items():assert sha(PUBLIC/rel)==h
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(256)))
    for i,audit in enumerate(audits):
        assert audit==read(PUBLIC/f'b{i:03d}/audit.json');order=read(PUBLIC/f'b{i:03d}/audit-order.json')
        assert order['seal_sha256']==sha(PUBLIC/'fit-seal.json') and order['audit_started_monotonic_ns']>seal['sealed_monotonic_ns']
    rows=[r for a in audits for r in a['rows']];assert rows==saved['rows']
    for key,value in aggregate(rows).items():assert saved[key]==value
    assert saved['new_LPs']==256 and saved['inherited_certificates_checked']==1536 and saved['inherited_mutations_rejected']==4608
    assert saved['new_source_draws']==saved['new_labels']==0
    rt=read(PUBLIC/'runtime.json');assert rt['status']=='complete' and rt['returncodes']==[0]*16 and rt['error'] is None
    assert rt['wall_seconds']<p['budget']['wall_seconds'] and rt['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
    assert rt['samples'][-1]['completed_LPs']==256 and rt['samples'][-1]['audited_banks']==256
    assert all(s['audited_banks']==0 for s in rt['samples'] if s['phase'] in ['hashes','fit_worker','all_fits_sealed'])
    print(json.dumps(dict(status='passed',**aggregate(rows),wall_seconds=time.perf_counter()-start,raw_sources_unavailable=sum(not (ROOT/j['archive']['source_path']).exists() for j in p['jobs']))))


if __name__=='__main__':main()
