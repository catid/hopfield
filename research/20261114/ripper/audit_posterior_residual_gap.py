"""Independent H1cb literal truth, exact posterior costs, selected proofs and gaps."""
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
from posterior_residual_gap import ROOT,HERE,PARENT,OUT,PUBLIC,PROTOCOL,ARMS,sha,read,verify
from audit_repeated_objective_gap import independent_truth,certify
from audit_posterior_fresh_confirm import score_reference


def audit_bank(seed,public=False,protocol=None):
    p=verify() if protocol is None else protocol
    directory=(PUBLIC if public else OUT)/f's{seed}'
    rows=read(directory/'rows.json');prior=read(PARENT/'summary.json')['rows']
    assert len(rows)==3
    truth=independent_truth();mutations=0
    seal=read(PARENT/'query-record-seal.json')
    assert seal['record_sha256'][str(seed)]==sha(PARENT/f's{seed}/query-records.npz')
    assert seal['audit_sha256'][str(seed)]==sha(PARENT/f's{seed}/query-record-audit.json')
    with np.load(PARENT/f's{seed}/query-records.npz',allow_pickle=False) as z:queries={k:z[k] for k in z.files}
    for ai,arm in enumerate(ARMS):
        index=3*seed+ai;old=prior[index];row=rows[ai]
        assert old['index']==index and old['seed']==seed and old['arm']==arm and old['selected_degree']==20
        receipt=read(PARENT/f's{seed}/{arm}-receipt.json')
        assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm
        assert receipt['protocol_sha256']==p['parent_protocol_sha256']==sha(HERE/'protocol-posterior-fresh-confirm.json')
        assert receipt['models_sha256']==sha(PARENT/f's{seed}/{arm}-models.npz')
        assert receipt['query_records_sha256']==sha(PARENT/f's{seed}/query-records.npz') and receipt['query_seal_sha256']==sha(PARENT/'query-record-seal.json')
        cost,joint,extra=score_reference(queries,arm)
        with np.load(PARENT/f's{seed}/{arm}-models.npz',allow_pickle=False) as z:
            assert np.array_equal(cost,z['cost']) and np.array_equal(joint,z['joint_count'])
            proof=[z['d20_'+key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
            assert np.array_equal(2*z['d20_primal_values'],proof[0]) and np.max(abs(2*z['d20_node_dual_values']-proof[1]))<=1e-6
        optimum=certify(81,20,cost,*proof)
        assert old['objective_twice'][20]==optimum
        record=receipt['records'][19];certificate=record['certificate'];solver=record['solver']
        assert record['degree']==20 and certificate['certified'] and certificate['primal_objective_twice']==certificate['dual_objective_twice']==optimum
        assert certificate['primal_rounding_used'] is False and solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']
        assert abs(solver['solver_objective']-optimum/2)<1e-5
        for which in range(3):
            altered=[v.copy() for v in proof];altered[which][0]+=1
            try:certify(81,20,cost,*altered)
            except AssertionError:mutations+=1
            else:raise AssertionError('Altered proof accepted')
        actual=sum(2*int(cost[e]) for e in range(3240) if truth[e]);gap=actual-optimum;assert gap>=0
        exact=all(int(proof[0][e])==2*int(truth[e]) for e in range(3240));integral=all(int(x) in [0,2] for x in proof[0])
        assert certificate['integral']==integral
        kind='success' if exact else 'resolved_failure_nonexact' if integral else 'resolved_failure_fractional'
        assert old['classification']==kind
        if exact:assert gap==0
        expected=dict(index=index,seed=seed,arm=arm,degree=20,parent_classification=kind,optimum_twice=optimum,truth_cost_twice=actual,
                      gap_twice=gap,exact=exact,integral=integral,truth_objective_status='cooptimal' if gap==0 else 'strictly_suboptimal')
        assert row==expected
    assert mutations==9
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),
                rows_sha256=sha(directory/'rows.json'),selected_certificates=3,mutations_rejected=mutations,independent_query_costs_verified=3)


def independent_summary(rows):
    assert len(rows)==192 and [(r['index'],r['seed'],r['arm']) for r in rows]==[(3*b+i,b,a) for b in range(64) for i,a in enumerate(ARMS)]
    grouped=[]
    failures=[]
    for arm in ARMS:
        group=dict(arm=arm,total=0,exact=0,strict=0,cooptimal_nonexact=0,zero_gap=0,selected_fractional=0)
        for row in rows:
            if row['arm']!=arm:continue
            group['total']+=1;group['exact']+=row['exact'];group['strict']+=row['gap_twice']>0
            group['cooptimal_nonexact']+=row['gap_twice']==0 and not row['exact'];group['zero_gap']+=row['gap_twice']==0
            group['selected_fractional']+=not row['integral']
            if arm=='posterior_triple90' and not row['exact']:failures.append(row)
        assert group['total']==64;grouped.append(group)
    assert [r['seed'] for r in failures]==[6,7,57] and [g['exact'] for g in grouped]==[64,51,61]
    strict=len([r for r in failures if r['gap_twice']>0])
    return dict(denominator=3,required_strict=2,strict_failures=strict,cooptimal_failures=3-strict,
                decision='supported' if strict>=2 else 'falsified',grouped=grouped,
                all192_retained=True,selected_certificates_rechecked=192,proof_mutations_rejected=576,new_LPs=0,new_labels=0,
                parent_physical_noisy_union=2073600,official_noisy_label_sequence=16760919)


def public_job(seed):return audit_bank(seed,True)


def main():
    started=time.perf_counter();verify();saved=read(PUBLIC/'summary.json')
    assert saved['status']=='validated' and saved['protocol_sha256']==sha(PROTOCOL)
    for rel,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/rel)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(64)))
    for seed,audit in enumerate(audits):assert audit==read(PUBLIC/f's{seed}/audit.json')
    rows=[row for seed in range(64) for row in read(PUBLIC/f's{seed}/rows.json')]
    assert rows==saved['rows']
    expected=independent_summary(rows)
    for key,value in expected.items():assert saved[key]==value,key
    assert sum(a['selected_certificates'] for a in audits)==192 and sum(a['mutations_rejected'] for a in audits)==576
    print(json.dumps(dict(status='passed',**expected,wall_seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
