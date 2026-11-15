"""Public H1ax cost/proof/graph verifier; raw suffix replay is a separate audit."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((PUBLIC/'prefix_rate_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-prefix-rate.json').read_text());s=json.loads((PUBLIC/'prefix_rate.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-prefix-rate.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,path
    assert p['seeds']==list(range(32)) and p['prefixes']==[100,300,600] and s['all_banks_retained']
    assert sha(PUBLIC/'prefix_rate_certificates.npz')==s['certificate_export_sha256']
    with np.load(PUBLIC/'prefix_rate_certificates.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    with np.load(PUBLIC/'laplace_rate_certificates.npz',allow_pickle=False) as z:previous={k:z[k].copy() for k in z.files}
    old=json.loads((PUBLIC/'laplace_rate.json').read_text());assert old['certificate_export_sha256']==sha(PUBLIC/'laplace_rate_certificates.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['prefixes'].tolist()==p['prefixes'] and a['arms'].tolist()==p['arms']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories={n:[] for n in p['prefixes']};new_proofs=old_proofs=mutations=0
    for seed in range(32):
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')}
        audit=s['bank_audits'][seed];assert audit['seed']==seed and audit['status']=='validated'
        assert audit['auditor_sha256']==sha(HERE/'audit_prefix_rate.py') and audit['protocol_sha256']==s['protocol_sha256']
        observed=f'artifacts/research/20261114/ripper/permuted_noise/s{seed}/observations.npz'
        baseline=f'artifacts/research/20261114/ripper/laplace_rate/s{seed}/model.npz'
        assert audit['observations_sha256']==p['bank_sha256'][str(seed)][observed]
        assert audit['baseline_model_sha256']==p['bank_sha256'][str(seed)][baseline]
        assert len(audit['suffix_substitution_gates'])==3 and audit['archived_optimality_certificate_verified']
        for gate,n in zip(audit['suffix_substitution_gates'],p['prefixes'],strict=True):
            assert gate['rows']==n and gate['status']=='validated' and gate['changed_suffix_rows']==1110-n
            assert gate['original_suffix_sha256']!=gate['substituted_suffix_sha256']
        previous_joint=previous_equal=np.zeros(3240,np.int64)
        for n in p['prefixes']+[1110]:
            prefix='baseline_' if n==1110 else f'n{n}_'
            m=z[prefix+'joint_count'];c=z[prefix+'equal_count']
            assert m.shape==c.shape==(3240,) and m.dtype.kind==c.dtype.kind=='i'
            assert ((previous_joint<=m)&(m<=n)).all() and ((previous_equal<=c)&(c<=m)).all()
            assert int(m.sum())==n*300;previous_joint=m;previous_equal=c
            cost=independent_rate_costs(m,c);assert np.array_equal(cost,z[prefix+'cost'])
            top=np.zeros(3240,bool);top[[edge for _,edge in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
            assert np.array_equal(top,z[prefix+'top810_support'])
            part=audit['prefix_audits'][str(n)]
            if n==1110:
                for key,value in z.items():
                    if key.startswith(prefix):assert np.array_equal(value,previous[f's{seed}_'+key.removeprefix(prefix)])
            t,y,upper=[z[prefix+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
            certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==part['certificate_verified']
            if n==1110:assert certified;old_proofs+=1
            else:new_proofs+=certified
            category='unresolved';support=None
            if certified:
                assert np.array_equal(z[prefix+'primal_values']*2,t)
                assert np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
                for k in range(3):
                    changed=[t.copy(),y.copy(),upper.copy()];changed[k][0]+=1
                    assert not check_certificate(81,20,ii,jj,cost,*changed);mutations+=1
                if np.isin(t,[0,2]).all():
                    support=t==2;category='success' if np.array_equal(support,truth) else 'resolved_failure_nonexact'
                    candidate=part['integral_candidate_graph'];assert candidate is not None
                    assert candidate['exact_graph']==(category=='success')
                    assert candidate['missing_true_edges']==int((truth&~support).sum()) and candidate['extra_edges']==int((~truth&support).sum())
                    assert candidate['retained']==int(support.sum()) and candidate['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
                else:category='resolved_failure_fractional';assert part['integral_candidate_graph'] is None
            assert category==part['classification']
            if n==1110:assert category=='success'
            else:categories[n].append(category)
            for arm,graph in [('top810',top),('degree20',support)]:
                if graph is None:continue
                row=next(row for row in s['rows'] if row['seed']==seed and row['rows']==n and row['arm']==arm)
                missing=int((truth&~graph).sum());extra=int((~truth&graph).sum())
                assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
                assert row['retained']==int(graph.sum()) and row['support_bytes_sha256']==hashlib.sha256(graph.tobytes()).hexdigest()
    chosen=categories[600];success=chosen.count('success');unresolved=chosen.count('unresolved')
    decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive')
    assert s['primary']==dict(rows=600,decision=decision,certified_integral_exact_graphs=success,resolved_failures=32-success-unresolved,
                              unresolved=unresolved,success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    for group in s['grouped']:
        n=group['rows'];assert group['banks']==32
        if n==1110 or group['arm']=='top810':
            rows=[row for row in s['rows'] if row['rows']==n and row['arm']==group['arm']];assert len(rows)==32
            assert group['exact_graphs']==sum(row['exact_graph'] for row in rows)
            assert group['missing_true_total']==sum(row['missing_true_edges'] for row in rows) and group['extra_total']==sum(row['extra_edges'] for row in rows)
        else:
            for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),
                                   ('certified_fractional','resolved_failure_fractional'),('unresolved','unresolved')]:assert group[field]==categories[n].count(category)
    assert s['query_accounting']==p['query_accounting'] and s['query_scope']==p['query_scope'] and s['suffix_substitution_gates']==96
    for row,n in zip(s['query_accounting'],[100,300,600,1110],strict=True):
        assert row['rows']==n and row['training_per_bank']==25*n and row['training_total']==32*25*n
        assert row['shared_calibration_per_bank']==2000 and row['shared_calibration_total']==64000
        assert row['total_historical_acquisition_per_bank']==25*n+2000 and row['total_historical_acquisition_all_banks']==32*(25*n+2000)
    assert all(s[key]==0 for key in ['new_queries','new_calibrations','new_source_rows','new_solver_inputs','gpu_seconds'])
    rt=json.loads((PUBLIC/'prefix_rate_runtime.json').read_text());assert rt==s['runtime']
    assert rt['protocol_sha256']==sha(HERE/'protocol-prefix-rate.json') and rt['wrapper_sha256']==sha(HERE/'run_prefix_rate.py')
    assert rt['status']=='validated' and rt['wall_s']<60 and rt['aggregate_cpu_s']<120 and rt['max_cpu_workers']==8
    assert len(rt['commands'])==34 and all(row['returncode']==0 for row in rt['commands'])
    assert sorted(row['seed'] for row in rt['commands'] if row['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',decision=decision,new_exact_certificates=new_proofs,archived_exact_certificates=old_proofs,
                         rejected_mutated_certificates=mutations,checks=['Fraction costs and half-even rounding','top810 tie rule','exact primal-dual proofs including fractional optima','graph counts','decision bounds','nested query/runtime accounting'],
                         raw_observation_replay_required_for_prefix_counts_and_suffix_gates=True)))


if __name__=='__main__':main()
