"""Public exact-rational rate costs and new/archived primal-dual proofs."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate,independent_costs

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((PUBLIC/'laplace_rate_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-laplace-rate.json').read_text());s=json.loads((PUBLIC/'laplace_rate.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-laplace-rate.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest
    assert p['seeds']==list(range(32)) and s['all_banks_retained'] and s['interpretation']==p['interpretation']
    assert sha(PUBLIC/'laplace_rate_certificates.npz')==s['certificate_export_sha256']
    with np.load(PUBLIC/'laplace_rate_certificates.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    with np.load(PUBLIC/'exclusion_repair_certificates.npz',allow_pickle=False) as z:previous={k:z[k].copy() for k in z.files}
    old=json.loads((PUBLIC/'exclusion_repair.json').read_text());assert old['certificate_export_sha256']==sha(PUBLIC/'exclusion_repair_certificates.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories=[];new_proofs=old_proofs=mutations=0
    for seed in range(32):
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')}
        audit=s['bank_audits'][seed];assert audit['seed']==seed and audit['status']=='validated'
        assert audit['auditor_sha256']==sha(HERE/'audit_laplace_rate.py') and audit['protocol_sha256']==s['protocol_sha256']
        for name in ['joint_count','equal_count','original_support']:assert np.array_equal(z[name],previous[f's{seed}_{name}'])
        source_path=f'artifacts/research/20261114/ripper/permuted_noise/s{seed}/fit.npz'
        baseline_path=f'artifacts/research/20261114/ripper/exclusion_repair/s{seed}/model.npz'
        assert audit['parent_fit_sha256']==p['bank_sha256'][str(seed)][source_path]
        assert audit['baseline_model_sha256']==p['bank_sha256'][str(seed)][baseline_path]
        costs=independent_rate_costs(z['joint_count'],z['equal_count']);assert np.array_equal(costs,z['cost'])
        selected=sorted((int(costs[e]),e) for e in range(3240))[:810]
        top=np.zeros(3240,bool);top[[edge for _,edge in selected]]=True;assert np.array_equal(top,z['top810_support'])
        original_cost,_=independent_costs(z['joint_count'],z['equal_count']);assert np.array_equal(original_cost,z['baseline_cost'])
        for name in ['cost','unrestricted_primal_values','unrestricted_row_dual_values','unrestricted_primal_twice','unrestricted_node_dual_twice','unrestricted_upper_dual_twice']:
            assert np.array_equal(z['baseline_'+name],previous[f's{seed}_{name}'])
        baseline_support=None;category='unresolved'
        for prefix,cost in [('baseline_unrestricted_',original_cost),('',costs)]:
            t,y,upper=[z[prefix+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
            certified=check_certificate(81,20,ii,jj,cost,t,y,upper)
            if prefix:
                assert certified and audit['archived_optimality_certificate_verified'];old_proofs+=1
            else:
                assert certified==audit['certificate_verified'];new_proofs+=certified
            if certified:
                assert np.array_equal(z[prefix+'primal_values']*2,t)
                assert np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
                for k in range(3):
                    changed=[t.copy(),y.copy(),upper.copy()];changed[k][0]+=1
                    assert not check_certificate(81,20,ii,jj,cost,*changed);mutations+=1
                if np.isin(t,[0,2]).all():
                    support=t==2;exact=np.array_equal(support,truth)
                    if prefix:assert exact;baseline_support=support
                    else:
                        category='success' if exact else 'resolved_failure_nonexact'
                        graph=audit['integral_candidate_graph'];assert graph['exact_graph']==exact
                        assert graph['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
                        assert graph['missing_true_edges']==int((truth&~support).sum()) and graph['extra_edges']==int((~truth&support).sum())
                elif not prefix:category='resolved_failure_fractional'
        assert category==audit['classification']==a['classification'][seed];categories.append(category)
        for arm,support in zip(p['arms'][:3],[z['original_support'],baseline_support,top],strict=True):
            row=next(row for row in s['rows'] if row['seed']==seed and row['arm']==arm)
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
            assert row['retained']==int(support.sum()) and row['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
    successes=categories.count('success');unresolved=categories.count('unresolved')
    decision='supported' if successes>=29 else ('falsified' if successes+unresolved<29 else 'inconclusive')
    assert s['primary']==dict(decision=decision,certified_integral_exact_graphs=successes,resolved_failures=32-successes-unresolved,
                              unresolved=unresolved,success_lower_bound=successes,success_upper_bound=successes+unresolved,required=29)
    for g in s['grouped'][:3]:
        rows=[row for row in s['rows'] if row['arm']==g['arm']];assert len(rows)==32
        assert g['exact_graphs']==sum(row['exact_graph'] for row in rows)
        assert g['missing_true_total']==sum(row['missing_true_edges'] for row in rows) and g['extra_total']==sum(row['extra_edges'] for row in rows)
    g=s['grouped'][3]
    for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),
                           ('certified_fractional','resolved_failure_fractional'),('unresolved','unresolved')]:assert g[field]==categories.count(category)
    for key,value in p['query_accounting'].items():assert s[key]==value
    rt=json.loads((PUBLIC/'laplace_rate_runtime.json').read_text());assert rt==s['runtime']
    assert rt['protocol_sha256']==sha(HERE/'protocol-laplace-rate.json') and rt['wrapper_sha256']==sha(HERE/'run_laplace_rate.py')
    assert rt['wall_s']<60 and rt['aggregate_cpu_s']<120 and rt['max_cpu_workers']==8
    assert len(rt['commands'])==34 and all(row['returncode']==0 for row in rt['commands'])
    assert sorted(row['seed'] for row in rt['commands'] if row['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',decision=decision,new_exact_certificates=new_proofs,archived_exact_certificates=old_proofs,
                         rejected_mutated_certificates=mutations,checks=['Fraction costs and half-even rounding','top810 tie rule','exact new and archived primal-dual proofs','graph counts','decision bounds','query/runtime accounting'],
                         raw_observation_replay_required_for_original_count_provenance=True)))


if __name__=='__main__':main()
