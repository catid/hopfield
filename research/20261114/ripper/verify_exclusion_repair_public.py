"""Portable H1av optimality/infeasibility proofs and graph outcome checker."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_degree_prior import check_certificate,independent_costs

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((PUBLIC/'exclusion_repair_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-exclusion-repair.json').read_text());s=json.loads((PUBLIC/'exclusion_repair.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-exclusion-repair.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest
    assert p['seeds']==list(range(32)) and s['all_banks_retained']
    assert s['certificate_export_sha256']==sha(PUBLIC/'exclusion_repair_certificates.npz')
    with np.load(PUBLIC/'exclusion_repair_certificates.npz',allow_pickle=False) as archive:a={k:archive[k].copy() for k in archive.files}
    with np.load(PUBLIC/'permuted_soundness_supports.npz',allow_pickle=False) as archive:original=archive['supports'].copy()
    old=json.loads((PUBLIC/'permuted_noise.json').read_text())
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories={arm:[] for arm in ['hard','unrestricted']};proofs=infeasible=mutations=0;missing_banks=missing_total=0
    for seed in range(32):
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')}
        audit=s['bank_audits'][seed];assert audit['seed']==seed and audit['status']=='validated'
        assert audit['protocol_sha256']==s['protocol_sha256'] and audit['auditor_sha256']==sha(HERE/'audit_exclusion_repair.py')
        assert audit['parent_fit_sha256']==p['bank_sha256'][str(seed)]['fit.npz']
        assert np.array_equal(z['original_support'],original[seed])
        previous=next(row for row in old['acquisition_audits'] if row['seed']==seed)
        assert previous['support_bytes_sha256']==hashlib.sha256(original[seed].tobytes()).hexdigest()
        cost,margin=independent_costs(z['joint_count'],z['equal_count'])
        assert np.array_equal(cost,z['cost']) and margin==audit['minimum_cost_rounding_margin']
        selected=sorted((int(cost[e]),e) for e in range(3240))[:810]
        top=np.zeros(3240,bool);top[[edge for _,edge in selected]]=True;assert np.array_equal(top,z['top810_support'])
        missing=int((truth&~original[seed]).sum());missing_total+=missing;missing_banks+=missing>0
        assert audit['original_missing_true_edges']==previous['missing_true_edges']==missing
        for arm,support in [('original_support',original[seed]),('top810_unrestricted',top)]:
            row=next(row for row in s['rows'] if row['seed']==seed and row['arm']==arm)
            lost=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert row['missing_true_edges']==lost and row['extra_edges']==extra and row['exact_graph']==(lost==extra==0)
            assert row['retained']==int(support.sum()) and row['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
        for arm,candidate in [('hard',original[seed]),('unrestricted',np.ones(3240,bool))]:
            report=audit['arms'][arm];indices=np.flatnonzero(candidate)
            assert np.array_equal(z[arm+'_candidate_indices'],indices)
            omitted=int((truth&~candidate).sum());assert report['omitted_true_edges']==omitted
            assert report['exact_truth_recovery_impossible_due_to_candidate_restriction']==(omitted>0)
            t,y,upper=[z[arm+'_'+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
            cert=check_certificate(81,20,ii[indices],jj[indices],cost[indices],t,y,upper)
            assert cert==report['certificate_verified'];witness=report['feasibility_certificate']
            if witness is not None:
                assert not cert and report['solver'] is None and arm=='hard'
                degrees=np.bincount(np.concatenate([ii[indices],jj[indices]]),minlength=81)
                bad=np.flatnonzero(degrees<20);assert len(bad)>0 and witness['vertex']==int(bad[0])
                assert witness['available_neighbors']==int(degrees[bad[0]])<witness['required_degree']==20
                assert len(z[arm+'_primal_values'])==len(t)==len(y)==len(upper)==0
                category='resolved_failure_infeasible';infeasible+=1
            elif cert:
                proofs+=1;raw=z[arm+'_primal_values'];assert np.array_equal(raw*2,t)
                assert np.all(np.abs(z[arm+'_row_dual_values']*2-y)<=1e-6)
                for index in range(3):
                    changed=[t.copy(),y.copy(),upper.copy()];changed[index][0]+=1
                    assert not check_certificate(81,20,ii[indices],jj[indices],cost[indices],*changed);mutations+=1
                if np.isin(t,[0,2]).all():
                    recovered=np.zeros(3240,bool);recovered[indices]=t==2
                    exact=np.array_equal(recovered,truth);category='success' if exact else 'resolved_failure_nonexact'
                    graph=report['integral_candidate_graph'];assert graph['exact_graph']==exact
                    assert graph['missing_true_edges']==int((truth&~recovered).sum()) and graph['extra_edges']==int((~truth&recovered).sum())
                    assert graph['support_bytes_sha256']==hashlib.sha256(recovered.tobytes()).hexdigest()
                else:category='resolved_failure_fractional'
            else:category='unresolved'
            assert category==report['classification']
            if omitted:assert category!='success'
            categories[arm].append(category)
    assert s['original_missing_true_banks']==missing_banks and s['original_missing_true_total']==missing_total
    full=categories['unrestricted'];success=full.count('success');unresolved=full.count('unresolved')
    decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive')
    assert s['primary']==dict(decision=decision,certified_integral_exact_graphs=success,resolved_failures=32-success-unresolved,
                              unresolved=unresolved,success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    for g in s['grouped'][:2]:
        rows=[row for row in s['rows'] if row['arm']==g['arm']];assert len(rows)==32
        assert g['exact_graphs']==sum(row['exact_graph'] for row in rows)
        assert g['missing_true_total']==sum(row['missing_true_edges'] for row in rows) and g['extra_total']==sum(row['extra_edges'] for row in rows)
    for arm,g in zip(['hard','unrestricted'],s['grouped'][2:],strict=True):
        assert g['arm']=='degree20_'+arm and g['banks']==32
        for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),
                               ('certified_fractional','resolved_failure_fractional'),('certified_infeasible','resolved_failure_infeasible'),('unresolved','unresolved')]:
            assert g[field]==categories[arm].count(category)
    for key,value in p['query_accounting'].items():assert s[key]==value
    rt=json.loads((PUBLIC/'exclusion_repair_runtime.json').read_text());assert rt==s['runtime']
    assert rt['protocol_sha256']==sha(HERE/'protocol-exclusion-repair.json') and rt['wrapper_sha256']==sha(HERE/'run_exclusion_repair.py')
    assert rt['wall_s']<60 and rt['aggregate_cpu_s']<120 and rt['max_cpu_workers']==8
    assert len(rt['commands'])==34 and all(row['returncode']==0 for row in rt['commands'])
    assert sorted(row['seed'] for row in rt['commands'] if row['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',decision=decision,exact_optimality_certificates=proofs,integer_infeasibility_certificates=infeasible,
                         rejected_mutated_optimality_certificates=mutations,original_missing_true_banks=missing_banks,
                         raw_observation_replay_required_for_source_count_provenance=True)))


if __name__=='__main__':main()
