"""H1be public fixed-prefix, mask-union, rate and exact-proof checker."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    manifest=json.loads((PUBLIC/'frozen_feedback_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-frozen-feedback.json').read_text());s=json.loads((PUBLIC/'frozen_feedback.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-frozen-feedback.json') and s['all_banks_retained'] and s['scope']==p['scope']
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,path
    assert p['seeds']==list(range(32)) and p['rows']==900 and p['arms']==['frozen','aligned'] and p['freeze_at']==300 and p['suffix_checkpoint']==450
    previous=json.loads((PUBLIC/'fresh_soft_feedback.json').read_text())
    assert p['input_sha256']['artifacts/research/20261114/ripper/fresh_soft_feedback/audit.json']==sha(PUBLIC/'fresh_soft_feedback.json')
    assert previous['certificate_export_sha256']==sha(PUBLIC/'fresh_soft_feedback_certificates.npz');old=load(PUBLIC/'fresh_soft_feedback_certificates.npz')
    assert sha(PUBLIC/'frozen_feedback_certificates.npz')==s['certificate_export_sha256'];a=load(PUBLIC/'frozen_feedback_certificates.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories={arm:[] for arm in p['arms']};new_proofs=old_proofs=mutations=0;accounting=[]
    for seed in range(32):
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')};audit=s['bank_audits'][seed]
        assert audit['seed']==seed and audit['status']=='validated' and audit['protocol_sha256']==s['protocol_sha256'] and audit['auditor_sha256']==sha(HERE/'audit_frozen_feedback.py')
        assert audit['independent_predictable_masks_replayed']==audit['common_noise_rows_replayed']==900 and audit['immutable_feedback_events']==600
        assert audit['archived_aligned_first300_masks_and_labels_identical'] and audit['feedback_snapshot_equals_first300_actual_equalities']
        gate=audit['suffix_substitution_gate'];assert gate['checkpoint']==450 and gate['status']=='validated'
        assert gate['prefix_masks_and_labels_unchanged'] and gate['prefix_counts_unchanged'] and gate['frozen_feedback_unchanged'] and gate['original_suffix_sha256']!=gate['substituted_suffix_sha256']
        for key,value in z.items():
            if key.startswith('aligned_') or key=='permuted_mask':assert np.array_equal(value,old[f's{seed}_'+key])
        frozen=z['frozen_mask'];aligned=z['aligned_mask'];permuted=z['permuted_mask']
        for mask in [frozen,aligned,permuted]:assert mask.shape==(900,81) and mask.dtype==bool and np.all(mask.sum(1)==25)
        assert np.array_equal(frozen[:300],aligned[:300]);prior=aligned|permuted
        counts=dict(frozen_queries=int(frozen.sum()),reused_queries=int((frozen&prior).sum()),new_queries=int((frozen&~prior).sum()),prior_h1bd_union=int(prior.sum()),combined_h1bd_union=int((frozen|prior).sum()),new_calibration_queries=0)
        assert counts==audit['query_accounting'];accounting.append(counts)
        assert counts['frozen_queries']==counts['reused_queries']+counts['new_queries']==22500 and counts['combined_h1bd_union']==counts['prior_h1bd_union']+counts['new_queries']
        snapshot=z['feedback_equal_count'];assert snapshot.shape==(3240,) and ((snapshot>=0)&(snapshot<=z['frozen_equal_count'])).all()
        assert int(snapshot.sum())==audit['frozen_feedback_equal_count_sum'] and int(z['frozen_equal_count'].sum())==audit['final_actual_equal_count_sum']
        for arm in p['arms']:
            prefix=arm+'_';part=audit['arm_audits'][arm];m,c=z[prefix+'joint_count'],z[prefix+'equal_count']
            assert m.shape==c.shape==(3240,) and ((0<=c)&(c<=m)&(m<=900)).all() and int(m.sum())==270000
            mask=z[arm+'_mask'];assert np.array_equal((mask[:,ii]&mask[:,jj]).sum(0),m)
            cost=independent_rate_costs(m,c);assert np.array_equal(cost,z[prefix+'cost']);graphs={}
            top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
            assert np.array_equal(top,z[prefix+'top810_support']);graphs[arm+'_top810']=top
            t,y,upper=[z[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
            certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==part['certificate_verified'];category='unresolved'
            if arm=='frozen':new_proofs+=certified
            else:assert certified;old_proofs+=1
            if certified:
                assert np.array_equal(z[prefix+'primal_values']*2,t) and np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
                for k in range(3):
                    changed=[t.copy(),y.copy(),upper.copy()];changed[k][0]+=1
                    assert not check_certificate(81,20,ii,jj,cost,*changed);mutations+=1
                if np.isin(t,[0,2]).all():graphs[arm]=t==2;category='success' if np.array_equal(t==2,truth) else 'resolved_failure_nonexact'
                else:category='resolved_failure_fractional'
            assert category==part['classification'];categories[arm].append(category)
            for name,graph in graphs.items():
                row=next(r for r in s['rows'] if r['seed']==seed and r['arm']==name);missing=int((truth&~graph).sum());extra=int((~truth&graph).sum())
                assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
                assert row['retained']==int(graph.sum()) and row['support_bytes_sha256']==hashlib.sha256(graph.tobytes()).hexdigest()
    successes=categories['frozen'].count('success');unresolved=categories['frozen'].count('unresolved')
    primary=dict(decision='supported' if successes>=29 else ('falsified' if successes+unresolved<29 else 'inconclusive'),success_lower=successes,success_upper=successes+unresolved,required=29,unresolved=unresolved)
    assert s['primary']==primary
    for group in s['grouped']:
        if group['arm'] in p['arms']:
            for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),('certified_fractional','resolved_failure_fractional'),('unresolved','unresolved')]:assert group[field]==categories[group['arm']].count(category)
        else:
            rows=[r for r in s['rows'] if r['arm']==group['arm']];assert len(rows)==32
            assert group['exact_graphs']==sum(r['exact_graph'] for r in rows) and group['missing_true_total']==sum(r['missing_true_edges'] for r in rows) and group['extra_total']==sum(r['extra_edges'] for r in rows)
    total={k:sum(r[k] for r in accounting) for k in accounting[0]};total['h1az_through_present_sequence_union']=3238105+total['new_queries'];assert total==s['physical_query_accounting']
    assert total['frozen_queries']==720000 and total['prior_h1bd_union']==previous['physical_query_accounting']['new_union_queries']==1216156
    assert s['alternative_query_accounting']==p['query_accounting']==dict(alternative_training=720000,alternative_calibration=0,prior_h1bd_union=1216156,prior_h1az_through_h1bd_sequence_union=3238105)
    assert s['physical_query_rule']==p['physical_query_rule'] and s['suffix_substitution_gates']==32 and all(s[k]==0 for k in ['new_source_ids','new_latent_draws','new_solver_inputs','gpu_seconds'])
    rt=json.loads((PUBLIC/'frozen_feedback_runtime.json').read_text());assert rt==s['runtime'] and rt['status']=='validated' and rt['protocol_sha256']==s['protocol_sha256']
    assert rt['wrapper_sha256']==sha(HERE/'run_frozen_feedback.py') and rt['wall_s']<120 and rt['aggregate_cpu_s']<240 and len(rt['commands'])==34
    assert all(r['returncode']==0 for r in rt['commands']) and sorted(r['seed'] for r in rt['commands'] if r['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',primary=primary,new_exact_certificates=new_proofs,archived_exact_certificates=old_proofs,rejected_mutated_certificates=mutations,physical_query_accounting=total,
        scope='Public first300 mask parity, full mask exposures and unions, snapshot count totals, actual-count rate costs, new and archived proofs, graphs and finite endpoint. Sequential policy/label replay, snapshot fidelity and suffix substitution require raw observations.')))


if __name__=='__main__':main()
