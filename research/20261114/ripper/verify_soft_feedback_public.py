"""Public H1bc two-endpoint proof and cumulative acquisition-accounting checker."""
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
    manifest=json.loads((PUBLIC/'soft_feedback_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-soft-feedback.json').read_text());s=json.loads((PUBLIC/'soft_feedback.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-soft-feedback.json') and s['all_banks_retained']
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,path
    assert p['seeds']==list(range(32)) and p['rows']==900 and p['arms']==['aligned','permuted'] and p['suffix_checkpoint']==450
    assert sha(PUBLIC/'soft_feedback_certificates.npz')==s['certificate_export_sha256'];a=load(PUBLIC/'soft_feedback_certificates.npz')
    previous={}
    for name,file in [('adaptive','fresh_rate'),('balanced','balanced_mask')]:
        summary=json.loads((PUBLIC/f'{file}.json').read_text());assert summary['certificate_export_sha256']==sha(PUBLIC/f'{file}_certificates.npz')
        previous[name]=load(PUBLIC/f'{file}_certificates.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories={arm:[] for arm in p['arms']};new_proofs=old_proofs=mutations=0;accounting=[]
    for seed in range(32):
        perm=np.random.default_rng(p['feedback_permutation_seed_base']+seed).permutation(81);assert np.array_equal(perm,p['feedback_permutations'][seed])
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')};audit=s['bank_audits'][seed]
        assert audit['seed']==seed and audit['status']=='validated' and audit['protocol_sha256']==s['protocol_sha256'] and audit['auditor_sha256']==sha(HERE/'audit_soft_feedback.py')
        assert audit['independent_predictable_masks_replayed']==1800 and audit['common_noise_rows_replayed']==900 and len(audit['suffix_substitution_gates'])==2
        for arm,gate in zip(p['arms'],audit['suffix_substitution_gates'],strict=True):
            assert gate['arm']==arm and gate['checkpoint']==450 and gate['status']=='validated'
            assert gate['prefix_masks_and_labels_unchanged'] and gate['prefix_counts_unchanged'] and gate['original_suffix_sha256']!=gate['substituted_suffix_sha256']
        oldunion=previous['balanced'][f's{seed}_adaptive_mask']|previous['balanced'][f's{seed}_uniform_mask']|previous['balanced'][f's{seed}_balanced_mask']
        assert np.array_equal(oldunion,z['previous_mask_union'])
        aligned=z['aligned_mask'];permuted=z['permuted_mask'];assert aligned.shape==permuted.shape==(900,81) and aligned.dtype==permuted.dtype==bool
        assert np.all(aligned.sum(1)==25) and np.all(permuted.sum(1)==25);shared=aligned&permuted
        counts=dict(prior_union_queries=int(oldunion.sum())+2000,new_union_queries=int(((aligned|permuted)&~oldunion).sum()),aligned_reused=int((aligned&oldunion).sum()),
                    aligned_new_relative_prior=int((aligned&~oldunion).sum()),permuted_reused=int((permuted&oldunion).sum()),permuted_new_relative_prior=int((permuted&~oldunion).sum()),
                    new_shared_between_arms=int((shared&~oldunion).sum()),new_calibration_queries=0)
        counts['combined_union_queries']=counts['prior_union_queries']+counts['new_union_queries'];assert counts==audit['query_accounting'];accounting.append(counts)
        assert counts['new_union_queries']==counts['aligned_new_relative_prior']+counts['permuted_new_relative_prior']-counts['new_shared_between_arms']
        for arm in p['arms']:assert counts[arm+'_reused']+counts[arm+'_new_relative_prior']==22500
        for arm in p['arms']+['adaptive','balanced']:
            prefix=arm+'_';part=audit['arm_audits'][arm]
            if arm in previous:
                for key in ['joint_count','equal_count','cost','primal_values','row_dual_values','primal_twice','node_dual_twice','upper_dual_twice']:
                    assert np.array_equal(z[prefix+key],previous[arm][f's{seed}_'+key])
            m,c=z[prefix+'joint_count'],z[prefix+'equal_count'];assert m.shape==c.shape==(3240,) and ((0<=c)&(c<=m)&(m<=900)).all() and int(m.sum())==270000
            if arm in p['arms']:
                mask=z[arm+'_mask'];assert np.array_equal((mask[:,ii]&mask[:,jj]).sum(0),m)
            cost=independent_rate_costs(m,c);assert np.array_equal(cost,z[prefix+'cost']);graphs={}
            if arm in p['arms']:
                top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
                assert np.array_equal(top,z[prefix+'top810_support']);graphs[arm+'_top810']=top
            t,y,upper=[z[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
            certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==part['certificate_verified']
            if arm in p['arms']:new_proofs+=certified
            else:assert certified;old_proofs+=1
            category='unresolved'
            if certified:
                assert np.array_equal(z[prefix+'primal_values']*2,t) and np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
                for k in range(3):
                    changed=[t.copy(),y.copy(),upper.copy()];changed[k][0]+=1
                    assert not check_certificate(81,20,ii,jj,cost,*changed);mutations+=1
                if np.isin(t,[0,2]).all():graphs[arm]=t==2;category='success' if np.array_equal(t==2,truth) else 'resolved_failure_nonexact'
                else:category='resolved_failure_fractional'
            assert category==part['classification']
            if arm in p['arms']:categories[arm].append(category)
            for name,graph in graphs.items():
                row=next(r for r in s['rows'] if r['seed']==seed and r['arm']==name);missing=int((truth&~graph).sum());extra=int((~truth&graph).sum())
                assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
                assert row['retained']==int(graph.sum()) and row['support_bytes_sha256']==hashlib.sha256(graph.tobytes()).hexdigest()
    sa,sp=[categories[arm].count('success') for arm in p['arms']];ua,up=[categories[arm].count('unresolved') for arm in p['arms']]
    procedure=dict(decision='supported' if sa>=29 else ('falsified' if sa+ua<29 else 'inconclusive'),success_lower=sa,success_upper=sa+ua,required=29,unresolved=ua)
    low=sa-sp-up;high=sa+ua-sp;alignment=dict(decision='supported' if low>=4 else ('falsified' if high<4 else 'inconclusive'),difference_lower=low,difference_upper=high,required=4,aligned_success=sa,permuted_success=sp,aligned_unresolved=ua,permuted_unresolved=up)
    assert s['primary']==procedure and s['alignment_endpoint']==alignment
    for group in s['grouped']:
        if group['arm'] in p['arms']:
            for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),('certified_fractional','resolved_failure_fractional'),('unresolved','unresolved')]:assert group[field]==categories[group['arm']].count(category)
        else:
            rows=[r for r in s['rows'] if r['arm']==group['arm']];assert len(rows)==32
            assert group['exact_graphs']==sum(r['exact_graph'] for r in rows) and group['missing_true_total']==sum(r['missing_true_edges'] for r in rows) and group['extra_total']==sum(r['extra_edges'] for r in rows)
    total={k:sum(r[k] for r in accounting) for k in accounting[0]};assert total==s['physical_query_accounting'] and total['prior_union_queries']==1619152
    assert total['combined_union_queries']==1619152+total['new_union_queries'] and s['alternative_query_accounting']==p['query_accounting']==dict(alternative_each_arm_training=720000,alternative_each_arm_calibration=0,prior_cumulative_union=1619152)
    assert s['physical_query_rule']==p['physical_query_rule'] and s['suffix_substitution_gates']==64 and all(s[k]==0 for k in ['new_source_ids','new_latent_draws','new_solver_inputs','gpu_seconds'])
    rt=json.loads((PUBLIC/'soft_feedback_runtime.json').read_text());assert rt==s['runtime'] and rt['status']=='validated' and rt['protocol_sha256']==s['protocol_sha256']
    assert rt['wrapper_sha256']==sha(HERE/'run_soft_feedback.py') and rt['wall_s']<120 and rt['aggregate_cpu_s']<240 and len(rt['commands'])==34
    assert all(r['returncode']==0 for r in rt['commands']) and sorted(r['seed'] for r in rt['commands'] if r['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',procedure_endpoint=procedure,alignment_endpoint=alignment,new_exact_certificates=new_proofs,archived_exact_certificates=old_proofs,
        rejected_mutated_certificates=mutations,physical_query_accounting=total,scope='Public feedback-permutation seeds, mask exposures/union accounting, costs/proofs/graphs and separate endpoint bounds verified. Sequential policy/label replay and suffix substitution require raw observations.')))


if __name__=='__main__':main()
