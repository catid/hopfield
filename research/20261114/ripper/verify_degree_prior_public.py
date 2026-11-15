"""Portable exact H1au certificates and outcomes; does not call an LP solver."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_degree_prior import check_certificate,independent_costs

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((PUBLIC/'degree_prior_public_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    p=json.loads((HERE/'protocol-degree-prior.json').read_text());s=json.loads((PUBLIC/'degree_prior.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-degree-prior.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest
    assert s['all_banks_retained'] and p['seeds']==list(range(32))
    assert sha(PUBLIC/'degree_prior_certificates.npz')==s['certificate_export_sha256']
    with np.load(PUBLIC/'degree_prior_certificates.npz',allow_pickle=False) as z:a={name:z[name].copy() for name in z.files}
    with np.load(PUBLIC/'degree_prior_counts.npz',allow_pickle=False) as z:counts={name:z[name].copy() for name in z.files}
    with np.load(PUBLIC/'adaptive_null_supports.npz',allow_pickle=False) as z:parent=z['supports'].copy()
    export=json.loads((PUBLIC/'degree_prior_count_export.json').read_text())
    assert export['counts_sha256']==sha(PUBLIC/'degree_prior_counts.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']
    assert counts['joint_count'].shape==counts['equal_count'].shape==(32,3240)
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories=[];proofs=0;mutation_rejections=0
    for seed in range(32):
        z={name.removeprefix(f's{seed}_'):value for name,value in a.items() if name.startswith(f's{seed}_')}
        audit=s['bank_audits'][seed];assert audit['seed']==seed and audit['status']=='validated'
        assert audit['auditor_sha256']==sha(HERE/'audit_degree_prior.py') and audit['protocol_sha256']==s['protocol_sha256']
        exported=export['banks'][seed];assert exported['seed']==seed and exported['parent_fit_sha256']==p['bank_sha256'][str(seed)]['fit.npz']
        m,c=counts['joint_count'][seed],counts['equal_count'][seed]
        assert exported['joint_count_bytes_sha256']==hashlib.sha256(m.tobytes()).hexdigest()
        assert exported['equal_count_bytes_sha256']==hashlib.sha256(c.tobytes()).hexdigest()
        costs,margin=independent_costs(m,c);assert np.array_equal(costs,z['cost']) and margin==audit['minimum_cost_rounding_margin']
        assert np.array_equal(z['original_support'],parent[seed])
        indices=np.flatnonzero(parent[seed]);assert np.array_equal(indices,z['candidate_indices'])
        expected=sorted((int(costs[edge]),int(edge)) for edge in indices)[:810]
        top=np.zeros(3240,bool);top[[edge for _,edge in expected]]=True;assert np.array_equal(top,z['top810_support'])
        for arm,support in [('original_support',parent[seed]),('top810_cardinality',top)]:
            row=next(row for row in s['rows'] if row['seed']==seed and row['arm']==arm)
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
            assert row['retained']==int(support.sum()) and row['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
        t,y,upper=[z[name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
        certified=check_certificate(81,20,ii[indices],jj[indices],costs[indices],t,y,upper)
        assert certified==audit['certificate_verified']
        if certified:
            assert np.array_equal(z['primal_values']*2,t)
            assert np.all(np.abs(z['row_dual_values']*2-y)<=1e-6)
            proofs+=1
            for index in range(3):
                changed=[t.copy(),y.copy(),upper.copy()];changed[index][0]+=1
                assert not check_certificate(81,20,ii[indices],jj[indices],costs[indices],*changed);mutation_rejections+=1
            if np.isin(t,[0,2]).all():
                support=np.zeros(3240,bool);support[indices]=t==2
                category='success' if np.array_equal(support,truth) else 'resolved_failure_nonexact'
                graph=audit['integral_candidate_graph'];assert graph is not None
                assert graph['exact_graph']==np.array_equal(support,truth)
                assert graph['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            else:category='resolved_failure_fractional'
        else:category='unresolved'
        assert category==audit['classification']==a['classification'][seed];categories.append(category)
    successes=categories.count('success');unresolved=categories.count('unresolved');failures=32-successes-unresolved
    decision='supported' if successes>=29 else ('falsified' if successes+unresolved<29 else 'inconclusive')
    assert s['primary']==dict(decision=decision,certified_integral_exact_graphs=successes,resolved_failures=failures,
                              unresolved=unresolved,success_lower_bound=successes,success_upper_bound=successes+unresolved,required=29)
    for group in s['grouped'][:2]:
        rows=[row for row in s['rows'] if row['arm']==group['arm']];assert len(rows)==32
        assert group['exact_graphs']==sum(row['exact_graph'] for row in rows)
        assert group['missing_true_total']==sum(row['missing_true_edges'] for row in rows)
        assert group['extra_total']==sum(row['extra_edges'] for row in rows)
    g=s['grouped'][2];assert g['certified_integral_exact']==successes and g['unresolved']==unresolved
    assert g['certified_integral_nonexact']==categories.count('resolved_failure_nonexact')
    assert g['certified_fractional']==categories.count('resolved_failure_fractional')
    for key,value in p['query_accounting'].items():assert s[key]==value
    runtime=json.loads((PUBLIC/'degree_prior_runtime.json').read_text());assert runtime==s['runtime']
    assert runtime['protocol_sha256']==sha(HERE/'protocol-degree-prior.json') and runtime['wrapper_sha256']==sha(HERE/'run_degree_prior.py')
    assert runtime['aggregate_cpu_s']<120 and runtime['wall_s']<60 and runtime['max_cpu_workers']==8
    assert len(runtime['commands'])==34 and all(row['returncode']==0 for row in runtime['commands'])
    assert sorted(row['seed'] for row in runtime['commands'] if row['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',decision=decision,exact_certificates=proofs,rejected_mutated_certificates=mutation_rejections,
                         checks=['Decimal110 quantization','top810 tie rule','exact integer primal/dual feasibility and complementarity','no primal rounding','graph statistics','decision bounds','query and runtime accounting'],
                         raw_acquisition_replay_required_for_original_observation_provenance=True)))


if __name__=='__main__':main()
