"""Portable H1az identity sampling/disjointness, costs, graph and proof verifier."""
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
    manifest=json.loads((PUBLIC/'fresh_rate_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-fresh-rate.json').read_text());s=json.loads((PUBLIC/'fresh_rate.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-fresh-rate.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,path
    assert p['seeds']==list(range(32)) and p['rows']==900 and s['all_banks_retained'] and s['scope']==p['scope']
    assert sha(PUBLIC/'fresh_rate_ledger.json')==p['ledger_sha256'] and sha(PUBLIC/'fresh_rate_excluded_ids.npz')==p['excluded_ids_sha256']
    ledger=json.loads((PUBLIC/'fresh_rate_ledger.json').read_text());excluded=load(PUBLIC/'fresh_rate_excluded_ids.npz')['excluded']
    assert excluded.dtype==np.int64 and excluded.shape==(1928477,) and np.array_equal(excluded,np.unique(excluded))
    assert ((excluded>=0)&(excluded<p['pool_size'])).all() and p['excluded_unique']==ledger['excluded_unique']==len(excluded)
    assert hashlib.sha256(excluded.tobytes()).hexdigest()==ledger['excluded_id_bytes_sha256']
    selected=load(PUBLIC/'fresh_rate_selected_sources.npz');seal=json.loads((PUBLIC/'fresh_rate_identity_seal.json').read_text());source=s['source_audit']
    assert seal==s['identity_seal'] and seal['protocol_sha256']==s['protocol_sha256'] and seal['status']=='identities_sealed_before_selected_labels'
    assert seal['selected_sources_sha256']==source['selected_sources_sha256']==sha(PUBLIC/'fresh_rate_selected_sources.npz')
    assert source['identity_seal_sha256']==sha(PUBLIC/'fresh_rate_identity_seal.json') and source['all_source_manifest_sha256']==sha(PUBLIC/'fresh_rate_all_source_manifest.json')
    assert source['status']=='validated' and source['protocol_sha256']==s['protocol_sha256'] and source['auditor_sha256']==sha(HERE/'audit_fresh_rate.py')
    assert source['ledger_sha256']==seal['ledger_sha256']==p['ledger_sha256'] and source['source_sha256']==p['source_sha256']
    eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==p['eligible_rows']==source['eligible_rows']==1903517
    pool=np.random.default_rng(p['streams']['pool_split']).permutation(eligible);half=len(pool)//2;unique={}
    for role,n,part in [('calibration',1000,pool[:half]),('training',900,pool[half:])]:
        ids=selected[role+'_source_row'];assert ids.shape==(32,n) and ids.dtype==np.int64
        for seed in range(32):
            expected=np.random.default_rng(p['streams'][role+'_source']+seed).choice(part,n,replace=False)
            assert np.array_equal(ids[seed],expected) and len(np.unique(ids[seed]))==n
        unique[role]=np.unique(ids);assert np.intersect1d(unique[role],excluded).size==0
        assert source[role+'_unique']==seal[role+'_unique']==len(unique[role])
        assert source[role+'_crossbank_repeats']==32*n-len(unique[role])
    assert np.intersect1d(unique['calibration'],unique['training']).size==0
    union=np.union1d(unique['calibration'],unique['training']);assert np.array_equal(union,selected['all_source_row'])
    assert source['new_unique_source_ids']==source['independent_csv_answers_verified']==len(union)
    assert source['updated_ripper_ledger_union']==len(excluded)+len(union)
    assert all(source[key]==0 for key in ['within_bank_repeats','calibration_training_overlap','declared_ledger_overlap','new_solver_inputs'])
    assert sha(PUBLIC/'fresh_rate_certificates.npz')==s['certificate_export_sha256'];a=load(PUBLIC/'fresh_rate_certificates.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']==['original','top810','degree20']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories=[];proofs=mutations=0
    for seed in range(32):
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')};audit=s['bank_audits'][seed]
        assert audit['seed']==seed and audit['status']=='validated' and audit['protocol_sha256']==s['protocol_sha256'] and audit['auditor_sha256']==sha(HERE/'audit_fresh_rate.py')
        acquired=audit['acquisition_audit'];assert acquired['status']=='validated' and acquired['protocol_sha256']==s['protocol_sha256']
        assert acquired['auditor_sha256']==sha(HERE/'audit_fresh_rate_acquisition.py') and acquired['independent_masks_replayed']==900
        assert acquired['training_queries']==22500 and acquired['calibration_queries']==2000 and acquired['log_lr_values_verified']==900*3240
        m=z['joint_count'];c=z['equal_count'];assert m.shape==c.shape==(3240,) and ((0<=c)&(c<=m)&(m<=900)).all() and int(m.sum())==270000
        cost=independent_rate_costs(m,c);assert np.array_equal(cost,z['cost'])
        top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,z['top810_support'])
        t,y,upper=[z[name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
        certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==audit['certificate_verified'];proofs+=certified
        category='unresolved';support=None
        if certified:
            assert np.array_equal(z['primal_values']*2,t) and np.all(np.abs(z['row_dual_values']*2-y)<=1e-6)
            for k in range(3):
                changed=[t.copy(),y.copy(),upper.copy()];changed[k][0]+=1
                assert not check_certificate(81,20,ii,jj,cost,*changed);mutations+=1
            if np.isin(t,[0,2]).all():support=t==2;category='success' if np.array_equal(support,truth) else 'resolved_failure_nonexact'
            else:category='resolved_failure_fractional'
        assert category==audit['classification'];categories.append(category)
        for arm,graph in [('original',z['original_support']),('top810',top),('degree20',support)]:
            if graph is None:continue
            row=next(row for row in s['rows'] if row['seed']==seed and row['arm']==arm)
            missing=int((truth&~graph).sum());extra=int((~truth&graph).sum())
            assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
            assert row['retained']==int(graph.sum()) and row['support_bytes_sha256']==hashlib.sha256(graph.tobytes()).hexdigest()
    successes=categories.count('success');unresolved=categories.count('unresolved');decision='supported' if successes>=29 else ('falsified' if successes+unresolved<29 else 'inconclusive')
    assert s['primary']==dict(decision=decision,certified_integral_exact_graphs=successes,resolved_failures=32-successes-unresolved,unresolved=unresolved,
                              success_lower_bound=successes,success_upper_bound=successes+unresolved,required=29)
    for group in s['grouped'][:2]:
        rows=[r for r in s['rows'] if r['arm']==group['arm']];assert len(rows)==32 and group['exact_graphs']==sum(r['exact_graph'] for r in rows)
        assert group['missing_true_total']==sum(r['missing_true_edges'] for r in rows) and group['extra_total']==sum(r['extra_edges'] for r in rows)
    for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),('certified_fractional','resolved_failure_fractional'),('unresolved','unresolved')]:assert s['grouped'][2][field]==categories.count(category)
    assert s['query_accounting']==p['query_accounting']==dict(training_per_bank=22500,calibration_per_bank=2000,total_per_bank=24500,new_training_queries=720000,new_calibration_queries=64000,new_queries=784000)
    rt=json.loads((PUBLIC/'fresh_rate_runtime.json').read_text());assert rt==s['runtime'] and rt['status']=='validated'
    assert rt['protocol_sha256']==s['protocol_sha256'] and rt['wrapper_sha256']==sha(HERE/'run_fresh_rate.py') and rt['wall_s']<180 and rt['aggregate_cpu_s']<900
    assert len(rt['commands'])==36 and all(r['returncode']==0 for r in rt['commands']) and sorted(r['seed'] for r in rt['commands'] if r['action']=='bank')==list(range(32))
    assert s['new_solver_inputs']==s['gpu_seconds']==0
    print(json.dumps(dict(status='passed',decision=decision,new_exact_certificates=proofs,rejected_mutated_certificates=mutations,new_unique_source_ids=len(union),updated_ripper_ledger_union=source['updated_ripper_ledger_union'],
        scope='Public exact source-ID sampling and disjointness from sealed Ripper ledger, costs, proofs and graph outcomes verified. Rebuilding the historical ledger, CSV labels, predictable masks and corruption counts requires the raw archive.')))


if __name__=='__main__':main()
