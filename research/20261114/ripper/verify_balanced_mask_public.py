"""Portable exposure-balanced mask generation, paired query accounting and exact proofs."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_balanced_mask import reference_masks,exposure_statistics
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    manifest=json.loads((PUBLIC/'balanced_mask_public_manifest.json').read_text())
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    p=json.loads((HERE/'protocol-balanced-mask.json').read_text());s=json.loads((PUBLIC/'balanced_mask.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-balanced-mask.json') and s['all_banks_retained']
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,path
    assert p['seeds']==list(range(32)) and p['rows']==900 and p['mask_seed_base']==202611790000
    assert sha(PUBLIC/'balanced_mask_certificates.npz')==s['certificate_export_sha256'];a=load(PUBLIC/'balanced_mask_certificates.npz')
    old=json.loads((PUBLIC/'fresh_rate.json').read_text());assert old['certificate_export_sha256']==sha(PUBLIC/'fresh_rate_certificates.npz');baseline=load(PUBLIC/'fresh_rate_certificates.npz')
    uniform_summary=json.loads((PUBLIC/'uniform_mask.json').read_text());assert uniform_summary['certificate_export_sha256']==sha(PUBLIC/'uniform_mask_certificates.npz');uniform_baseline=load(PUBLIC/'uniform_mask_certificates.npz')
    assert np.array_equal(a['banks'],np.arange(32)) and a['arms'].tolist()==p['arms']==['adaptive_degree20','archived_uniform_degree20','balanced_top810','balanced_degree20']
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    categories=[];uniform_categories=[];new_proofs=old_proofs=mutations=0;accounting=[]
    for seed in range(32):
        z={key.removeprefix(f's{seed}_'):value for key,value in a.items() if key.startswith(f's{seed}_')};audit=s['bank_audits'][seed]
        assert audit['seed']==seed and audit['status']=='validated' and audit['protocol_sha256']==s['protocol_sha256'] and audit['auditor_sha256']==sha(HERE/'audit_balanced_mask.py')
        assert audit['independent_masks_replayed']==audit['common_noise_rows_replayed']==900
        assert audit['baseline_sha256']==p['bank_sha256'][str(seed)][f'artifacts/research/20261114/ripper/fresh_rate/s{seed}/recovery.npz']
        balanced=reference_masks(p['mask_seed_base']+seed);adaptive=z['adaptive_mask'];uniform=z['uniform_mask']
        assert np.array_equal(balanced,z['balanced_mask']) and adaptive.shape==uniform.shape==balanced.shape==(900,81)
        assert adaptive.dtype==uniform.dtype==bool and (adaptive.sum(1)==25).all() and (uniform.sum(1)==25).all()
        assert np.array_equal(uniform,uniform_baseline[f's{seed}_uniform_mask']) and np.array_equal(adaptive,uniform_baseline[f's{seed}_adaptive_mask'])
        previous=adaptive|uniform;reused=int((balanced&previous).sum());new=int((balanced&~previous).sum());prior_union=24500+int((uniform&~adaptive).sum())
        accounting.append(dict(reused_queries=reused,newly_exposed_queries=new,balanced_training_queries=22500,new_calibration_queries=0,prior_cumulative_union_queries=prior_union,combined_union_queries=prior_union+new))
        assert accounting[-1]==audit['query_accounting'] and reused+new==22500
        for key,value in z.items():
            if key.startswith('baseline_'):assert np.array_equal(value,baseline[f's{seed}_'+key.removeprefix('baseline_')])
            if key.startswith('uniformbaseline_'):assert np.array_equal(value,uniform_baseline[f's{seed}_'+key.removeprefix('uniformbaseline_')])
        m=z['joint_count'];c=z['equal_count'];exposed=balanced[:,ii]&balanced[:,jj]
        assert np.array_equal(m,exposed.sum(0)) and ((0<=c)&(c<=m)&(m<=900)).all() and int(m.sum())==270000
        cost=independent_rate_costs(m,c);assert np.array_equal(cost,z['cost']);top=np.zeros(3240,bool)
        top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,z['top810_support'])
        oldcost=independent_rate_costs(z['baseline_joint_count'],z['baseline_equal_count']);assert np.array_equal(oldcost,z['baseline_cost'])
        uniformcost=independent_rate_costs(z['uniformbaseline_joint_count'],z['uniformbaseline_equal_count']);assert np.array_equal(uniformcost,z['uniformbaseline_cost'])
        diagnostic={arm:exposure_statistics(counts) for arm,counts in [('balanced',m),('uniform',z['uniformbaseline_joint_count']),('adaptive',z['baseline_joint_count'])]}
        assert diagnostic==audit['exposure_diagnostics'] and dict(seed=seed,**diagnostic)==s['exposure_diagnostics'][seed]
        for mask,counts in [(uniform,z['uniformbaseline_joint_count']),(adaptive,z['baseline_joint_count'])]:assert np.array_equal((mask[:,ii]&mask[:,jj]).sum(0),counts)
        supports={'balanced_top810':top};category='unresolved';ucategory='unresolved'
        for prefix,weights in [('baseline_',oldcost),('uniformbaseline_',uniformcost),('',cost)]:
            t,y,upper=[z[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
            certified=check_certificate(81,20,ii,jj,weights,t,y,upper)
            if prefix:assert certified and audit['archived_optimality_certificates_verified'];old_proofs+=1
            else:assert certified==audit['certificate_verified'];new_proofs+=certified
            if certified:
                assert np.array_equal(z[prefix+'primal_values']*2,t) and np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
                for k in range(3):
                    changed=[t.copy(),y.copy(),upper.copy()];changed[k][0]+=1
                    assert not check_certificate(81,20,ii,jj,weights,*changed);mutations+=1
                if np.isin(t,[0,2]).all():
                    graph=t==2
                    if prefix=='baseline_':assert np.array_equal(graph,truth);supports['adaptive_degree20']=graph
                    elif prefix=='uniformbaseline_':ucategory='success' if np.array_equal(graph,truth) else 'resolved_failure_nonexact';supports['archived_uniform_degree20']=graph
                    else:category='success' if np.array_equal(graph,truth) else 'resolved_failure_nonexact';supports['balanced_degree20']=graph
                elif prefix=='uniformbaseline_':ucategory='resolved_failure_fractional'
                elif not prefix:category='resolved_failure_fractional'
        assert category==audit['classification'];categories.append(category)
        assert ucategory==audit['uniform_classification'];uniform_categories.append(ucategory)
        for arm,graph in supports.items():
            row=next(row for row in s['rows'] if row['seed']==seed and row['arm']==arm);missing=int((truth&~graph).sum());extra=int((~truth&graph).sum())
            assert row['missing_true_edges']==missing and row['extra_edges']==extra and row['exact_graph']==(missing==extra==0)
            assert row['retained']==int(graph.sum()) and row['support_bytes_sha256']==hashlib.sha256(graph.tobytes()).hexdigest()
    success=categories.count('success');unresolved=categories.count('unresolved');decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive')
    assert s['primary']==dict(decision=decision,certified_integral_exact_graphs=success,resolved_failures=32-success-unresolved,unresolved=unresolved,success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    for group in [s['grouped'][0],s['grouped'][2]]:
        rows=[r for r in s['rows'] if r['arm']==group['arm']];assert len(rows)==32 and group['exact_graphs']==sum(r['exact_graph'] for r in rows)
        assert group['missing_true_total']==sum(r['missing_true_edges'] for r in rows) and group['extra_total']==sum(r['extra_edges'] for r in rows)
    for field,category in [('certified_integral_exact','success'),('certified_integral_nonexact','resolved_failure_nonexact'),('certified_fractional','resolved_failure_fractional'),('unresolved','unresolved')]:assert s['grouped'][3][field]==categories.count(category) and s['grouped'][1][field]==uniform_categories.count(category)
    total={key:sum(row[key] for row in accounting) for key in accounting[0]};assert total==s['physical_query_accounting']
    assert total['reused_queries']+total['newly_exposed_queries']==720000 and total['combined_union_queries']==1281927+total['newly_exposed_queries'] and total['prior_cumulative_union_queries']==1281927
    assert s['alternative_query_accounting']==p['query_accounting']==dict(balanced_training_per_bank=22500,balanced_calibration_per_bank=0,balanced_alternative_total=720000,uniform_alternative_total=720000,adaptive_alternative_total=784000,prior_cumulative_union=1281927)
    assert s['physical_query_rule']==p['physical_query_rule'] and all(s[key]==0 for key in ['new_source_ids','new_latent_draws','new_solver_inputs','gpu_seconds'])
    rt=json.loads((PUBLIC/'balanced_mask_runtime.json').read_text());assert rt==s['runtime'] and rt['status']=='validated' and rt['protocol_sha256']==s['protocol_sha256']
    assert rt['wrapper_sha256']==sha(HERE/'run_balanced_mask.py') and rt['wall_s']<60 and rt['aggregate_cpu_s']<120 and len(rt['commands'])==34
    assert all(r['returncode']==0 for r in rt['commands']) and sorted(r['seed'] for r in rt['commands'] if r['action']=='bank')==list(range(32))
    print(json.dumps(dict(status='passed',decision=decision,new_exact_certificates=new_proofs,archived_exact_certificates=old_proofs,rejected_mutated_certificates=mutations,
        physical_query_accounting=total,scope='Public balanced masks, all-pair exposure diagnostics, cumulative overlap accounting, rate costs, proofs and graph outcomes verified. Common-noise labels/equality counts and original adaptive-mask provenance require the raw archive.')))


if __name__=='__main__':main()
