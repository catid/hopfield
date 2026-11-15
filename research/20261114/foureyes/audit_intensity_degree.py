"""Independent public integer-proof replay for H4ah; no solver call."""
import json,sys
from pathlib import Path
import numpy as np
from intensity_degree_study import ROOT,HERE,RIPPER,OUT,PUBLIC,PROTOCOL,PARENT,CONDITIONS,sha,save
from audit_degree_prior import check_certificate


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def evaluate(index,arrays,receipt,parent,protocol):
    n=81;degree=20;ii,jj=np.triu_indices(n,1);cost=parent['counts'][index]
    assert cost.dtype==np.int64 and np.array_equal(cost,arrays['cost'])
    assert receipt['index']==index and receipt['group']==index//2 and receipt['condition']==CONDITIONS[index%2]
    assert receipt['protocol_sha256']==sha(PROTOCOL)
    solver=receipt['solver'];options=solver['solver_options']
    assert solver['highs_version']==protocol['solver']['highspy'] and solver['numpy']==protocol['solver']['numpy']
    for key,val in protocol['solver'].items():
        if key in ['highspy','numpy']:continue
        if key=='random_seed_base':assert options['random_seed']==val+index
        else:assert options[key]==val
    t,y,z=[arrays[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    certified=check_certificate(n,degree,ii,jj,cost,t,y,z)
    assert certified==receipt['certificate']['certified']
    x=arrays['primal_values'];node=arrays['node_dual_values'];mutations=0
    if certified:
        assert solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']
        assert x.dtype==node.dtype==np.float64 and np.isin(x,[0.,.5,1.]).all() and np.array_equal(2*x,t)
        assert node.shape==(81,) and np.max(abs(2*node-y))<=1e-6
        assert receipt['certificate']['primal_rounding_used'] is False
        assert receipt['certificate']['primal_objective_twice']==receipt['certificate']['dual_objective_twice']==sum(int(c)*int(v) for c,v in zip(cost,t))
        assert solver['solver_objective']==receipt['certificate']['primal_objective_twice']/2
        assert receipt['certificate']['integral']==bool(np.isin(t,[0,2]).all())
        for k in range(3):
            bad=[t.copy(),y.copy(),z.copy()];bad[k][0]+=1
            assert not check_certificate(n,degree,ii,jj,cost,*bad);mutations+=1
    # Only evaluated after certificate checks; geometry does not enter costs or LP.
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    order=sorted(range(3240),key=lambda k:(int(cost[k]),k));top=np.zeros(3240,bool);top[order[:810]]=True
    assert np.array_equal(top,arrays['top810_support']) and arrays['top810_support'].dtype==bool
    missing=extra=None
    if certified and np.isin(t,[0,2]).all():
        graph=t==2;missing=int((truth&~graph).sum());extra=int((~truth&graph).sum());assert missing==extra
        category='success' if missing==extra==0 else 'resolved_failure_nonexact'
    elif certified:category='resolved_failure_fractional'
    else:category='unresolved'
    result=dict(index=index,group=index//2,condition=CONDITIONS[index%2],classification=category,missing_true=missing,extra=extra,
                certified=certified,fractional_variables=int((t==1).sum()) if certified else None,
                top810_exact=bool(np.array_equal(top,truth)),top810_missing=int((truth&~top).sum()),top810_extra=int((~truth&top).sum()),
                archived_regular_exact=bool(np.array_equal(parent['supports'][index,3],truth)),mutated_proofs_rejected=mutations)
    return result


def aggregate(rows):
    grouped=[];decisions={}
    for c in CONDITIONS:
        part=[r for r in rows if r['condition']==c];assert len(part)==64
        counts={kind:sum(r['classification']==kind for r in part) for kind in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']}
        lower=counts['success'];upper=lower+counts['unresolved']
        decisions[c]=dict(decision='supported' if lower>=61 else 'falsified' if upper<61 else 'inconclusive',exact_lower=lower,exact_upper=upper,required=61)
        grouped.append(dict(condition=c,**counts,integral_graphs_with_error_counts=sum(r['missing_true'] is not None for r in part),
                            missing_total_integral_only=sum(r['missing_true'] or 0 for r in part),extra_total_integral_only=sum(r['extra'] or 0 for r in part),
                            top810_exact=sum(r['top810_exact'] for r in part),top810_missing=sum(r['top810_missing'] for r in part),top810_extra=sum(r['top810_extra'] for r in part)))
    return grouped,decisions


def export_and_audit():
    p=json.loads(PROTOCOL.read_text());parent=npz(PARENT/'models.npz');rows=[];receipts=[];arrays={}
    assert parent['groups'].tolist()==np.repeat(np.arange(64),2).tolist() and parent['conditions'].tolist()==CONDITIONS*64
    for index in range(128):
        d=OUT/f'j{index:03d}';receipt=json.loads((d/'receipt.json').read_text());a=npz(d/'model.npz')
        assert receipt['model_sha256']==sha(d/'model.npz') and receipt['log_sha256']==sha(d/'highs.log')
        rows.append(evaluate(index,a,receipt,parent,p));receipts.append(receipt)
        arrays.update({f'j{index:03d}_{key}':value for key,value in a.items()})
    grouped,decisions=aggregate(rows);PUBLIC.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(PUBLIC/'certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(Path(__file__)),source_models_sha256=sha(PARENT/'models.npz'),
                certificates_sha256=sha(PUBLIC/'certificates.npz'),rows=rows,receipts=receipts,grouped=grouped,primary=decisions[CONDITIONS[1]],control=decisions[CONDITIONS[0]],
                mutations_rejected=sum(r['mutated_proofs_rejected'] for r in rows),certified_proofs=sum(r['certified'] for r in rows),
                source_scope=p['scope'],new_source_rows=0,new_noise_draws=0,new_solver_inputs=0,gpu_seconds=0,
                error_total_scope='Missing/extra totals describe only integral certified returned graphs. Fractional outcomes remain separate procedure failures; unknown outputs remain unresolved.')
    save(PUBLIC/'summary.json',result)
    print(json.dumps(dict(status='validated',grouped=grouped,primary=result['primary'],control=result['control'])),flush=True)


def main():
    p=json.loads(PROTOCOL.read_text());s=json.loads((PUBLIC/'summary.json').read_text());assert s['status']=='validated'
    assert s['protocol_sha256']==sha(PROTOCOL) and s['auditor_sha256']==sha(Path(__file__))
    skipped=[]
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():
        path=ROOT/rel
        if path.exists():assert sha(path)==digest,rel
        else:assert rel.startswith('artifacts/');skipped.append(rel)
    assert s['source_models_sha256']==sha(PARENT/'models.npz') and s['certificates_sha256']==sha(PUBLIC/'certificates.npz')
    a=npz(PUBLIC/'certificates.npz');parent=npz(PARENT/'models.npz');rows=[]
    for index in range(128):
        values={key.split('_',1)[1]:value for key,value in a.items() if key.startswith(f'j{index:03d}_')}
        rows.append(evaluate(index,values,s['receipts'][index],parent,p))
    grouped,decisions=aggregate(rows);assert rows==s['rows'] and grouped==s['grouped'] and s['primary']==decisions[CONDITIONS[1]] and s['control']==decisions[CONDITIONS[0]]
    assert s['certified_proofs']==sum(r['certified'] for r in rows) and s['mutations_rejected']==sum(r['mutated_proofs_rejected'] for r in rows)
    runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime['status']=='complete' and runtime['error'] is None and runtime['returncodes']==[0]*8
    assert runtime['protocol_sha256']==sha(PROTOCOL) and runtime['wall_seconds']<p['budget']['wall_seconds'] and runtime['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
    assert runtime['samples'][-1]['completed']==128 and runtime['samples'][-1]['active_pids']==[]
    print(json.dumps(dict(status='passed',proofs=s['certified_proofs'],mutation_rejections=s['mutations_rejected'],primary=s['primary'],control=s['control'],scope='Solver-free public replay of all counts, proofs, raw primal/dual values and finite endpoints. Proofs establish optimality, not uniqueness.')))


if __name__=='__main__':main()
