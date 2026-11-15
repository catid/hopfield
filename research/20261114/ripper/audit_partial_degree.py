"""Independent exact grid-proof, Fraction-rate and degree-selection audit."""
import argparse,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from partial_degree_study import ROOT,HERE,OUT,PUBLIC,PARENT,PROTOCOL,ARMS,sha,npz,verify
from audit_degree_prior import check_certificate
from audit_laplace_rate import independent_rate_costs


def literal_curvature(values):
    v=list(map(int,values));second=[v[d+1]+v[d-1]-2*v[d] for d in range(1,len(v)-1)]
    degree=sorted(range(1,len(v)-1),key=lambda d:(-second[d-1],d))[0]
    return degree,second


def grid(arrays,records,cost,solver_config,index,n=81):
    ii,jj=np.triu_indices(n,1);t0,y0,z0=[arrays['d00_'+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert all((a==0).all() for a in [t0,y0,z0]) and check_certificate(n,0,ii,jj,cost,t0,y0,z0)
    values=[0];certified=fractional=mutations=0;unresolved=[]
    for degree,record in enumerate(records,1):
        assert record['degree']==degree;solver=record['solver'];options=solver['solver_options']
        assert solver['highs_version']==solver_config['highspy'] and solver['numpy']==solver_config['numpy']
        for key,value in solver_config.items():
            if key in ['highspy','numpy']:continue
            if key=='random_seed_base':assert options['random_seed']==value+index*41+degree
            else:assert options[key]==value,(key,options[key],value)
        prefix=f'd{degree:02d}_';t,y,z=[arrays[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
        ok=check_certificate(n,degree,ii,jj,cost,t,y,z);assert ok==record['certificate']['certified']
        if not ok:values.append(None);unresolved.append(degree);continue
        certified+=1;raw=arrays[prefix+'primal_values'];dual=arrays[prefix+'node_dual_values'];cert=record['certificate']
        assert solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']
        assert raw.dtype==dual.dtype==np.float64 and np.isin(raw,[0.,.5,1.]).all() and np.array_equal(2*raw,t)
        assert dual.shape==(n,) and np.max(abs(2*dual-y))<=1e-6
        value=sum(int(c)*int(v) for c,v in zip(cost,t,strict=True))
        assert cert['primal_objective_twice']==cert['dual_objective_twice']==value and abs(solver['solver_objective']-value/2)<1e-5
        integral=bool(np.isin(t,[0,2]).all());assert cert['integral']==integral and cert['primal_rounding_used'] is False
        if n%2 and degree%2:assert not integral,'Odd n and d exclude an integral regular graph'
        fractional+=int(not integral);values.append(value)
        for which in range(3):
            bad=[t.copy(),y.copy(),z.copy()];bad[which][0]+=1
            assert not check_certificate(n,degree,ii,jj,cost,*bad);mutations+=1
    selected=second=ties=None
    if not unresolved:
        selected,second=literal_curvature(values);assert min(second)>=0
        ties=[d for d in range(1,len(values)-1) if second[d-1]==max(second)];assert selected==min(ties)
    return dict(objective_twice=values,curvature_twice=second,maximum_curvature_tie_degrees=ties,selected_degree=selected,certified_grid_LPs=certified,intermediate_fractional_LPs=fractional,unresolved_degrees=unresolved,mutations_rejected=mutations,analytic_zero_proof=True)


def audit_bank(index,public=False):
    p=verify();seed,arm_index=divmod(index,2);arm=ARMS[arm_index];prefix=f's{seed}_{arm}_';d=(PUBLIC if public else OUT)/f'j{index:03d}'
    receipt=json.loads((d/'receipt.json').read_text());assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm and receipt['protocol_sha256']==sha(PROTOCOL)
    assert receipt['models_sha256']==sha(d/'models.npz');arrays=npz(d/'models.npz');assert len(receipt['records'])==40
    with np.load(PARENT,allow_pickle=False) as z:parent={k:z[prefix+k].copy() for k in ['cost','joint_count','equal_count','primal_twice','node_dual_twice','upper_dual_twice','top810_support']}
    for name in ['cost','joint_count','equal_count']:assert np.array_equal(arrays[name],parent[name])
    cost=independent_rate_costs(arrays['joint_count'],arrays['equal_count']);assert cost.dtype==np.int64 and np.array_equal(cost,arrays['cost'])
    if not public:
        for record in receipt['records']:assert record['log_sha256']==sha(d/f'd{record["degree"]:02d}.log')
    row=grid(arrays,receipt['records'],cost,p['solver'],index);selected=row['selected_degree'];classification='unresolved';missing=extra=retained=support_sha=None
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    assert check_certificate(81,20,ii,jj,cost,*[parent[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']])
    archived_exact=bool(np.isin(parent['primal_twice'],[0,2]).all() and np.array_equal(parent['primal_twice']==2,truth))
    if selected is not None:
        t=arrays[f'd{selected:02d}_primal_twice']
        if np.isin(t,[0,2]).all():
            support=t==2;missing=int((truth&~support).sum());extra=int((~truth&support).sum());retained=int(support.sum());assert retained==81*selected//2
            support_sha=hashlib.sha256(support.tobytes()).hexdigest();classification='success' if missing==extra==0 else 'resolved_failure_nonexact'
        else:classification='resolved_failure_fractional'
    return dict(status='validated',index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),models_sha256=sha(d/'models.npz'),receipt_sha256=sha(d/'receipt.json'),
                **row,classification=classification,missing_true=missing,extra=extra,retained=retained,selected_support_bytes_sha256=support_sha,archived_fixed_degree_exact=archived_exact,
                archived_top810_exact=bool(np.array_equal(parent['top810_support'],truth)))


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(64));groups=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        groups.append(dict(arm=arm,**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
            selected_degree20=sum(r['selected_degree']==20 for r in part),selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},
            missing_integral_only=sum(r['missing_true'] or 0 for r in part),extra_integral_only=sum(r['extra'] or 0 for r in part),archived_fixed_degree_exact=sum(r['archived_fixed_degree_exact'] for r in part),archived_top810_exact=sum(r['archived_top810_exact'] for r in part)))
    a=groups[0];endpoints={}
    for name,key,required in [('degree20','selected_degree20',31),('exact_graph','success',27)]:
        lower=a[key];upper=lower+a['unresolved'];endpoints[name]=dict(decision='supported' if lower>=required else 'falsified' if upper<required else 'inconclusive',lower=lower,upper=upper,required=required)
    return groups,endpoints


def public_job(index):return audit_bank(index,True)


def public_audit():
    start=time.monotonic();p=verify();saved=json.loads((PUBLIC/'summary.json').read_text());assert saved['protocol_sha256']==sha(PROTOCOL)
    for path,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/path)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(public_job,range(64)))
    grouped,endpoints=aggregate(rows);assert rows==saved['rows'] and grouped==saved['grouped'] and endpoints==saved['endpoints']
    assert sum(r['certified_grid_LPs'] for r in rows)==saved['certified_LPs'] and sum(r['mutations_rejected'] for r in rows)==saved['mutations_rejected']
    print(json.dumps(dict(status='passed',endpoints=endpoints,grouped=grouped,certified_LPs=saved['certified_LPs'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,
        scope='Reconstructs unchanged rates from exported counts, all LP proofs, degree selection, outcomes and archived graph controls. Original-label and acquisition provenance require frozen H1bh audit.')))


if __name__=='__main__':public_audit()
