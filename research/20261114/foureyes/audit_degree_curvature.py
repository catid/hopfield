"""H4aj public exact proofs and independent discrete-curvature selection."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from degree_curvature_study import ROOT,HERE,RIPPER,OUT,PUBLIC,PARENT,PROTOCOL,CONDITIONS,sha,save
from audit_degree_prior import check_certificate


def literal_curvature(values):
    v=list(map(int,values));second=[v[d+1]+v[d-1]-2*v[d] for d in range(1,len(v)-1)]
    degree=sorted(range(1,len(v)-1),key=lambda d:(-second[d-1],d))[0]
    return degree,np.array(second,np.int64)


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def bank(index,arrays,receipt,cost,p):
    ii,jj=np.triu_indices(81,1);assert np.array_equal(arrays['cost'],cost) and cost.dtype==np.int64
    assert receipt['index']==index and receipt['group']==index//2 and receipt['condition']==CONDITIONS[index%2] and receipt['protocol_sha256']==sha(PROTOCOL)
    t0,y0,z0=[arrays['d00_'+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert all((v==0).all() for v in [t0,y0,z0]) and check_certificate(81,0,ii,jj,cost,t0,y0,z0)
    assert len(receipt['records'])==40
    values=[0];certified=mutations=fractional=0;unresolved=[]
    for degree,record in enumerate(receipt['records'],1):
        assert record['degree']==degree;solver=record['solver'];options=solver['solver_options']
        assert solver['highs_version']==p['solver']['highspy'] and solver['numpy']==p['solver']['numpy']
        for key,value in p['solver'].items():
            if key in ['highspy','numpy']:continue
            if key=='random_seed_base':assert options['random_seed']==value+index*41+degree
            else:assert options[key]==value
        prefix=f'd{degree:02d}_';t,y,z=[arrays[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
        ok=check_certificate(81,degree,ii,jj,cost,t,y,z);assert ok==record['certificate']['certified']
        if not ok:values.append(None);unresolved.append(degree);continue
        certified+=1;raw=arrays[prefix+'primal_values'];dual=arrays[prefix+'node_dual_values']
        assert solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']
        assert raw.dtype==dual.dtype==np.float64 and np.isin(raw,[0.,.5,1.]).all() and np.array_equal(2*raw,t)
        assert dual.shape==(81,) and np.max(abs(2*dual-y))<=1e-6
        cert=record['certificate'];value=sum(int(c)*int(v) for c,v in zip(cost,t))
        assert cert['primal_objective_twice']==cert['dual_objective_twice']==value and solver['solver_objective']==value/2
        integral=bool(np.isin(t,[0,2]).all());assert cert['integral']==integral and cert['primal_rounding_used'] is False
        if degree%2:assert not integral,'An81-node integral odd-degree graph is impossible'
        fractional+=int(not integral);values.append(value)
        for which in range(3):
            bad=[t.copy(),y.copy(),z.copy()];bad[which][0]+=1
            assert not check_certificate(81,degree,ii,jj,cost,*bad);mutations+=1
    selected=None;second=None;ties=None;missing=extra=None;classification='unresolved';support=None
    if not unresolved:
        selected,curvatures=literal_curvature(values);assert (curvatures>=0).all(),'Exact LP degree value must be convex'
        second=curvatures.tolist();ties=[d for d in range(1,40) if curvatures[d-1]==max(curvatures)]
        assert selected==min(ties);t=arrays[f'd{selected:02d}_primal_twice']
        if np.isin(t,[0,2]).all():
            support=t==2;truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert int(support.sum())==81*selected//2
            classification='success' if missing==extra==0 else 'resolved_failure_nonexact'
        else:classification='resolved_failure_fractional'
    row=dict(index=index,group=index//2,condition=CONDITIONS[index%2],classification=classification,objective_twice=values,curvature_twice=second,
             maximum_curvature_tie_degrees=ties,selected_degree=selected,missing_true=missing,extra=extra,
             retained=None if support is None else int(support.sum()),selected_support_bytes_sha256=None if support is None else __import__('hashlib').sha256(support.tobytes()).hexdigest(),
             certified_grid_LPs=certified,intermediate_fractional_LPs=fractional,unresolved_degrees=unresolved,mutations_rejected=mutations,analytic_zero_proof=True)
    return row


def aggregate(rows):
    groups=[];decisions={}
    for condition in CONDITIONS:
        part=[r for r in rows if r['condition']==condition];assert len(part)==64
        counts={k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']}
        exact=counts['success'];upper=exact+counts['unresolved']
        decisions[condition]=dict(decision='supported' if exact>=61 else 'falsified' if upper<61 else 'inconclusive',exact_lower=exact,exact_upper=upper,required=61)
        histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)}
        groups.append(dict(condition=condition,**counts,selected_degree_histogram=histogram,integral_graphs_with_error_counts=sum(r['missing_true'] is not None for r in part),
                           missing_total_integral_only=sum(r['missing_true'] or 0 for r in part),extra_total_integral_only=sum(r['extra'] or 0 for r in part)))
    return groups,decisions


def audit(export=False):
    start=time.monotonic();cpu=time.process_time();p=json.loads(PROTOCOL.read_text())
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    parent=npz(PARENT/'models.npz');assert parent['counts'].shape==(128,3240) and parent['groups'].tolist()==np.repeat(np.arange(64),2).tolist() and parent['conditions'].tolist()==CONDITIONS*64
    if export:PUBLIC.mkdir(parents=True,exist_ok=False)
    rows=[];files={}
    for index in range(128):
        source=(OUT if export else PUBLIC)/f'j{index:03d}';receipt=json.loads((source/'receipt.json').read_text())
        assert receipt['models_sha256']==sha(source/'models.npz');arrays=npz(source/'models.npz')
        if export:
            for record in receipt['records']:assert record['log_sha256']==sha(source/f'd{record["degree"]:02d}.log')
            dest=PUBLIC/f'j{index:03d}';dest.mkdir();shutil.copyfile(source/'receipt.json',dest/'receipt.json');shutil.copyfile(source/'models.npz',dest/'models.npz')
        rows.append(bank(index,arrays,receipt,parent['counts'][index],p))
        for name in ['receipt.json','models.npz']:files[f'j{index:03d}/{name}']=sha(source/name)
        assert time.monotonic()-start<p['budget']['wall_seconds'],'audit_wall_ceiling'
    grouped,decisions=aggregate(rows)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(Path(__file__)),rows=rows,grouped=grouped,
                primary=decisions[CONDITIONS[1]],control=decisions[CONDITIONS[0]],public_sha256=files,
                certified_LPs=sum(r['certified_grid_LPs'] for r in rows),analytic_zero_proofs=128,mutations_rejected=sum(r['mutations_rejected'] for r in rows),
                all128_banks_retained=True,all5120_LPs_retained=True,source_scope=p['scope'],new_source_rows=0,new_corruptions=0,new_solver_inputs=0,gpu_seconds=0,
                error_total_scope='Graph errors apply only to certified integral selected primals; intermediate fractional optima supply valid objective evidence.')
    timing=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    if export:save(PUBLIC/'summary.json',result);save(PUBLIC/'audit-timing.json',timing)
    else:
        assert result==json.loads((PUBLIC/'summary.json').read_text());runtime=json.loads((PUBLIC/'runtime.json').read_text())
        assert runtime['status']=='complete' and runtime['error'] is None and runtime['returncodes']==[0]*9 and runtime['protocol_sha256']==sha(PROTOCOL)
        assert runtime['wall_seconds']<p['budget']['wall_seconds'] and runtime['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
        assert runtime['samples'][-1]['complete_groups']==128 and runtime['samples'][-1]['complete_degrees']==5120 and runtime['samples'][-1]['active_pids']==[]
    print(json.dumps(dict(status='passed',primary=result['primary'],control=result['control'],grouped=grouped,certified_LPs=result['certified_LPs'],mutations_rejected=result['mutations_rejected'],timing=timing)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--export',action='store_true');audit(ap.parse_args().export)
