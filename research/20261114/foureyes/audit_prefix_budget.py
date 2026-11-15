"""Independent H4an prefix counts, exact LP proofs and finite endpoint checks."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from prefix_budget import ROOT,HERE,OUT,PUBLIC,PROTOCOL,FAMILIES,SIZES,CONDITIONS,sha,save
from audit_degree_prior import check_certificate
from audit_cycle_power_lp import literal_curvature,npz


def reference_counts(labels,n):
    return np.concatenate([(labels[:n,i,None]==labels[:n,i+1:]).sum(0,dtype=np.int64) for i in range(80)])


def graph(job):
    path=ROOT/f"results/research/foureyes/{job['parent']}/controls/graphs.npz"
    arrays=npz(path);i=job['parent_index'];assert arrays['groups'][i]==job['group'] and arrays['conditions'][i]==job['family']
    if job['parent']=='cycle_power':from audit_cycle_power_controls import independent_graph
    else:from audit_varying_degree_controls import independent_graph
    truth=independent_graph(arrays['orders'][i],job['family']);assert np.array_equal(truth,arrays['truths'][i]);return truth


def bank(index,arrays,receipt,cost,p,truth):
    ii,jj=np.triu_indices(81,1);assert truth.dtype==bool and truth.shape==(3240,)
    assert np.array_equal(arrays['cost'],cost) and cost.dtype==np.int64
    assert receipt['index']==index and receipt['group']==index//12 and receipt['condition']==CONDITIONS[index%12] and receipt['protocol_sha256']==sha(PROTOCOL)
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
            support=t==2
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert int(support.sum())==81*selected//2
            classification='success' if missing==extra==0 else 'resolved_failure_nonexact'
        else:classification='resolved_failure_fractional'
    row=dict(index=index,group=index//12,condition=CONDITIONS[index%12],classification=classification,objective_twice=values,curvature_twice=second,
             maximum_curvature_tie_degrees=ties,selected_degree=selected,missing_true=missing,extra=extra,
             retained=None if support is None else int(support.sum()),selected_support_bytes_sha256=None if support is None else __import__('hashlib').sha256(support.tobytes()).hexdigest(),
             certified_grid_LPs=certified,intermediate_fractional_LPs=fractional,unresolved_degrees=unresolved,mutations_rejected=mutations,analytic_zero_proof=True)
    from audit_centered_regularity import literal_center,incremental_selection
    centered,_,_=literal_center(cost);support,_,chosen,_=incremental_selection(centered)
    assert np.array_equal(support,arrays['centered_support']) and chosen==receipt['centered_selection']
    row['centered_control']=dict(exact=bool(np.array_equal(support,truth)),missing_true=int((truth&~support).sum()),extra=int((~truth&support).sum()),selection=chosen)
    d20=arrays['d20_primal_twice'];valid=receipt['records'][19]['certificate']['certified']
    row['degree20_control']=dict(certified=valid,integral=bool(d20.shape==truth.shape and np.isin(d20,[0,2]).all()) if valid else None,exact=bool(np.array_equal(d20,2*truth)) if valid else None)
    row['true_degree']=int(2*truth.sum()//81);row['true_degree_selected']=selected==row['true_degree']
    from audit_regularity_gap import independent_selection
    raw,record=independent_selection(cost);assert np.array_equal(raw,arrays['raw_support']) and record['selected']==receipt['raw_selection']
    distinct=sorted(set(map(int,cost)));median=float(np.median(cost));gaps=[(distinct[i+1]-distinct[i],i) for i in range(len(distinct)-1) if distinct[i+1]<=median]
    if gaps:
        largest=max(g for g,i in gaps);k=min(i for g,i in gaps if g==largest);cut=(distinct[k]+distinct[k+1])//2
    else:cut=0
    restricted=cost<=cut;assert cut==receipt['restricted_cut'] and np.array_equal(restricted,arrays['restricted_support'])
    for name,support in [('raw_control',raw),('restricted_control',restricted)]:row[name]=dict(exact=bool(np.array_equal(support,truth)),missing_true=int((truth&~support).sum()),extra=int((~truth&support).sum()))
    row.update(family=receipt['family'],rows=receipt['rows'])
    assert cost.shape==(3240,) and ((cost>=0)&(cost<=receipt['rows'])).all()
    return row


def validate_jobs(p):
    manifests={}
    for parent in ['varying_degree','cycle_power']:
        public=ROOT/f'results/research/foureyes/{parent}/controls/generation-manifest.json'
        raw=f'artifacts/research/20261114/foureyes/{parent}/inputs.json'
        assert sha(public)==p['input_sha256'][raw]
        manifests[parent]=json.loads(public.read_text())
    for index,j in enumerate(p['jobs']):
        assert j['index']==index and j['group']==index//12 and j['family']==FAMILIES[index%12//3] and j['rows']==SIZES[index%3] and j['condition']==CONDITIONS[index%12]
        expected_parent='varying_degree' if j['family'] in ['cycle81','cliques9'] else 'cycle_power'
        expected_index=2*j['group']+(j['family'] in ['cliques9','power3'])
        assert j['parent']==expected_parent and j['parent_index']==expected_index
        e=manifests[expected_parent]['entries'][expected_index]
        assert e['path']==j['source_path'] and e['sha256']==j['source_sha256'] and e['group']==j['group'] and e['condition']==j['family']


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(768));groups=[];decisions={}
    for fi,family in enumerate(FAMILIES):
        for ni,n in enumerate(SIZES):
            part=[r for r in rows if r['family']==family and r['rows']==n];assert len(part)==64
            counts={k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']}
            record=dict(family=family,rows=n,**counts,true_degree_selected=sum(r['true_degree_selected'] for r in part),selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},integral_graphs_with_error_counts=sum(r['missing_true'] is not None for r in part),missing_integral_only=sum(r['missing_true'] or 0 for r in part),extra_integral_only=sum(r['extra'] or 0 for r in part))
            for name in ['raw_control','centered_control','restricted_control']:record[name+'_exact']=sum(r[name]['exact'] for r in part)
            groups.append(record)
            if n==600:
                lower=counts['success'];upper=lower+counts['unresolved'];decisions[family]=dict(decision='supported' if lower>=61 else 'falsified' if upper<61 else 'inconclusive',lower=lower,upper=upper,required=61)
    nonmonotonic=[]
    for group in range(64):
        for fi,family in enumerate(FAMILIES):
            part=rows[12*group+3*fi:12*group+3*fi+3];assert [r['rows'] for r in part]==SIZES
            for a,b in zip(part,part[1:]):
                if a['classification']=='success' and b['classification'].startswith('resolved_failure'):nonmonotonic.append(dict(group=group,family=family,lower_rows=a['rows'],higher_rows=b['rows']))
    return groups,decisions,nonmonotonic


def worker(worker_id):
    p=json.loads(PROTOCOL.read_text());folder=OUT/'audits';folder.mkdir(exist_ok=True);cached=None;labels=None
    for index in [12*g+k for g in range(worker_id,64,8) for k in range(12)]:
        start=time.monotonic();cpu=time.process_time();job=p['jobs'][index];source=OUT/f'j{index:03d}';receipt=json.loads((source/'receipt.json').read_text())
        assert receipt['source_path']==job['source_path'] and receipt['source_sha256']==job['source_sha256']
        assert receipt['family']==job['family'] and receipt['rows']==job['rows']
        if cached!=job['source_path']:
            path=ROOT/job['source_path'];assert sha(path)==job['source_sha256']
            with np.load(path,allow_pickle=False) as z:labels=z['noisy'].copy()
            cached=job['source_path']
        counts=reference_counts(labels,job['rows']);assert receipt['models_sha256']==sha(source/'models.npz')
        for record in receipt['records']:assert record['log_sha256']==sha(source/f'd{record["degree"]:02d}.log')
        row=bank(index,npz(source/'models.npz'),receipt,counts,p,graph(job))
        save(folder/f'j{index:03d}.json',dict(row=row,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(Path(__file__)),models_sha256=sha(source/'models.npz'),receipt_sha256=sha(source/'receipt.json'),source_sha256=job['source_sha256'],independent_prefix_counts=True,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu))
        print(json.dumps(dict(audited=index)),flush=True)


def regenerated_source(job):
    parent=json.loads((HERE/(job['parent']+'.json')).read_text());index=job['parent_index'];order=np.random.default_rng(parent['graph_seed_base']+index).permutation(81)
    if job['parent']=='cycle_power':from audit_cycle_power_controls import independent_colorings
    else:from audit_varying_degree_controls import independent_colorings
    from audit_regularity_intensity import literal_corruption
    clean,_=independent_colorings(order,job['family'],parent['coloring_seed_base']+index)
    hot=np.sort(np.random.default_rng(parent['heatmap_seed_base']+index).choice(81,18,replace=False));rng=np.random.default_rng(parent['noise_rng_base']+index)
    noisy,_=literal_corruption(clean,np.ones_like(clean,bool),1-rng.random((2400,81)),rng.integers(1,9,(2400,81),dtype=np.int64),hot,'hot18_weight32')
    return noisy


def audit(export=False):
    start=time.monotonic();cpu=time.process_time();p=json.loads(PROTOCOL.read_text());skipped=[]
    for rel,h in (p['source_sha256']|p['input_sha256']).items():
        path=ROOT/rel
        if path.exists():assert sha(path)==h,rel
        else:assert not export and rel.startswith(('artifacts/','upstream/')),rel;skipped.append(rel)
    assert len(p['jobs'])==768 and [j['index'] for j in p['jobs']]==list(range(768))
    validate_jobs(p)
    if export:PUBLIC.mkdir(parents=True,exist_ok=False)
    rows=[];files={};cached=None;labels=None
    for index,job in enumerate(p['jobs']):
        assert job['group']==index//12 and job['condition']==CONDITIONS[index%12] and job['family']==FAMILIES[index%12//3] and job['rows']==SIZES[index%3]
        source=(OUT if export else PUBLIC)/f'j{index:03d}';receipt=json.loads((source/'receipt.json').read_text());proof=(OUT if export else PUBLIC)/'audits'/f'j{index:03d}.json';checked=json.loads(proof.read_text())
        assert receipt['models_sha256']==sha(source/'models.npz') and receipt['source_path']==job['source_path'] and receipt['source_sha256']==job['source_sha256'] and receipt['family']==job['family'] and receipt['rows']==job['rows']
        assert checked['protocol_sha256']==sha(PROTOCOL) and checked['auditor_sha256']==sha(Path(__file__))
        assert checked['models_sha256']==sha(source/'models.npz') and checked['receipt_sha256']==sha(source/'receipt.json') and checked['source_sha256']==job['source_sha256'] and checked['independent_prefix_counts']
        if export:
            dest=PUBLIC/f'j{index:03d}';dest.mkdir();shutil.copyfile(source/'receipt.json',dest/'receipt.json');shutil.copyfile(source/'models.npz',dest/'models.npz')
            (PUBLIC/'audits').mkdir(exist_ok=True);shutil.copyfile(proof,PUBLIC/'audits'/proof.name);row=checked['row']
        else:
            if cached!=job['source_path']:labels=regenerated_source(job);cached=job['source_path']
            row=bank(index,npz(source/'models.npz'),receipt,reference_counts(labels,job['rows']),p,graph(job));assert row==checked['row']
        rows.append(row)
        for name in ['receipt.json','models.npz']:files[f'j{index:03d}/{name}']=sha(source/name)
        files[f'audits/{proof.name}']=sha(proof)
        if export:assert time.monotonic()-start<p['lp_budget']['wall_seconds']
    grouped,decisions,nonmonotonic=aggregate(rows)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(Path(__file__)),rows=rows,grouped=grouped,primary=decisions,nonmonotonic_exact_recovery=nonmonotonic,public_sha256=files,certified_LPs=sum(r['certified_grid_LPs'] for r in rows),analytic_zero_proofs=768,mutations_rejected=sum(r['mutations_rejected'] for r in rows),all768_grids_retained=True,all30720_LPs_retained=True,scope=p['scope'],new_source_rows=0,new_labels=0,new_solver_inputs=0,new_training_runs=0,baseline_scope='BP AdamW not run; frozen2400-row source studies are separate contextual baselines. Threshold/count controls recomputed on same prefixes.',error_total_scope='Missing/extra counts only for certified integral selected primals; intermediatefractional validobjectiveevidence.')
    timing=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    if export:save(PUBLIC/'summary.json',result);save(PUBLIC/'export-timing.json',timing)
    else:
        assert result==json.loads((PUBLIC/'summary.json').read_text());runtime=json.loads((PUBLIC/'runtime.json').read_text())
        assert runtime['status']=='complete' and runtime['error'] is None and runtime['returncodes']==[0]*17 and runtime['protocol_sha256']==sha(PROTOCOL)
        assert runtime['wall_seconds']<p['lp_budget']['wall_seconds'] and runtime['aggregate_cpu_seconds']<p['lp_budget']['aggregate_cpu_seconds']
        assert runtime['samples'][-1]['complete_groups']==768 and runtime['samples'][-1]['complete_degrees']==30720 and runtime['samples'][-1]['active_pids']==[]
    print(json.dumps(dict(status='passed',primary=decisions,grouped=grouped,nonmonotonic=nonmonotonic,certified_LPs=result['certified_LPs'],mutations_rejected=result['mutations_rejected'],unavailable_raw_dependencies=skipped,timing=timing)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--export',action='store_true');ap.add_argument('--worker',type=int);args=ap.parse_args()
    if args.worker is not None:worker(args.worker)
    else:audit(args.export)
