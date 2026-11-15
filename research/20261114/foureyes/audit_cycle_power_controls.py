"""Independent H4am synthetic generators, corruption and estimator audits."""
import argparse,json,math,os,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from cycle_power_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITIONS,ARMS,sha,save,verify
from audit_heterogeneous_noise import load_npz,bytes_sha,canonical_support,count_support,adamw_support,audit_canonical_fit
from audit_regularity_gap import independent_selection
from audit_regularity_intensity import literal_corruption


def independent_graph(order,family,q=9):
    k=int(family[-1]);n=len(order);inverse=np.empty(n,np.int64)
    for pos,cell in enumerate(order):inverse[cell]=pos
    return np.array([min(abs(int(inverse[i])-int(inverse[j])),n-abs(int(inverse[i])-int(inverse[j])))<=k for i in range(n) for j in range(i+1,n)],bool)


def independent_colorings(order,family,seed,rows=2400,q=9,cap=10000,batch=512):
    k=int(family[-1]);n=len(order);assert n>2*k and q>k
    rng=np.random.default_rng(seed);accepted=[];indices=[];generated=0;kept=0
    while kept<rows and generated<cap:
        size=min(batch,cap-generated);path=np.empty((size,n),np.int64)
        path[:,:k]=rng.permuted(np.tile(np.arange(q),(size,1)),axis=1)[:,:k]
        for pos in range(k,n):
            choices=rng.integers(0,q-k,size,dtype=np.int64)
            allowed=np.all(np.arange(q)[None,:,None]!=path[:,None,pos-k:pos],axis=2)
            ranks=np.cumsum(allowed,axis=1)
            path[:,pos]=np.argmax(allowed&(ranks==choices[:,None]+1),axis=1)
        valid=np.ones(size,bool)
        for i in range(n):
            for j in range(i+1,n):
                if j-i>k and n-(j-i)<=k:valid&=path[:,i]!=path[:,j]
        good=np.flatnonzero(valid);accepted.append(path[good]);indices.extend((generated+good).tolist());generated+=size;kept+=len(good)
    if kept<rows:raise RuntimeError('proper_coloring_candidate_cap')
    canonical=np.concatenate(accepted)[:rows];result=np.empty((rows,n),np.uint8)
    for pos,cell in enumerate(order):result[:,cell]=canonical[:,pos]
    return result,dict(generated_candidates=generated,retained_rows=rows,accepted_candidate_indices=indices[:rows],candidate_cap=cap,batch=batch)


def sources():
    start=time.monotonic();cpu=time.process_time();p=json.loads(PROTOCOL.read_text());verify(p)
    manifest=json.loads((OUT/'inputs.json').read_text());seal=json.loads((OUT/'graph-order-seal.json').read_text());orders=load_npz(OUT/'graph-orders.npz')['orders']
    assert orders.shape==(128,81) and orders.dtype==np.int64
    assert seal['sealed_before_colorings_and_noise'] and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL)
    assert seal['orders_sha256']==sha(OUT/'graph-orders.npz') and manifest['graph_order_seal_sha256']==sha(OUT/'graph-order-seal.json')
    assert len(manifest['entries'])==128 and [e['index'] for e in manifest['entries']]==list(range(128))
    evidence={};rows=[];truths=[];all_duplicates=0;total_candidates=0
    ii,jj=np.triu_indices(81,1)
    for index,e in enumerate(manifest['entries']):
        group=index//2;family=CONDITIONS[index%2];assert e['group']==group and e['condition']==family and e['banks']==[group] and e['job_id']==f'g{group:02d}-{family}'
        order=np.random.default_rng(p['graph_seed_base']+index).permutation(81);assert np.array_equal(order,orders[index]);truth=independent_graph(order,family);truths.append(truth)
        degree=np.zeros(81,np.int64);np.add.at(degree,ii,truth);np.add.at(degree,jj,truth);assert (degree==p['true_degrees_for_evaluation_only'][family]).all()
        clean,reference=independent_colorings(order,family,p['coloring_seed_base']+index);assert reference==e['sampler_receipt'];total_candidates+=reference['generated_candidates']
        assert (clean[:,ii[truth]]!=clean[:,jj[truth]]).all()
        rng=np.random.default_rng(p['noise_rng_base']+index);hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+index).choice(81,18,replace=False))
        expected=dict(solutions=clean,free=np.ones((2400,81),bool),uniforms=1-rng.random((2400,81)),offsets=rng.integers(1,9,(2400,81),dtype=np.int64))
        path=ROOT/e['path'];assert sha(path)==e['sha256'];z=load_npz(path);evidence[e['path']]=e['sha256']
        assert set(z)==set(expected)|{'noisy','cells'}
        for key,value in expected.items():assert z[key].dtype==value.dtype and np.array_equal(z[key],value),(index,key)
        assert e['hot_cells']==manifest['hot_cells_by_input'][index]==hot.tolist()
        noisy,cells=literal_corruption(clean,z['free'],z['uniforms'],z['offsets'],hot,'hot18_weight32')
        assert np.array_equal(z['noisy'],noisy) and np.array_equal(z['cells'],cells) and ((noisy!=clean).sum(1)==9).all()
        duplicates=2400-len(np.unique(clean,axis=0));all_duplicates+=duplicates
        rows.append(dict(index=index,group=group,condition=family,true_degree=int(degree[0]),generated_candidates=reference['generated_candidates'],rows=2400,within_bank_duplicate_instances=duplicates,input_sha256=e['sha256']))
        assert time.monotonic()-start<p['budget']['audit_wall_seconds']
    queue=json.loads((OUT/'queue.json').read_text());assert len(queue)==128 and [j['job_id'] for j in queue]==[e['job_id'] for e in manifest['entries']] and all(j['status']=='pending' for j in queue)
    PUBLIC.mkdir(parents=True,exist_ok=False);np.savez_compressed(PUBLIC/'graphs.npz',orders=orders,truths=np.stack(truths),groups=np.repeat(np.arange(64),2),conditions=np.tile(CONDITIONS,64))
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'inputs.json'),input_sha256=evidence,graphs_sha256=sha(PUBLIC/'graphs.npz'),
        rows=rows,synthetic_observation_instances=307200,total_generated_candidates=total_candidates,within_bank_duplicate_instances=all_duplicates,duplicates_retained=True,new_official_source_rows=0,new_solver_inputs=0,
        scope=p['scope'],timing=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu))
    save(OUT/'source-audit.json',result);save(PUBLIC/'source-audit.json',result);save(PUBLIC/'generation-manifest.json',manifest);save(PUBLIC/'graph-order-seal.json',seal)
    print(json.dumps(dict(status='validated',banks=128,duplicates_retained=all_duplicates,generated=total_candidates,timing=result['timing'])))


def outcomes():
    started=time.monotonic();cpu_started=time.process_time()
    import torch
    torch.set_num_threads(1)
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['auditor_sha256']==sha(__file__)
    assert source['input_manifest_sha256']==sha(OUT/'inputs.json')
    for path,digest in source['input_sha256'].items():assert sha(ROOT/path)==digest
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==128 and all(job['status']=='complete' for job in queue),'Incomplete fits remain retained; no filtered outcome summary.'
    assert {job['job_id'] for job in queue}=={e['job_id'] for e in manifest['entries']}
    for gpu in range(4):
        gate=json.loads((OUT/f'gate-gpu{gpu}.json').read_text())
        assert gate['status']=='passed' and gate['gpu']==gpu and gate['protocol_sha256']==sha(PROTOCOL)
        assert [case['condition'] for case in gate['cases']]==CONDITIONS
        for case in gate['cases']:
            e=next(e for e in manifest['entries'] if e['group']==gpu and e['condition']==case['condition'])
            assert case['input_sha256']==e['sha256'] and len(case['canonical'])==3
            for test in case['canonical']:assert 0<=test['loss_error']<1e-12 and 0<=test['gradient_error']<1e-12
            raw=case['raw'];assert raw['rows']==17 and 0<=raw['loss_error']<2e-6
            assert 0<=raw['w_gradient_error']<2e-7 and 0<=raw['b_gradient_error']<2e-7
    ii,jj=np.triu_indices(81,1);graphs=load_npz(PUBLIC/'graphs.npz');truths=graphs['truths'];assert truths.shape==(128,3240)
    selected=load_npz(OUT/'regularity'/'supports.npz');selection_receipt=json.loads((OUT/'regularity'/'receipt.json').read_text())
    assert selected['supports'].shape==(128,3240) and selected['supports'].dtype==bool and np.array_equal(selected['groups'],np.repeat(np.arange(64),2)) and np.array_equal(selected['conditions'],np.tile(CONDITIONS,64))
    assert selection_receipt['status']=='complete' and selection_receipt['protocol_sha256']==sha(PROTOCOL)
    assert selection_receipt['supports_sha256']==sha(OUT/'regularity'/'supports.npz')
    assert selection_receipt['selector_sha256']==sha(HERE/'regularity_gap.py')
    assert 0<selection_receipt['wall_seconds']<p['budget']['selection_wall_seconds'] and len(selection_receipt['rows'])==128
    selections=[]
    rows=[];fits=[];diagnostics=[];evidence=dict(source['input_sha256']);public_arrays={name:[] for name in
        ['supports','adamw_coefficients','canonical_coefficients','canonical_gradient','counts','free_counts','corruption_counts']}
    for index,e in enumerate(manifest['entries']):
        truth=truths[index]
        hot=np.zeros(81,bool);hot[e['hot_cells']]=True
        assert e['hot_cells']==manifest['hot_cells_by_input'][index]
        categories={'hot_hot':hot[ii]&hot[jj],'hot_cold':hot[ii]^hot[jj],'cold_cold':~hot[ii]&~hot[jj]}
        assert sum(int(value.sum()) for value in categories.values())==3240
        assert time.monotonic()-started<p['budget']['audit_wall_seconds']
        job=next(job for job in queue if job['job_id']==e['job_id']);d=OUT/'fits'/e['job_id']
        data=load_npz(ROOT/e['path']);models=load_npz(d/'models.npz');receipt=json.loads((d/'receipt.json').read_text())
        assert receipt['group']==job['bank']==e['group'] and receipt['condition']==job['condition']==e['condition']
        assert receipt['gpu']==job['gpu'] and 0<=receipt['gpu']<4
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['input_sha256']==e['sha256']
        assert 0<receipt['wall_seconds']<p['budget']['max_group_wall_seconds']
        assert set(receipt['output_sha256'])=={'adamw.pt','models.npz','canonical-fit.json'}
        for name,digest in receipt['output_sha256'].items():assert sha(d/name)==digest;evidence[str((d/name).relative_to(ROOT))]=digest
        evidence[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        assert set(models)=={'adamw_coefficients','adamw_support','canonical_coefficients','canonical_support','counts','restricted_support'}
        for name,array in models.items():
            assert array.shape==(3240,)
            if name.endswith('support'):assert array.dtype==bool
            elif name=='counts':assert array.dtype==np.int64
            else:assert array.dtype==np.float64 and np.isfinite(array).all()
        checkpoint=torch.load(d/'adamw.pt',map_location='cpu',weights_only=True)
        assert set(checkpoint)=={'W','B'} and checkpoint['W'].shape==(3240,9,9) and checkpoint['B'].shape==(81,9)
        for value in checkpoint.values():assert value.dtype==torch.float32 and torch.isfinite(value).all()
        w=checkpoint['W'].numpy().astype(np.float64);adamw,coeff,meta=adamw_support(w)
        assert np.max(np.abs(coeff-models['adamw_coefficients']))<1e-14 and np.array_equal(adamw,models['adamw_support'])
        actual=receipt['adamw'];assert actual['eligible']==meta['eligible']
        for name,value in meta.items():
            if isinstance(value,float):assert abs(actual[name]-value)<1e-12
            else:assert actual[name]==value
        assert actual['augmentation_seed']==p['augmentation_rng_base']+e['group']
        assert len(actual['history'])==32 and all(np.isfinite(x) and x>=0 for x in actual['history'])
        assert abs(actual['history'][0]-float(np.log(9)))<1e-6 and 0<actual['loop_seconds']<=receipt['wall_seconds']
        fit=json.loads((d/'canonical-fit.json').read_text());j=models['canonical_coefficients']
        gradient,fit_audit=audit_canonical_fit(j,fit,data['noisy'],data['free'])
        assert fit['seconds']<=receipt['wall_seconds'] and fit['seconds']+actual['loop_seconds']<=receipt['wall_seconds']+1e-6
        canonical,cut=canonical_support(j);assert np.array_equal(canonical,models['canonical_support'])
        assert receipt['canonical_eligible']==(cut is not None) and receipt['canonical_threshold']==cut
        restricted,counts,count_cut=count_support(data['noisy'])
        assert np.array_equal(counts,models['counts']) and np.array_equal(restricted,models['restricted_support'])
        assert count_cut==receipt['restricted_threshold']
        regularity,selection=independent_selection(counts)
        selection_path=OUT/'regularity'/(e['job_id']+'.json');actual_selection=json.loads(selection_path.read_text())
        row=selection_receipt['rows'][index]
        assert row['group']==e['group'] and row['job_id']==e['job_id'] and row['record_sha256']==sha(selection_path)
        assert actual_selection['job_id']==e['job_id'] and actual_selection['group']==e['group']
        assert actual_selection['protocol_sha256']==sha(PROTOCOL) and actual_selection['models_sha256']==sha(d/'models.npz')
        for key,value in selection.items():assert actual_selection[key]==value,(e['job_id'],key)
        assert np.array_equal(regularity,selected['supports'][index])
        evidence[str(selection_path.relative_to(ROOT))]=sha(selection_path)
        selections.append(actual_selection)
        supports=np.stack([adamw,canonical,restricted,regularity]);changed=data['noisy']!=data['solutions']
        free_counts=data['free'].sum(0,dtype=np.int64);corruption_counts=changed.sum(0,dtype=np.int64)
        assert int(corruption_counts.sum())==21600 and np.all(corruption_counts<=free_counts)
        for name,array in [('supports',supports),('adamw_coefficients',models['adamw_coefficients']),('canonical_coefficients',j),
                           ('canonical_gradient',gradient),('counts',counts),('free_counts',free_counts),('corruption_counts',corruption_counts)]:public_arrays[name].append(array)
        fits.append(dict(job_id=e['job_id'],group=e['group'],condition=e['condition'],banks=e['banks'],
                         canonical_threshold=cut,restricted_threshold=count_cut,adamw_metadata=actual,
                         wall_seconds=receipt['wall_seconds'],receipt_sha256=sha(d/'receipt.json'),**fit_audit))
        true_max=int(counts[truth].max());nonpeer_min=int(counts[~truth].min())
        diagnostics.append(dict(job_id=e['job_id'],group=e['group'],condition=e['condition'],
                                true_count_min=int(counts[truth].min()),true_count_max=true_max,
                                nonpeer_count_min=nonpeer_min,nonpeer_count_max=int(counts[~truth].max()),
                                any_low_count_scalar_threshold_separates=true_max<nonpeer_min,
                                separation_margin=nonpeer_min-true_max,
                                regularity_selected_cut=selection['selected']['cut'],
                                regularity_selected_dispersion=selection['selected']['dispersion'],
                                regularity_exact=bool(np.array_equal(regularity,truth)),
                                no_separating_threshold=true_max>=nonpeer_min,
                                wrong_regularity_choice_despite_separability=bool(true_max<nonpeer_min and not np.array_equal(regularity,truth)),
                                separating_threshold_in_candidate_set=bool(any(true_max<=v['cut']<nonpeer_min for v in selection['candidates'])),
                                hot_free_exposures=int(free_counts[hot].sum()),hot_corruptions=int(corruption_counts[hot].sum()),
                                cold_free_exposures=int(free_counts[~hot].sum()),cold_corruptions=int(corruption_counts[~hot].sum())))
        eligibility=[bool(meta['eligible']),cut is not None,True,True]
        for arm,support,eligible in zip(ARMS,supports,eligibility,strict=True):
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            stratified={name:dict(missing_true=int((mask&truth&~support).sum()),extra=int((mask&~truth&support).sum()),
                                   true_pairs=int((mask&truth).sum()),nonpeer_pairs=int((mask&~truth).sum())) for name,mask in categories.items()}
            assert sum(r['missing_true'] for r in stratified.values())==missing and sum(r['extra'] for r in stratified.values())==extra
            rows.append(dict(group=e['group'],condition=e['condition'],arm=arm,job_id=e['job_id'],banks=e['banks'],
                             eligible=eligible,retained=int(support.sum()),missing_true=missing,extra=extra,all_true_retained=missing==0,
                             exact_graph=missing==extra==0,support_sha256=bytes_sha(support),hot_categories=stratified))
    grouped=[]
    for condition in CONDITIONS:
        for arm in ARMS:
            part=[row for row in rows if row['condition']==condition and row['arm']==arm];assert len(part)==64
            grouped.append(dict(condition=condition,arm=arm,exact_graphs=sum(row['exact_graph'] for row in part),
                                eligible_groups=sum(row['eligible'] for row in part),ineligible_groups=sum(not row['eligible'] for row in part),
                                all_true_retaining_groups=sum(row['all_true_retained'] for row in part),
                                missing_true_total=sum(row['missing_true'] for row in part),extra_total=sum(row['extra'] for row in part)))
    primary=dict(status='descriptive_controls_only',note='Per-family graph/degree endpoints audited separately in LP summary.')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PUBLIC/'models.npz',**{name:np.stack(values) for name,values in public_arrays.items()},
                        groups=np.array([e['group'] for e in manifest['entries']],np.int64),conditions=np.array([e['condition'] for e in manifest['entries']]),
                        arms=np.array(ARMS),hot_cells_by_input=np.array(manifest['hot_cells_by_input'],np.int64))
    launch={}
    for phase in ['gate','fit']:
        record=json.loads((OUT/f'{phase}-launch.json').read_text());assert record['status']=='complete' and record['returncodes']==[0,0,0,0]
        assert record['failure'] is None and 0<record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds']
        assert np.isfinite(record['aggregate_cpu_seconds']) and record['aggregate_cpu_seconds']>=0;launch[phase]=record
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),scope=p['scope'],primary=primary,
                grouped=grouped,rows=rows,fit_audits=fits,diagnostics=diagnostics,source_audit=source,input_sha256=evidence,
                models_sha256=sha(PUBLIC/'models.npz'),graphs_sha256=sha(PUBLIC/'graphs.npz'),launch=launch,selection_receipt=selection_receipt,selection_records=selections,
                all_groups_retained=True,synthetic_banks=128,per_family_banks=64,groups=64,conditions=2,logical_graphs=512,synthetic_observation_instances=307200,
                new_official_source_rows=0,new_solver_inputs=0,inference_gpu_seconds=0,optimizer_scope='Finite BP32 and canonical256 recipes; descriptive controls, no matched-compute or optimum claim.')
    elapsed=time.monotonic()-started;assert elapsed<p['budget']['audit_wall_seconds']
    result['outcome_audit_timing']=dict(wall_seconds=elapsed,process_cpu_seconds=time.process_time()-cpu_started,wall_ceiling_seconds=p['budget']['audit_wall_seconds'])
    save(OUT/'outcome-audit-timing.json',result['outcome_audit_timing'])
    save(PUBLIC/'selection-records.json',selections);result['selection_records_sha256']=sha(PUBLIC/'selection-records.json')
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(status='validated',primary=primary,grouped=grouped)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','outcomes']);globals()[ap.parse_args().action]()
