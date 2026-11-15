"""Independent H4ae sources, all threshold candidates and unchanged fit audits."""
import argparse,hashlib,json,os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[_key]='1'
import numpy as np
from regularity_gap_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,PARENT,CONDITION,ARMS,sha,save,verify
from audit_heterogeneous_noise import (load_npz,bytes_sha,independent_corruption,canonical_support,
                                     count_support,adamw_support,audit_canonical_fit,numpy_raw_loss_gradient)
CONDITIONS=[CONDITION]


def independent_selection(counts,n=81):
    assert counts.shape==(n*(n-1)//2,) and counts.dtype.kind in 'iu' and (counts>=0).all()
    ordered=sorted(map(int,counts));size=len(ordered)
    median=ordered[size//2] if size%2 else (ordered[size//2-1]+ordered[size//2])/2
    candidates=[];pairs=list(zip(*np.triu_indices(n,1),strict=True))
    for cut in sorted(set(ordered)):
        if cut>median:continue
        degree=[0]*n;retained=0
        for edge,(left,right) in enumerate(pairs):
            if int(counts[edge])<=cut:degree[left]+=1;degree[right]+=1;retained+=1
        if retained==0:continue
        total=sum(degree);square=sum(d*d for d in degree)
        candidates.append(dict(cut=cut,retained=retained,degree_min=min(degree),degree_max=max(degree),degree_sum=total,
                               degree_square_sum=square,dispersion=n*square-total*total))
    assert candidates
    ordered_candidates=sorted(candidates,key=lambda r:(r['dispersion'],r['retained'],r['cut']))
    best=ordered_candidates[0]
    return counts<=best['cut'],dict(n=n,median_count=float(median),candidates=candidates,selected=dict(best),
                                  minimum_dispersion_ties=sum(r['dispersion']==best['dispersion'] for r in candidates))


def sources():
    p=json.loads(PROTOCOL.read_text());verify(p)
    assert p['groups']==[list(range(g*4,g*4+4)) for g in range(16)] and p['condition']==CONDITION and p['arms']==ARMS
    manifest=json.loads((OUT/'inputs.json').read_text());assert manifest['protocol_sha256']==sha(PROTOCOL)
    assert len(manifest['hot_cells_by_group'])==16
    old=json.loads((PARENT/'inputs.json').read_text());prior=json.loads((PARENT/'source-audit.json').read_text())
    assert prior['status']=='validated' and prior['protocol_sha256']==old['protocol_sha256']==sha(HERE/'canonical_confirm.json')
    assert [e['bank'] for e in old['banks']]==list(range(64)) and len(manifest['entries'])==16
    expected_ids=[f'g{g:02d}-{CONDITION}' for g in range(16)]
    assert [e['job_id'] for e in manifest['entries']]==expected_ids
    evidence={};groups=[]
    for group,banks in enumerate(p['groups']):
        hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+group).choice(81,18,replace=False))
        assert manifest['hot_cells_by_group'][group]==hot.tolist() and len(set(hot))==18
        original={name:[] for name in ['solutions','free','source_row','source_bank','uniforms','offsets']}
        for bank in banks:
            e=old['banks'][bank];path=ROOT/e['path'];assert sha(path)==e['sha256']
            assert prior['input_sha256'][e['path']]==e['sha256'];evidence[e['path']]=e['sha256']
            z=load_npz(path);assert z['solutions'].shape==z['free'].shape==(600,81)
            assert z['solutions'].dtype==np.uint8 and z['free'].dtype==bool and z['source_row'].dtype==np.int64
            assert ((z['solutions']>=0)&(z['solutions']<9)).all() and (z['free'].sum(1)>=9).all()
            for name in ['solutions','free','source_row']:original[name].append(z[name])
            original['source_bank'].append(np.full(600,bank,np.int64))
            rng=np.random.default_rng(p['noise_rng_base']+bank)
            original['uniforms'].append(1-rng.random((600,81)))
            original['offsets'].append(rng.integers(1,9,(600,81),dtype=np.int64))
        expected={name:np.concatenate(values) for name,values in original.items()}
        e=manifest['entries'][group]
        assert e['group']==group and e['condition']==CONDITION and e['banks']==banks and e['hot_cells']==hot.tolist()
        path=ROOT/e['path'];assert sha(path)==e['sha256'];evidence[e['path']]=e['sha256'];z=load_npz(path)
        assert set(z)==set(expected)|{'noisy','cells'}
        for name,array in expected.items():assert z[name].dtype==array.dtype and np.array_equal(z[name],array),(e['job_id'],name)
        noisy,cells=independent_corruption(z['solutions'],z['free'],z['uniforms'],z['offsets'],hot,CONDITION)
        assert z['noisy'].dtype==np.uint8 and z['cells'].dtype==np.int64
        assert np.array_equal(noisy,z['noisy']) and np.array_equal(cells,z['cells'])
        changed=z['noisy']!=z['solutions'];assert (changed.sum(1)==9).all() and not changed[~z['free']].any()
        assert np.array_equal(np.take_along_axis(z['free'],z['cells'],1),np.ones((2400,9),bool))
        assert all(len(set(row))==9 for row in z['cells'])
        groups.append(dict(group=group,banks=banks,condition=CONDITION,input_sha256=e['sha256'],hot_cells=hot.tolist(),rows=2400,
                           changed_free_cells=21600,unique_source_rows=int(len(np.unique(expected['source_row']))),
                           repeated_source_row_instances=2400-int(len(np.unique(expected['source_row'])))))
    queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==16 and [job['job_id'] for job in queue]==expected_ids and all(job['status']=='pending' for job in queue)
    for job,e in zip(queue,manifest['entries'],strict=True):assert job['bank']==e['group'] and job['condition']==e['condition']
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'inputs.json'),
                parent_source_audit_sha256=sha(PARENT/'source-audit.json'),hot_cells_by_group=manifest['hot_cells_by_group'],groups=groups,
                input_sha256=evidence,clean_observation_instances=38400,condition_observation_instances=38400,
                source_banks=64,new_source_rows=0,new_solver_inputs=0,all_repeated_instances_retained=True)
    save(OUT/'source-audit.json',result);print(json.dumps(dict(status='validated',groups=16,entries=16,new_source_rows=0)))


def outcomes():
    import torch
    torch.set_num_threads(1)
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['auditor_sha256']==sha(__file__)
    assert source['input_manifest_sha256']==sha(OUT/'inputs.json')
    for path,digest in source['input_sha256'].items():assert sha(ROOT/path)==digest
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==16 and all(job['status']=='complete' for job in queue),'Incomplete fits remain retained; no filtered outcome summary.'
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
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    selected=load_npz(OUT/'regularity'/'supports.npz');selection_receipt=json.loads((OUT/'regularity'/'receipt.json').read_text())
    assert selected['supports'].shape==(16,3240) and selected['supports'].dtype==bool and np.array_equal(selected['groups'],np.arange(16))
    assert selection_receipt['status']=='complete' and selection_receipt['protocol_sha256']==sha(PROTOCOL)
    assert selection_receipt['supports_sha256']==sha(OUT/'regularity'/'supports.npz')
    assert selection_receipt['selector_sha256']==sha(HERE/'regularity_gap.py')
    assert 0<selection_receipt['wall_seconds']<p['budget']['selection_wall_seconds'] and len(selection_receipt['rows'])==16
    selections=[]
    rows=[];fits=[];diagnostics=[];evidence=dict(source['input_sha256']);public_arrays={name:[] for name in
        ['supports','adamw_coefficients','canonical_coefficients','canonical_gradient','counts','free_counts','corruption_counts']}
    for e in manifest['entries']:
        hot=np.zeros(81,bool);hot[e['hot_cells']]=True
        assert e['hot_cells']==manifest['hot_cells_by_group'][e['group']]
        categories={'hot_hot':hot[ii]&hot[jj],'hot_cold':hot[ii]^hot[jj],'cold_cold':~hot[ii]&~hot[jj]}
        assert sum(int(value.sum()) for value in categories.values())==3240
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
        row=selection_receipt['rows'][e['group']]
        assert row['group']==e['group'] and row['job_id']==e['job_id'] and row['record_sha256']==sha(selection_path)
        assert actual_selection['job_id']==e['job_id'] and actual_selection['group']==e['group']
        assert actual_selection['protocol_sha256']==sha(PROTOCOL) and actual_selection['models_sha256']==sha(d/'models.npz')
        for key,value in selection.items():assert actual_selection[key]==value,(e['job_id'],key)
        assert np.array_equal(regularity,selected['supports'][e['group']])
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
            part=[row for row in rows if row['condition']==condition and row['arm']==arm];assert len(part)==16
            grouped.append(dict(condition=condition,arm=arm,exact_graphs=sum(row['exact_graph'] for row in part),
                                eligible_groups=sum(row['eligible'] for row in part),ineligible_groups=sum(not row['eligible'] for row in part),
                                all_true_retaining_groups=sum(row['all_true_retained'] for row in part),
                                missing_true_total=sum(row['missing_true'] for row in part),extra_total=sum(row['extra'] for row in part)))
    target=next(g for g in grouped if g['arm']=='regularity')
    success=target['exact_graphs']>=15 and target['missing_true_total']==0
    primary=dict(decision='supported' if success else 'falsified',exact_graphs=target['exact_graphs'],missing_true_total=target['missing_true_total'],
                 attribution='New heat maps and corruption draws on the same fixed clean quartets; finite randomization, not fresh-source confirmation.')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PUBLIC/'models.npz',**{name:np.stack(values) for name,values in public_arrays.items()},
                        groups=np.array([e['group'] for e in manifest['entries']],np.int64),conditions=np.array([e['condition'] for e in manifest['entries']]),
                        arms=np.array(ARMS),hot_cells_by_group=np.array(manifest['hot_cells_by_group'],np.int64))
    launch={}
    for phase in ['gate','fit']:
        record=json.loads((OUT/f'{phase}-launch.json').read_text());assert record['status']=='complete' and record['returncodes']==[0,0,0,0]
        assert record['failure'] is None and 0<record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds']
        assert np.isfinite(record['aggregate_cpu_seconds']) and record['aggregate_cpu_seconds']>=0;launch[phase]=record
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),scope=p['scope'],primary=primary,
                grouped=grouped,rows=rows,fit_audits=fits,diagnostics=diagnostics,source_audit=source,input_sha256=evidence,
                models_sha256=sha(PUBLIC/'models.npz'),launch=launch,selection_receipt=selection_receipt,selection_records=selections,
                all_groups_retained=True,source_banks=64,groups=16,conditions=1,logical_graphs=64,clean_observation_instances=38400,condition_observation_instances=38400,
                new_source_rows=0,new_solver_inputs=0,inference_gpu_seconds=0,optimizer_scope='Finite BP32 and canonical256 recipes; descriptive controls, no matched-compute or optimum claim.')
    save(PUBLIC/'selection-records.json',selections);result['selection_records_sha256']=sha(PUBLIC/'selection-records.json')
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(status='validated',primary=primary,grouped=grouped)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','outcomes']);globals()[ap.parse_args().action]()
