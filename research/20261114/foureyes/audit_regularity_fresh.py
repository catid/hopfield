"""Independent H4af source identities, raw labels/corruptions and64bank outcome audit."""
import argparse,csv,hashlib,json,os,time
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[_key]='1'
import numpy as np
from regularity_fresh_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,PARENT,DESIGN,CONDITION,ARMS,sha,save,verify
from audit_heterogeneous_noise import (load_npz,bytes_sha,independent_corruption,canonical_support,
                                     count_support,adamw_support,audit_canonical_fit,numpy_raw_loss_gradient)
from audit_regularity_gap import independent_selection
CONDITIONS=[CONDITION]


def sources():
    started=time.monotonic();cpu_started=time.process_time();p=json.loads(PROTOCOL.read_text());verify(p)
    assert p['groups']==[[g] for g in range(64)] and p['group_count']==64 and p['rows_per_group']==2400
    assert p['condition']==CONDITION and p['arms']==ARMS
    assert p['source_ledger']==str((DESIGN/'ledger.json').relative_to(ROOT)) and p['source_ledger_sha256']==sha(DESIGN/'ledger.json')
    ledger=json.loads((DESIGN/'ledger.json').read_text());parent=json.loads((HERE/'canonical_confirm.json').read_text());evidence={}
    assert ledger['status']=='frozen_before_draws' and ledger['source']==parent['source'] and ledger['source_sha256']==parent['source_sha256_value']
    source=ROOT/ledger['source'];assert sha(source)==ledger['source_sha256'];evidence[ledger['source']]=ledger['source_sha256']
    for path,digest in ledger['exclusion_input_sha256'].items():assert sha(ROOT/path)==digest;evidence[path]=digest
    pieces=[];counts=[]
    for spec in parent['exclusions']:
        assert ledger['exclusion_input_sha256'][spec['path']]==spec['sha256']
        ids=[]
        for entry in json.loads((ROOT/spec['path']).read_text())['inputs']:
            assert ledger['exclusion_input_sha256'][entry['completion_path']]==entry['completion_sha256']
            with np.load(ROOT/entry['completion_path'],allow_pickle=False) as z:ids.append(z['source_row'].copy())
        ids=np.unique(np.concatenate(ids));pieces.append(ids);counts.append(dict(manifest=spec['path'],unique_ids=len(ids)))
    assert counts==ledger['prior_manifest_counts'] and len(counts)==4
    old=json.loads((PARENT/'inputs.json').read_text());assert len(old['banks'])==64 and [e['bank'] for e in old['banks']]==list(range(64))
    h4y=[]
    for e in old['banks']:
        assert ledger['exclusion_input_sha256'][e['path']]==e['sha256']
        with np.load(ROOT/e['path'],allow_pickle=False) as z:h4y.append(z['source_row'].copy())
    h4y=np.unique(np.concatenate(h4y));excluded=np.unique(np.concatenate(pieces+[h4y]))
    assert len(h4y)==ledger['h4y_unique_rows'] and len(excluded)==ledger['excluded_rows']
    assert excluded.dtype==np.int64 and ((excluded>=0)&(excluded<ledger['total_source_rows'])).all()
    eligible=np.setdiff1d(np.arange(ledger['total_source_rows'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==ledger['eligible_rows'] and len(eligible)>64*2400
    for name,actual in [('eligible.npy',eligible),('excluded.npy',excluded),('h4y_ids.npy',h4y)]:
        assert sha(DESIGN/name)==ledger['array_sha256'][name] and np.array_equal(np.load(DESIGN/name,allow_pickle=False),actual)
        evidence[str((DESIGN/name).relative_to(ROOT))]=ledger['array_sha256'][name]
    cache=ROOT/ledger['cache'];cm=json.loads((cache/'manifest.json').read_text())
    assert cm['source_sha256']==ledger['source_sha256'] and cm['files']==ledger['cache_sha256']
    for name,digest in ledger['cache_sha256'].items():assert sha(cache/name)==digest;evidence[str((cache/name).relative_to(ROOT))]=digest
    clean=np.load(cache/'solutions.npy',mmap_mode='r');given=np.load(cache/'given.npy',mmap_mode='r')
    assert clean.shape==given.shape==(ledger['total_source_rows'],81) and clean.dtype==np.uint8 and given.dtype==bool
    manifest=json.loads((OUT/'inputs.json').read_text());draw_receipt=json.loads((OUT/'source-draws.json').read_text());draws=load_npz(OUT/'source-draws.npz')
    assert set(draws)=={'source_rows'};draws=draws['source_rows'];assert draws.shape==(64,2400) and draws.dtype==np.int64
    assert manifest['protocol_sha256']==draw_receipt['protocol_sha256']==sha(PROTOCOL)
    assert draw_receipt['source_ledger_sha256']==p['source_ledger_sha256'] and draw_receipt['sealed_before_materializing_labels'] is True
    assert draw_receipt['array_sha256']==manifest['source_draws_sha256']==sha(OUT/'source-draws.npz')
    assert manifest['source_draws_receipt_sha256']==sha(OUT/'source-draws.json')
    assert draw_receipt['bank_rng_base']==p['bank_rng_base'] and draw_receipt['rows_per_bank']==2400 and draw_receipt['banks']==64
    unique=np.unique(draws);assert len(unique)==draw_receipt['unique_source_rows'] and draws.size-len(unique)==draw_receipt['crossbank_repeated_instances']
    assert np.intersect1d(unique,excluded).size==0
    assert len(manifest['entries'])==64 and len(manifest['hot_cells_by_group'])==64
    expected_ids=[f'g{g:02d}-{CONDITION}' for g in range(64)];assert [e['job_id'] for e in manifest['entries']]==expected_ids
    groups=[]
    for group,e in enumerate(manifest['entries']):
        assert time.monotonic()-started<p['budget']['audit_wall_seconds']
        ids=np.random.default_rng(p['bank_rng_base']+group).choice(eligible,2400,replace=False)
        assert np.array_equal(ids,draws[group]) and len(np.unique(ids))==2400
        hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+group).choice(81,18,replace=False))
        assert e['group']==group and e['banks']==[group] and e['condition']==CONDITION
        assert e['hot_cells']==manifest['hot_cells_by_group'][group]==hot.tolist() and len(set(hot))==18
        path=ROOT/e['path'];assert sha(path)==e['sha256'];evidence[e['path']]=e['sha256'];z=load_npz(path)
        rng=np.random.default_rng(p['noise_rng_base']+group)
        expected=dict(solutions=np.asarray(clean[ids]),free=~given[ids],source_row=ids,source_bank=np.full(2400,group,np.int64),
                      uniforms=1-rng.random((2400,81)),offsets=rng.integers(1,9,(2400,81),dtype=np.int64))
        assert set(z)==set(expected)|{'noisy','cells'}
        for name,array in expected.items():assert z[name].dtype==array.dtype and np.array_equal(z[name],array),(e['job_id'],name)
        assert ((z['solutions']>=0)&(z['solutions']<9)).all() and (z['free'].sum(1)>=9).all()
        boards=z['solutions'].reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):assert ((boards==digit).sum(1)==1).all() and ((boards==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()
        noisy,cells=independent_corruption(z['solutions'],z['free'],z['uniforms'],z['offsets'],hot,CONDITION)
        assert z['noisy'].dtype==np.uint8 and z['cells'].dtype==np.int64 and np.array_equal(noisy,z['noisy']) and np.array_equal(cells,z['cells'])
        changed=z['noisy']!=z['solutions'];assert (changed.sum(1)==9).all() and not changed[~z['free']].any()
        assert np.take_along_axis(z['free'],z['cells'],1).all() and all(len(set(row))==9 for row in z['cells'])
        groups.append(dict(group=group,banks=[group],condition=CONDITION,input_sha256=e['sha256'],hot_cells=hot.tolist(),rows=2400,unique_source_rows=2400,
                           repeated_source_row_instances=0,changed_free_cells=21600))
    # Independently parse every newly selected answer and clue mask from original CSV.
    remaining=set(map(int,unique));verified=0
    with source.open(newline='') as f:
        reader=csv.reader(f);header=next(reader);answer_col=header.index('answer');question_col=header.index('question')
        for index,row in enumerate(reader):
            if index%500000==0:assert time.monotonic()-started<p['budget']['audit_wall_seconds']
            if index not in remaining:continue
            board=np.array([int(c)-1 for c in row[answer_col]],np.uint8);mask=np.array([c in '123456789' for c in row[question_col]],bool)
            assert board.shape==mask.shape==(81,) and np.array_equal(board,clean[index]) and np.array_equal(mask,given[index])
            assert all(int(c)-1==int(board[cell]) for cell,c in enumerate(row[question_col]) if mask[cell])
            remaining.remove(index);verified+=1
            if not remaining:break
    assert not remaining and verified==len(unique)
    queue=json.loads((OUT/'queue.json').read_text());assert len(queue)==64 and [j['job_id'] for j in queue]==expected_ids and all(j['status']=='pending' for j in queue)
    for job,e in zip(queue,manifest['entries'],strict=True):assert job['bank']==e['group'] and job['condition']==CONDITION
    elapsed=time.monotonic()-started;assert elapsed<p['budget']['audit_wall_seconds']
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'inputs.json'),
                source_ledger_sha256=p['source_ledger_sha256'],source_draws_sha256=sha(OUT/'source-draws.npz'),source_draws_receipt_sha256=sha(OUT/'source-draws.json'),
                source_sha256=ledger['source_sha256'],hot_cells_by_group=manifest['hot_cells_by_group'],groups=groups,input_sha256=evidence,
                prior_excluded_rows=len(excluded),h4y_excluded_rows=len(h4y),eligible_rows=len(eligible),new_unique_source_rows=len(unique),
                independent_csv_answers_and_clue_masks_verified=verified,new_source_draw_instances=153600,crossbank_repeated_instances=draws.size-len(unique),
                within_bank_repeated_instances=0,declared_exclusion_overlap=0,h4y_source_overlap=0,clean_observation_instances=153600,condition_observation_instances=153600,
                source_banks=64,new_solver_inputs=0,all_repeated_instances_retained=True,scope=p['scope'],
                timing=dict(wall_seconds=elapsed,process_cpu_seconds=time.process_time()-cpu_started,wall_ceiling_seconds=p['budget']['audit_wall_seconds']))
    save(OUT/'source-audit.json',result);print(json.dumps(dict(status='validated',groups=64,entries=64,new_unique_source_rows=len(unique),timing=result['timing'])))

def outcomes():
    started=time.monotonic();cpu_started=time.process_time()
    import torch
    torch.set_num_threads(1)
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['auditor_sha256']==sha(__file__)
    assert source['input_manifest_sha256']==sha(OUT/'inputs.json')
    for path,digest in source['input_sha256'].items():assert sha(ROOT/path)==digest
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==64 and all(job['status']=='complete' for job in queue),'Incomplete fits remain retained; no filtered outcome summary.'
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
    assert selected['supports'].shape==(64,3240) and selected['supports'].dtype==bool and np.array_equal(selected['groups'],np.arange(64))
    assert selection_receipt['status']=='complete' and selection_receipt['protocol_sha256']==sha(PROTOCOL)
    assert selection_receipt['supports_sha256']==sha(OUT/'regularity'/'supports.npz')
    assert selection_receipt['selector_sha256']==sha(HERE/'regularity_gap.py')
    assert 0<selection_receipt['wall_seconds']<p['budget']['selection_wall_seconds'] and len(selection_receipt['rows'])==64
    selections=[]
    rows=[];fits=[];diagnostics=[];evidence=dict(source['input_sha256']);public_arrays={name:[] for name in
        ['supports','adamw_coefficients','canonical_coefficients','canonical_gradient','counts','free_counts','corruption_counts']}
    for e in manifest['entries']:
        hot=np.zeros(81,bool);hot[e['hot_cells']]=True
        assert e['hot_cells']==manifest['hot_cells_by_group'][e['group']]
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
            part=[row for row in rows if row['condition']==condition and row['arm']==arm];assert len(part)==64
            grouped.append(dict(condition=condition,arm=arm,exact_graphs=sum(row['exact_graph'] for row in part),
                                eligible_groups=sum(row['eligible'] for row in part),ineligible_groups=sum(not row['eligible'] for row in part),
                                all_true_retaining_groups=sum(row['all_true_retained'] for row in part),
                                missing_true_total=sum(row['missing_true'] for row in part),extra_total=sum(row['extra'] for row in part)))
    target=next(g for g in grouped if g['arm']=='regularity')
    success=target['exact_graphs']>=61 and target['missing_true_total']==0
    primary=dict(decision='supported' if success else 'falsified',exact_graphs=target['exact_graphs'],missing_true_total=target['missing_true_total'],
                 attribution='New2400distinct-row banks and heatmaps, disjoint from H4y and four declared preceding manifests; finite confirmation, not a population confidence bound or global novelty claim.')
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
                all_groups_retained=True,source_banks=64,groups=64,conditions=1,logical_graphs=256,clean_observation_instances=153600,condition_observation_instances=153600,
                new_source_draw_instances=153600,new_unique_source_rows=source['new_unique_source_rows'],new_solver_inputs=0,inference_gpu_seconds=0,optimizer_scope='Finite BP32 and canonical256 recipes; descriptive controls, no matched-compute or optimum claim.')
    elapsed=time.monotonic()-started;assert elapsed<p['budget']['audit_wall_seconds']
    result['outcome_audit_timing']=dict(wall_seconds=elapsed,process_cpu_seconds=time.process_time()-cpu_started,wall_ceiling_seconds=p['budget']['audit_wall_seconds'])
    save(OUT/'outcome-audit-timing.json',result['outcome_audit_timing'])
    save(PUBLIC/'selection-records.json',selections);result['selection_records_sha256']=sha(PUBLIC/'selection-records.json')
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(status='validated',primary=primary,grouped=grouped)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','outcomes']);globals()[ap.parse_args().action]()
