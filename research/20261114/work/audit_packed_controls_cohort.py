"""Independent H2bt full-identity, source-state and charged-host cohort audit."""
import json
import math
from pathlib import Path
import numpy as np
from packed_controls_source_parity import Source,sha
from audit_bp_first_fresh_states import numeric_error


def segment(record,before,after):
    assert set(record)=={'started_monotonic','finished_monotonic','seconds'}
    start,end,seconds=record['started_monotonic'],record['finished_monotonic'],record['seconds']
    assert all(math.isfinite(v) for v in [start,end,seconds]) and seconds>=0
    assert abs(end-start-seconds)<1e-8 and start>=max(before) and end<=after[0]
    return seconds


def verify(root,folder,p,ph,expected_indices,worker_directory,command):
    root,folder=Path(root),Path(folder);r=json.loads((folder/'result.json').read_text());source=Source(root,p['source'])
    assert r['status']=='complete' and r['protocol_sha256']==ph and r['mode'] in ['LP96','BP_geometry_LP96','packed_dispatch']
    assert r['all_writes_closed_hashed_and_joined'] and r['archive_tasks_during_answer_segments']==0
    assert r['archive_writer_metrics']['maximum_pending']<=2 and r['archive_writer_metrics']['maximum_rss_kib']<=8*1024**2
    indices=np.asarray(expected_indices,np.int64);assert np.array_equal(indices,np.unique(indices)) and len(indices)>0
    assert r['model_inputs']==3*len(indices)
    tokens={}
    for shard in np.unique(indices//256):
        ds=p['source']['datasets'][int(shard)];assert sha(root/ds['path'])==ds['sha256']
        with np.load(root/ds['path'],allow_pickle=False) as z:tokens[int(shard)]=z['tokens']
    slots=(len(indices)+31)//32+1;expected_stages=['readiness-begin','readiness-done']
    for seed in range(3):
        expected_stages += [f'seed{seed}-preparation-begin',f'seed{seed}-preparation-done']
        for slot in range(slots):expected_stages += [f'seed{seed}-slot{slot:03d}-'+suffix for suffix in ['materialize-begin','materialize-done','answer-done','scatter-begin','scatter-done']]
        expected_stages += [f'seed{seed}-finalization-begin',f'seed{seed}-finalization-done']
    expected_stages+=['archives-done'];barrier=r['host_barrier'];events=barrier['events'];assert [e['stage'] for e in events]==expected_stages
    times={}
    for e in events:
        times[e['stage']]=[]
        for side,gpu in [('own',barrier['gpu']),('peer',1-barrier['gpu'])]:
            file=root/barrier['directory']/e[side+'_marker'];assert sha(file)==e[side+'_sha256'];marker=json.loads(file.read_text())
            worker=json.loads((Path(worker_directory)/f'{command}-worker{gpu}.json').read_text())
            assert marker['stage']==e['stage'] and marker['gpu']==gpu and marker['pid']==worker['pid'] and marker['protocol_sha256']==ph
            times[e['stage']].append(marker['arrived_monotonic'])
        assert 0<=e['wait_seconds']<=60.01
    assert abs(sum(e['wait_seconds'] for e in events)-barrier['wait_seconds'])<1e-8
    total=segment(r['readiness'],times['readiness-begin'],times['readiness-done'])
    ready=r['model_readiness'];required_models=['lp'] if r['mode']=='LP96' else ['bp','lp']
    assert len(ready)==3*len(required_models) and {(x['seed'],x['model']) for x in ready}=={(seed,m) for seed in range(3) for m in required_models}
    for record in ready:assert record['checkpoint_sha256']==p['models'][str(record['seed'])]['old_binary' if record['model']=='lp' else 'pl']['sha256']
    assert all(x['seconds']>0 and math.isfinite(x['seconds']) for x in ready) and r['readiness_orchestration_seconds']>=0
    assert abs(sum(x['seconds'] for x in ready)+r['readiness_orchestration_seconds']-total)<1e-8
    assert len(r['models'])==3 and len(r['rows'])==3*slots;copies=candidates=states=zero_slots=0
    for seed,model in enumerate(r['models']):
        assert model['seed']==seed and model['indices']==indices.tolist() and model['actual_inputs']==len(indices) and model['slots_per_model']==slots
        for part in ['preparation','finalization']:total+=segment(model[part],times[f'seed{seed}-{part}-begin'],times[f'seed{seed}-{part}-done'])
        distance=np.empty(len(indices),np.int64)
        for j,index in enumerate(indices):
            t=tokens[int(index)//256][int(index)%256];starts=np.argwhere(t==2);goals=np.argwhere(t==3);assert starts.shape==goals.shape==(1,2)
            distance[j]=int(np.abs(starts[0]-goals[0]).sum())
        high=distance>=182;groups=[('original',indices)] if r['mode']!='packed_dispatch' else [('high',indices[high]),('low',indices[~high])]
        batches=[(group,values[start:start+32]) for group,values in groups for start in range(0,len(values),32)]
        assert model['zero_work_slots']==slots-len(batches)
        for key in ['queue_file','global_answer_file']:assert sha(folder/model[key])==model[key+'_sha256']
        with np.load(folder/model['queue_file'],allow_pickle=False) as z:
            assert np.array_equal(z['global_indices'],indices)
            if r['mode']=='packed_dispatch':assert set(z.files)=={'global_indices','distances','high'} and np.array_equal(z['distances'],distance) and np.array_equal(z['high'],high)
            else:assert z.files==['global_indices']
        with np.load(folder/model['global_answer_file'],allow_pickle=False) as z:global_arrays={name:z[name] for name in z.files}
        source.compare_global(seed,indices,'endpoint_dispatch' if r['mode']=='packed_dispatch' else r['mode'],global_arrays)
        del global_arrays
        selected=r['rows'][seed*slots:(seed+1)*slots]
        for slot,row in enumerate(selected):
            assert row['seed']==seed and row['slot']==slot and row['archive_tasks_at_answer_start']==0 and row['archive_drain_seconds']>=0
            label=f'seed{seed}-slot{slot:03d}';zero=slot>=len(batches);assert row['zero_work']==zero
            if zero:
                assert row['indices']==[] and row['group'] is None and row['file'] is None and row['trace_files']==[]
                assert row['answer_timing'] is row['materialization'] is row['scatter'] is None;zero_slots+=1;continue
            group,ids=batches[slot];assert row['group']==group and row['indices']==ids.tolist()
            total+=segment(row['materialization'],times[label+'-materialize-begin'],times[label+'-materialize-done'])
            total+=segment(row['scatter'],times[label+'-scatter-begin'],times[label+'-scatter-done'])
            t=row['answer_timing'];c=t['components'];assert t['n']==len(ids) and 1<=len(ids)<=32 and t['mode']==('endpoint_dispatch' if r['mode']=='packed_dispatch' else r['mode'])
            required={'input_transfer','hidden_readout_and_flags','LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return','final_numerical_acceptance'}
            required|={'model_provenance'} if r['mode']=='LP96' else {'BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return','geometry_screen_and_indices','LP_subset_gather','subset_scatter_and_provenance'}
            if r['mode']=='packed_dispatch':required|={'endpoint_extraction_and_dispatch','BP_subset_gather'}
            assert required<=c.keys() and all(math.isfinite(v) and v>=0 for v in c.values()) and t['unattributed_in_segment_seconds']>=0
            assert abs(sum(c.values())+t['unattributed_in_segment_seconds']-t['answer_seconds'])<1e-8
            assert t['answer_started_monotonic']>=max(times[label+'-materialize-done']) and t['answer_finished_monotonic']<=times[label+'-answer-done'][0]
            assert abs(t['answer_finished_monotonic']-t['answer_started_monotonic']-t['answer_seconds'])<1e-8
            assert t['diagnostic_started_monotonic']>=max(times[label+'-answer-done']) and t['diagnostic_started_monotonic']<=times[label+'-scatter-begin'][0]
            if r['mode']=='packed_dispatch':
                assert t['actual_BP_inputs']+t['direct_LP_inputs']==len(ids) and t['actual_subset_inputs']==t['direct_LP_inputs']+t['rescue_LP_inputs']
                for key in ['BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return']:assert (c[key]==0)==(t['actual_BP_inputs']==0)
            if r['mode']!='LP96':
                for key in ['LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return']:assert (c[key]==0)==(t['actual_subset_inputs']==0)
            else:assert t['actual_subset_inputs']==0 and not any(key.startswith('BP_') for key in c)
            total+=t['answer_seconds'];host_tokens=np.stack([tokens[int(i)//256][int(i)%256] for i in ids])
            checked=source.check_batch(folder,row,ph,host_tokens);copies+=checked['model_inputs'];candidates+=checked['candidate_score_comparisons'];states+=checked['active_trace_states']
            with np.load(folder/row['file'],allow_pickle=False) as z:assert np.array_equal(z['original_order_positions'],np.searchsorted(indices,ids))
    assert copies==3*len(indices) and abs(total-r['complete_answer_seconds'])<1e-8
    return {'status':'passed','mode':r['mode'],'block':r['block'],'model_pipeline_copies':copies,'candidate_score_comparisons':candidates,'active_trace_states':states,'zero_work_slots':zero_slots,'barrier_events':len(events),'complete_answer_seconds':r['complete_answer_seconds'],'independently_summed_seconds':total,'cohort_receipt_sha256':sha(folder/'result.json')}
