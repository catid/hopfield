"""Public H2bs complete final-board/source preservation and exact full-cost checks."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from verify_training_feasibility import flip,LD
from verify_far_placement_public import close


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment(record,before,after):
    assert record['started_monotonic']>=max(before) and record['finished_monotonic']<=after[0]
    assert record['seconds']>=0 and abs(record['finished_monotonic']-record['started_monotonic']-record['seconds'])<1e-8
    return record['seconds']


def costs(directory,ph,distances):
    markers=json.loads((directory/'barrier-markers.json').read_text())['markers'];assert len(markers)==23940
    totals={};partial=set();events_checked=0
    for pair in range(6):
        for mode in ['original_order','packed']:
            folder=directory/'runs'/f'pair{pair}-{mode}';r=json.loads((folder/'result.json').read_text());assert r['status']=='complete' and r['protocol_sha256']==ph and r['pair']==pair and r['mode']==mode
            assert r['model_inputs']==12288 and len(r['rows'])==387 and len(r['models'])==3
            assert r['all_writes_closed_hashed_and_joined'] and r['archive_tasks_during_answer_segments']==0
            expected=['readiness-begin','readiness-done']
            for seed in range(3):
                expected += [f'seed{seed}-preparation-begin',f'seed{seed}-preparation-done']
                for slot in range(129):expected += [f'seed{seed}-slot{slot:03d}-'+suffix for suffix in ['materialize-begin','materialize-done','answer-done','scatter-begin','scatter-done']]
                expected += [f'seed{seed}-finalization-begin',f'seed{seed}-finalization-done']
            expected+=['archives-done'];barrier=r['host_barrier'];assert [e['stage'] for e in barrier['events']]==expected and barrier['gpu']==int(pair>=3)
            times={};base=Path(barrier['directory']).relative_to('artifacts/research/20261114/work/bp_first_packed_dispatch')
            for e in barrier['events']:
                times[e['stage']]=[]
                for side,gpu in [('own',barrier['gpu']),('peer',1-barrier['gpu'])]:
                    raw=markers[str(base/e[side+'_marker'])];assert hashlib.sha256(raw.encode()).hexdigest()==e[side+'_sha256'];marker=json.loads(raw);worker=json.loads((directory/f'timing-worker{gpu}.json').read_text())
                    assert worker['status']=='complete' and worker['worker_wall_seconds']<=2400 and worker['peak_cpu_rss_kib']<=8*1024**2
                    assert marker['stage']==e['stage'] and marker['gpu']==gpu and marker['pid']==worker['pid'] and marker['protocol_sha256']==ph
                    times[e['stage']].append(marker['arrived_monotonic'])
                assert 0<=e['wait_seconds']<=60.01;events_checked+=1
            total=segment(r['readiness'],times['readiness-begin'],times['readiness-done']);ready=r['model_readiness']
            assert len(ready)==6 and {(q['seed'],q['model']) for q in ready}=={(s,m) for s in range(3) for m in ['bp','lp']}
            assert r['readiness_orchestration_seconds']>=0 and abs(sum(q['seconds'] for q in ready)+r['readiness_orchestration_seconds']-total)<1e-8
            for seed,model in enumerate(r['models']):
                assert model['seed']==seed and model['indices']==list(range(4096)) and model['actual_inputs']==4096 and model['slots_per_model']==129
                for stage in ['preparation','finalization']:total+=segment(model[stage],times[f'seed{seed}-{stage}-begin'],times[f'seed{seed}-{stage}-done'])
                for key in ['queue_file','global_answer_file']:assert sha(folder/model[key])==model[key+'_sha256']
                with np.load(folder/model['queue_file'],allow_pickle=False) as q:
                    assert np.array_equal(q['global_indices'],np.arange(4096))
                    if mode=='packed':assert set(q.files)=={'global_indices','high','distances'} and np.array_equal(q['distances'],distances) and np.array_equal(q['high'],distances>=182)
                    else:assert q.files==['global_indices']
                groups=[('original',np.arange(4096))] if mode=='original_order' else [('high',np.flatnonzero(distances>=182)),('low',np.flatnonzero(distances<182))]
                batches=[(name,ids[start:start+32]) for name,ids in groups for start in range(0,len(ids),32)]
                assert model['zero_work_slots']==129-len(batches)
                for slot,row in enumerate(r['rows'][seed*129:(seed+1)*129]):
                    label=f'seed{seed}-slot{slot:03d}';zero=slot>=len(batches);assert row['seed']==seed and row['slot']==slot and row['zero_work']==zero and row['archive_tasks_at_answer_start']==0
                    if zero:
                        assert row['indices']==[] and row['file'] is None and row['trace_files']==[] and row['answer_timing'] is row['materialization'] is row['scatter'] is None;continue
                    group,ids=batches[slot];assert row['group']==group and row['indices']==ids.tolist()
                    if len(ids)<32:partial.update((int(i),seed) for i in ids)
                    total+=segment(row['materialization'],times[label+'-materialize-begin'],times[label+'-materialize-done'])+segment(row['scatter'],times[label+'-scatter-begin'],times[label+'-scatter-done'])
                    t=row['answer_timing'];c=t['components'];required={'input_transfer','hidden_readout_and_flags','LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return','final_numerical_acceptance','BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return','geometry_screen_and_indices','LP_subset_gather','subset_scatter_and_provenance','endpoint_extraction_and_dispatch','BP_subset_gather'}
                    assert required<=c.keys() and all(np.isfinite(v) and v>=0 for v in c.values()) and t['n']==len(ids)
                    assert t['unattributed_in_segment_seconds']>=0 and abs(sum(c.values())+t['unattributed_in_segment_seconds']-t['answer_seconds'])<1e-8
                    assert t['answer_started_monotonic']>=max(times[label+'-materialize-done']) and t['answer_finished_monotonic']<=times[label+'-answer-done'][0]
                    assert t['diagnostic_started_monotonic']>=max(times[label+'-answer-done']) and t['diagnostic_started_monotonic']<=times[label+'-scatter-begin'][0]
                    assert t['actual_BP_inputs']+t['direct_LP_inputs']==len(ids) and t['actual_subset_inputs']==t['direct_LP_inputs']+t['rescue_LP_inputs']
                    for fields,count in [(['BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return'],t['actual_BP_inputs']),(['LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return'],t['actual_subset_inputs'])]:
                        for field in fields:assert (c[field]==0)==(count==0)
                    total+=t['answer_seconds']
            assert abs(total-r['complete_answer_seconds'])<1e-8;totals[pair,mode]=r['complete_answer_seconds']
    assert events_checked==23400
    values=[Fraction(1)-Fraction(totals[i,'packed'])/Fraction(totals[i,'original_order']) for i in range(6)];ordered=sorted(values);median=(ordered[2]+ordered[3])/2
    return totals,values,median,partial,events_checked


def verify(directory):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text());seal=json.loads((directory/'sealed-replays.json').read_text());manifest=json.loads((directory/'evidence-manifest.json').read_text());ph=sha(directory/'protocol.json')
    assert a['status']=='passed' and seal['status']=='complete' and a['protocol_sha256']==seal['protocol_sha256']==manifest['protocol_sha256']==ph
    assert seal['endpoint_aggregates_joined'] is False and len(seal['receipt_sha256'])==35 and sha(directory/'sealed-replays.json')==a['seal_sha256']
    assert manifest['audit_sha256']==sha(directory/'audit.json') and len(manifest['timed_globals'])==36 and set(manifest['source_globals'])=={'0','1','2'}
    assert a['fresh_scientific_inputs']==a['new_scientific_jobs']==a['new_scientific_evaluations']==0 and a['timed_model_pipeline_copies']==147456
    for name,digest in seal['receipt_sha256'].items():assert sha(directory/name)==digest
    context=directory/'source-context';sa=json.loads((context/'audit.json').read_text());assert sa['status']=='passed' and sha(context/'audit.json')==p['source']['audit_sha256']==a['source_audit_sha256']
    assert sha(context/'protocol.json')==p['source']['protocol_sha256']==sa['protocol_sha256'] and sha(context/'sealed-replays.json')==p['source']['seal_sha256']
    with (context/'per-model-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==49152 and sha(context/'per-model-input.csv')==sa['per_model_rows_sha256'];bycase={(int(r['input_index']),int(r['model_seed']),r['arm']):r for r in rows}
    datasets=json.loads((context/'datasets.json').read_text())['datasets'];tokens=[];labels=[]
    for shard,ds in enumerate(datasets):
        file=context/'datasets'/Path(ds['path']).name;assert sha(file)==ds['sha256'] and ds==p['source']['datasets'][shard]
        with np.load(file,allow_pickle=False) as z:assert np.array_equal(z['indices'],np.arange(shard*256,(shard+1)*256));tokens.append(z['tokens']);labels.append(z['labels'])
    tokens=np.concatenate(tokens);labels=np.concatenate(labels);assert all(valid_shortest(t,y) for t,y in zip(tokens,labels))
    distance=np.array([int(np.abs(np.argwhere(t==2)[0]-np.argwhere(t==3)[0]).sum()) for t in tokens]);totals,values,median,partial,barriers=costs(directory,ph,distance)
    endpoint=a['cost_endpoint'];assert str(median)==endpoint['exact_median_fraction'] and float(median)==endpoint['median_saving'] and endpoint['minimum_saving']==.10
    decision='supported' if median>=Fraction(1,10) else 'falsified';assert endpoint['decision']==decision
    for pair,r in enumerate(endpoint['pairs']):assert r=={'pair':pair,'original_order_seconds':totals[pair,'original_order'],'packed_seconds':totals[pair,'packed'],'saving':float(values[pair])}
    fdir=directory/'state-fixtures';fixtures=json.loads((fdir/'manifest.json').read_text());fixed=set()
    for shard in range(16):
        for seed in range(3):
            for high in [False,True]:
                ids=np.flatnonzero((np.arange(4096)//256==shard)&((distance>=182)==high))
                if len(ids):fixed.add((int(ids[0]),seed))
    source_selected=fixed|{(int(r['input_index']),int(r['model_seed'])) for r in rows if r['routed']=='1' or r['success']=='0' or r['BP_geometry_valid_longer']=='1'}
    selected={(r['input_index'],r['model_seed']) for r in fixtures['records']};assert selected==source_selected|partial and len(source_selected)==fixtures['source_fixed_selection'] and len(partial)==fixtures['all_actual_partial_batch_inputs']
    for name,digest in fixtures['parameters_sha256'].items():assert sha(fdir/name)==digest and p['source_evidence_sha256'][p['source']['directory']+'/'+name]==digest
    checked=0;max_margin=max_repeat_margin=0.
    for seed in range(3):
        entry=manifest['source_globals'][str(seed)];file=directory/entry['file'];assert sha(file)==entry['sha256'] and len(entry['source_batches'])==128
        assert {(x['shard'],x['batch']) for x in entry['source_batches']}=={(shard,batch) for shard in range(16) for batch in range(8)}
        reference_rows={}
        cohort=json.loads((context/'runs/block0-endpoint_dispatch/result.json').read_text());assert sha(context/'runs/block0-endpoint_dispatch/result.json')==p['source']['cohort_receipts_sha256']['endpoint_dispatch']
        for row in cohort['rows']:reference_rows[row['seed'],row['shard'],row['batch']]=row
        for item in entry['source_batches']:assert reference_rows[seed,item['shard'],item['batch']]['file_sha256']==item['sha256']
        with np.load(file,allow_pickle=False) as z:reference={k:z[k] for k in z.files}
        assert np.array_equal(reference['global_indices'],np.arange(4096)) and np.array_equal(reference['model_seed'],np.full(4096,seed))
        for i,board in enumerate(reference['prediction']):
            scalar=bycase[i,seed,'dispatcher'];graph=path_valid(tokens[i],board) and int(board.sum())==int(labels[i].sum());assert int(scalar['graph_valid_shortest'])==graph
            assert int(reference['chosen_model'][i])==int(scalar['chosen_model']) and int(reference['accepted'][i])==int(scalar['numerically_accepted']) and int(graph and reference['accepted'][i])==int(scalar['success'])
        selected_copies=[r for r in manifest['timed_globals'] if r['model_seed']==seed];assert {(r['pair'],r['mode']) for r in selected_copies}=={(i,m) for i in range(6) for m in ['original_order','packed']}
        for copy in selected_copies:
            file=directory/copy['file'];assert sha(file)==copy['sha256'];receipt=json.loads((file.parent/'result.json').read_text());assert receipt['models'][seed]['global_answer_file_sha256']==copy['sha256']
            with np.load(file,allow_pickle=False) as z:
                assert set(z.files)==set(reference)
                for key in z.files:
                    if key=='own_margin':max_repeat_margin=max(max_repeat_margin,close(z[key],reference[key],1e-9))
                    else:assert np.array_equal(z[key],reference[key],equal_nan=True)
            checked+=4096
        params=[]
        for arm in ['pl','old_binary']:
            with np.load(fdir/f'model-seed{seed}-{arm}.npz',allow_pickle=False) as z:params.append((z['theta'].astype(LD),z['mu_offsets'],z['aoffs']))
        for item in [r for r in fixtures['records'] if r['model_seed']==seed]:
            i=item['input_index'];file=fdir/item['file'];assert sha(file)==item['sha256'] and item['H2br_fixed_selection']==((i,seed) in source_selected) and item['actual_partial_batch_input']==((i,seed) in partial)
            with np.load(file,allow_pickle=False) as z:
                assert np.array_equal(z['tokens'][0],tokens[i]) and np.array_equal(z['labels'][0],labels[i]) and z['indices'].tolist()==[i]
                model=int(reference['chosen_model'][i]);delta=flip(z['tokens'],reference['prediction'][i:i+1],z['couplings'].astype(LD),*params[model]);margin=np.where(z['tokens']==1,delta,np.inf).min();max_margin=max(max_margin,close(margin,reference['own_margin'][i],1e-9))
                board=reference['prediction'][i];clamps=((tokens[i]==1)|(board==(tokens[i]>=2))).all();accepted=reference['hidden_good'][i] and reference['finite'][i] and reference['stopped'][i] and not reference['native_cap'][i] and not reference['block_cap'][i] and clamps and np.isfinite(margin) and margin>=-1e-6
                assert accepted==reference['accepted'][i]
    assert checked==147456
    result={'status':'passed','protocol_sha256':ph,'timed_final_boards_and_scalars_checked':checked,'source_model_inputs_graph_checked':12288,'source_labels_checked':4096,'direct_factor_fixture_cases':len(selected),'all_actual_partial_batch_inputs':len(partial),'maximum_original_margin_error':max_margin,'maximum_repeated_margin_error':max_repeat_margin,'measured_barrier_events':barriers,'exact_median_fraction':str(median),'cost_decision':decision,'new_scientific_observations':0,'seconds':time.perf_counter()-started,'checker_sha256':sha(Path(__file__)),'scope':'All final boards/scalars and charged costs independently checked against source-derived global outputs; selected original-factor margins reconstructed. Complete original source and internal trajectories remain hash-bound under the full Work audit. Performance-only examined-cohort claim.'}
    (directory/'public-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);verify(parser.parse_args().directory)
