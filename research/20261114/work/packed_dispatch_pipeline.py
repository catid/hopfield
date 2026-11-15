"""Prospective H2bs common complete-cost harness; original batch solver unchanged."""
from pathlib import Path
import time
import numpy as np
import study as S
from bounded_archive_writer import ArchiveWriter
from endpoint_dispatch_batch import run as batch
from endpoint_dispatch_pipeline import load_models,payloads,drain
from bp_first_geometry_parity import array_sha
from packed_dispatch_queue import make_queue,materialize,scatter,identities
from packed_dispatch_accounting import segment,complete


def cohort(p,ph,mode,tokens_by_shard,indices,directory,watchdog,progress,barrier,pair=None,source_gate=False):
    torch,_,_=S.imports();directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();writer=ArchiveWriter(asynchronous=True,max_pending=2,memory_gib=8)
    rows=[];model_records=[];drain_seconds=payload_seconds=0.
    try:
        assert not writer.pending;barrier.gate('readiness-begin');torch.cuda.synchronize();began=time.perf_counter()
        models,doubles,readiness=load_models(p,'endpoint_dispatch');torch.cuda.synchronize();readiness_segment=segment(began,time.perf_counter())
        barrier.gate('readiness-done');watchdog()
        with torch.no_grad():
            for seed in range(3):
                drain_seconds+=drain(writer);assert not writer.pending
                barrier.gate(f'seed{seed}-preparation-begin');began=time.perf_counter()
                original=identities(indices);queue=make_queue(tokens_by_shard,original,mode)
                final={};seen=np.zeros(len(original),bool)
                preparation=segment(began,time.perf_counter());barrier.gate(f'seed{seed}-preparation-done')
                for slot in range(queue['slots_per_model']):
                    watchdog();drained=drain(writer);drain_seconds+=drained;assert not writer.pending
                    label=f'seed{seed}-slot{slot:03d}';zero=slot>=len(queue['batches'])
                    barrier.gate(label+'-materialize-begin')
                    if not zero:
                        began=time.perf_counter();record=queue['batches'][slot];actual_indices=record['indices'];host_tokens=materialize(tokens_by_shard,actual_indices)
                        host_segment=segment(began,time.perf_counter())
                    else:actual_indices=np.zeros(0,np.int64);record=None;host_segment=None
                    barrier.gate(label+'-materialize-done')
                    if not zero:
                        answer=batch('endpoint_dispatch',host_tokens,models[seed],doubles[seed],watchdog,after_answer=lambda:barrier.gate(label+'-answer-done'))
                    else:barrier.gate(label+'-answer-done');answer=None
                    # The unchanged batch routine has completed all diagnostics
                    # before entering this host-wide charged scatter boundary.
                    barrier.gate(label+'-scatter-begin')
                    if not zero:
                        began=time.perf_counter();positions=scatter(final,seen,original,actual_indices,answer['answer']);scatter_segment=segment(began,time.perf_counter())
                    else:scatter_segment=None
                    barrier.gate(label+'-scatter-done')
                    row={'seed':seed,'slot':slot,'zero_work':zero,'indices':actual_indices.tolist(),'group':record['group'] if record else None,
                         'materialization':host_segment,'scatter':scatter_segment,'answer_timing':answer['timing'] if answer else None,
                         'archive_tasks_at_answer_start':0,'archive_drain_seconds':drained,'file':None,'trace_files':[]}
                    if not zero:
                        before=time.perf_counter();item,traces=payloads(answer,directory,label,ph,True,0)
                        item[1]['global_indices']=actual_indices.copy();item[1]['original_order_positions']=positions.copy()
                        payload_seconds+=time.perf_counter()-before;writer.submit([item])
                        if traces:writer.submit(traces)
                        row.update(file=item[0].name,trace_files=[x[0].name for x in traces],actual_couplings_sha256=array_sha(answer['audit']['couplings']),tokens_sha256=array_sha(host_tokens))
                        del answer,item,traces,host_tokens
                    rows.append(row);S.save(directory/(label+'.json'),row)
                    progress(mode=mode,seed=seed,slot=slot,slots_complete=len(rows),model_inputs_complete=sum(len(r['indices']) for r in rows))
                drain_seconds+=drain(writer);assert not writer.pending
                barrier.gate(f'seed{seed}-finalization-begin');began=time.perf_counter()
                assert seen.all() and all(len(value)==len(original) for value in final.values())
                final['global_indices']=original.copy();final['model_seed']=np.full(len(original),seed,np.int64)
                finalization=segment(began,time.perf_counter());barrier.gate(f'seed{seed}-finalization-done')
                model_record={'seed':seed,'indices':original.tolist(),'actual_inputs':len(original),'slots_per_model':queue['slots_per_model'],'zero_work_slots':queue['zero_work_slots'],
                    'preparation':preparation,'finalization':finalization,'global_answer_file':f'seed{seed}-global-answer.npz','queue_file':f'seed{seed}-queue.npz'}
                queue_arrays={'global_indices':queue['indices']}
                if queue['high'] is not None:queue_arrays.update(distances=queue['distances'],high=queue['high'])
                writer.submit([(directory/model_record['global_answer_file'],final),(directory/model_record['queue_file'],queue_arrays)])
                model_records.append(model_record);del final,seen,queue,queue_arrays;watchdog()
        files=writer.close();barrier.gate('archives-done');byname={Path(r['path']).name:r for r in files}
        for row in rows:
            if not row['zero_work']:
                row['file_sha256']=byname[row['file']]['sha256'];row['trace_sha256']={name:byname[name]['sha256'] for name in row['trace_files']}
        for row in model_records:
            for key in ['global_answer_file','queue_file']:row[key+'_sha256']=byname[row[key]]['sha256']
        cost=complete(readiness_segment['seconds'],model_records,rows)
        result={'status':'complete','mode':mode,'protocol_sha256':ph,'pair':pair,'source_gate':source_gate,'model_inputs':3*len(indices),'rows':rows,'models':model_records,
            'model_readiness':readiness,'readiness':readiness_segment,'readiness_orchestration_seconds':readiness_segment['seconds']-sum(r['seconds'] for r in readiness),
            'complete_answer_seconds':cost,'archive_boundary_drain_seconds':drain_seconds,'payload_seconds':payload_seconds,'archive_writer_metrics':writer.metrics,
            'all_writes_closed_hashed_and_joined':writer.closed,'archive_tasks_during_answer_segments':0,'cohort_wall_seconds':time.perf_counter()-started,
            'host_barrier':{'directory':str(barrier.directory.relative_to(S.ROOT)),'gpu':barrier.gpu,'events':barrier.events,'wait_seconds':sum(r['wait_seconds'] for r in barrier.events)},
            'scope':'Complete offline cohort processing under newly shared charged host harness. Every actual input exactly once; trailing slots add host waits only. No new scientific observations.'}
        S.save(directory/'result.json',result);del models,doubles;torch.cuda.empty_cache();return result
    finally:
        if not writer.closed:writer.close()
