"""H2bi independent raw-file parity, timing, queue and resource accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory,root):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());ph=sha(directory/'protocol.json');expected=p['source_cohort'];rows=[];total_files=0;total_bytes=0
    for path,h in {**p['source_sha256'],**p['source_evidence_sha256']}.items():assert sha(root/path)==h
    gates=[]
    for gpu in [0,1]:
        gate=json.loads((directory/f'source-gate-gpu{gpu}.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==ph and gate['worker']['worker_seconds']<=120;gates.append(gate)
        result=json.loads((directory/f'benchmark-gpu{gpu}.json').read_text());worker=json.loads((directory/f'timing-worker{gpu}.json').read_text());assert result['status']=='passed' and result['protocol_sha256']==ph and worker['status']=='complete' and worker['worker_seconds']<=240
        assert [w['mode'] for w in result['warmups']]==['serialized','async'];assert all(w['pipeline']['n']==32 and w['pipeline']['all_writes_closed_hashed_and_joined'] for w in result['warmups'])
        assert result['mandatory_gate_sha256']=={f'source-gate-gpu{g}.json':sha(directory/f'source-gate-gpu{g}.json') for g in [0,1]}
        assert [r['repetition'] for r in result['rows']]==([0,1] if gpu==0 else [2,3])
        for row in result['rows']:
            assert row['gpu']==gpu and row['order']==p['paired_order'][row['repetition']];durations={}
            for mode in ['serialized','async']:
                run=row['runs'][mode];r=run['pipeline'];metrics=r['writer_metrics'];assert r['n']==256 and r['seeds']==[0,1,2] and r['asynchronous']==(mode=='async');assert r['all_writes_closed_hashed_and_joined'];assert np.isfinite(r['seconds']) and r['seconds']>0;durations[mode]=r['seconds'];assert metrics['maximum_pending']<=(2 if mode=='async' else 0);assert metrics['maximum_rss_kib']<=8*1024**2 and metrics['maximum_rss_kib']>0;assert metrics['tasks']==27 and metrics['files']==171 and r['complete_artifact_files']==174
                assert all(np.isfinite(metrics[k]) and metrics[k]>=0 for k in ['ownership_copy_seconds','backpressure_seconds','final_join_seconds','writer_service_seconds']);files={f['path']:f for f in r['files']};assert len(files)==171 and len(r['receipts'])==3
                actual_bytes=0
                for job in expected['jobs']:
                    d=root/run['directory']/job['id'];receipt_path=d/'result.json';receipt=json.loads(receipt_path.read_text());assert receipt['status']=='complete' and receipt['source_job']==job['source_job'] and receipt['n']==256 and receipt['source_protocol_sha256']==expected['source_protocol_sha256'];receipt_record=next(x for x in r['receipts'] if x['source_job']==job['source_job']);assert sha(receipt_path)==receipt_record['sha256'] and receipt_path.stat().st_size==receipt_record['bytes'];actual_bytes+=receipt_record['bytes'];assert len(receipt['trace_archives'])==48 and len(receipt['batch_archives'])==8 and len(receipt['batch_timings'])==8;assert np.allclose(np.sum([b['charged_answer_seconds'] for b in receipt['batch_timings']],0),receipt['answer_seconds_by_arm'],rtol=0,atol=1e-12)
                    assert {(x['offset'],x['arm']) for x in receipt['trace_archives']}=={(offset,arm) for offset in range(0,256,32) for arm in p['source_internal_arms'][2:]};actual={f.name:sha(f) for f in d.glob('*.npz')};assert actual==job['files_sha256'];assert receipt['outputs_sha256']==actual['outputs.npz'];assert all(x['sha256']==actual[x['path']] for x in receipt['trace_archives']+receipt['batch_archives'])
                    for name,h in actual.items():
                        path=d/name;record=files[str(path)];assert record['sha256']==h and record['bytes']==path.stat().st_size;actual_bytes+=record['bytes'];total_files+=1;total_bytes+=record['bytes']
                assert actual_bytes==r['complete_artifact_bytes'] and sum(f['bytes'] for f in r['files'])==metrics['bytes']
            ratio=durations['serialized']/durations['async'];assert ratio==row['paired_pipeline_speedup'];rows.append({'repetition':row['repetition'],'gpu':gpu,'order':row['order'],'seconds':durations,'paired_pipeline_speedup':ratio,'writer_metrics':{mode:row['runs'][mode]['pipeline']['writer_metrics'] for mode in ['serialized','async']},'complete_artifact_bytes':{mode:row['runs'][mode]['pipeline']['complete_artifact_bytes'] for mode in ['serialized','async']},'peak_cuda_allocated_bytes':{mode:row['runs'][mode]['pipeline']['peak_cuda_allocated_bytes'] for mode in ['serialized','async']},'peak_cuda_reserved_bytes':{mode:row['runs'][mode]['pipeline']['peak_cuda_reserved_bytes'] for mode in ['serialized','async']}})
    assert sum(g['model_inputs_per_mode'] for g in gates)==768 and len(rows)==4;rows.sort(key=lambda r:r['repetition']);median=float(np.median([r['paired_pipeline_speedup'] for r in rows]));assert time.perf_counter()-started<120
    result={'status':'passed','hypothesis':'H2bi','protocol_sha256':ph,'rows':rows,'median_paired_complete_pipeline_speedup':median,'minimum_target':1.15,'decision':'supported' if median>=1.15 else 'falsified','byte_identical_npz_files':total_files,'byte_identical_npz_bytes':total_bytes,'complete_pipeline_model_inputs_compared':6144,'reported_arm_model_states_compared':24576,'all_internal_prefix_model_states_compared':49152,'source_joint_success_by_internal_arm':expected['source_joint_success_by_internal_arm'],'worker_receipts':[json.loads((directory/f'timing-worker{g}.json').read_text()) for g in [0,1]],'mandatory_gate_sha256':{f'source-gate-gpu{g}.json':sha(directory/f'source-gate-gpu{g}.json') for g in [0,1]},'seconds':time.perf_counter()-started,'audit_source_sha256':sha(Path(__file__)),'scope':'Fourbalanced matched complete researchpipeline timings, all171scientificNPZfiles perreplay independently bytecompared with frozenH2bhsource, every receipt/queue/memorybudget checked. Timed writecompletion includes full finaljoin/hash/receiptwork. Zero fresh accuracy samples; no model-inference/fsyncdurability or universalhardware claim.'};(directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('root',type=Path);a=ap.parse_args();r=audit(a.directory,a.root);print(json.dumps({k:v for k,v in r.items() if k not in ['worker_receipts']},indent=2))
