"""Complete H2bk evidence with immutable prefixes and bounded async archives."""
import json
from pathlib import Path
import time
import numpy as np
import study as S
import fallback_confirmation_batch as H
import confirmation96_batch as Q
from factorial_transfer import hidden
from bounded_archive_writer import ArchiveWriter
from artifact_overlap_pipeline import KEYS,AXIS0,REPORTED


def run_job(p,job,tokens,models,doubles,run,protocol_hash,watchdog=None,asynchronous=True):
    torch,_,_=S.imports();run=Path(run);run.mkdir(parents=True,exist_ok=True)
    assert not (run/'result.json').exists();n=len(tokens);assert n in [32,256]
    writer=ArchiveWriter(asynchronous);started=time.perf_counter();saved={k:[] for k in KEYS};hybrids={k:[] for k in H.HYBRID_KEYS}
    timings=[];ht=[];traces=[];batches=[];subset_receipts=[];hybrid_receipts=[];input_seconds=hidden_seconds=diagnostic_seconds=payload_seconds=0.
    try:
        with torch.no_grad():
            for lo in range(0,n,32):
                torch.cuda.synchronize();before=time.perf_counter();t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');torch.cuda.synchronize();input_seconds+=time.perf_counter()-before
                before=time.perf_counter();a,hs,good=hidden(models[0],t);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                result=H.solve(models,doubles,t,a,good,rotation=(job['seed']+job['shard']+lo//32)%8,watchdog=watchdog)
                out=result['outputs'];prefix=result['prefixes'];block=result['blocks'];timings.append(result['timing']);ht.append(result['hybrid_timing']);before=time.perf_counter()
                batch={k:np.stack([out[v]['cpu_output'][name] if v in REPORTED else out[v][name].cpu().numpy() for v in range(8)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                for v in range(8):
                    model=doubles[Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(model.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(model.energy(t,board,a.double()).cpu().numpy())
                batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1]]),block_moves=np.stack([block[v]['block_moves'] for v in [2,3,4,5,6,7]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [2,3,4,5,6,7]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [2,3,4,5,6,7]]),block_cap=np.stack([block[v]['block_cap'] for v in [2,3,4,5,6,7]]),initial_simple_path=np.stack([block[v]['initial_simple_path'] for v in [2,3,4,5,6,7]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());torch.cuda.synchronize();diagnostic_seconds+=time.perf_counter()-before+result['hybrid_timing']['subset_diagnostic_seconds'];before=time.perf_counter()
                bp=run/f'batch{lo//32:02d}.npz';items=[(bp,batch)];batches.append(bp.name)
                for v in [2,3,4,5,6,7]:
                    raw=block[v]['raw'];path=run/f'batch{lo//32:02d}-{Q.ARMS[v]}-trace.npz';payload={'protocol_sha256':protocol_hash,'offset':lo,'arm':Q.ARMS[v],'trace':raw['trace'] if raw else [],'initial_terminal_checks':block[v]['initial_terminal_records'],'final_terminal_checks':raw.get('terminal_check_records',[]) if raw else []};arrays={'trace_utf8':np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8),'valid_prefix_indices':block[v]['valid_prefix_indices'],'states_before':raw['states_before'] if raw else np.zeros((0,0,180,180),bool),'initial_terminal':block[v]['initial_terminal'],'pre_terminal_stopped':raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool),'pre_terminal_block_cap':raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool)};items.append((path,arrays));traces.append({'offset':lo,'arm':Q.ARMS[v],'path':path.name})
                for name,arrays,receipts in [('hybrid',result['hybrid'],hybrid_receipts),('subset',result['subset'],subset_receipts)]:
                    path=run/f'batch{lo//32:02d}-{name}.npz';items.append((path,arrays));receipts.append({'offset':lo,'path':path.name})
                payload_seconds+=time.perf_counter()-before;writer.submit(items)
                for k in KEYS:saved[k].append(batch[k])
                for k in H.HYBRID_KEYS:hybrids[k].append(result['hybrid'][k])
                if watchdog:watchdog()
        before=time.perf_counter();arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};hybrid_arrays={k:np.concatenate(v) for k,v in hybrids.items()};payload_seconds+=time.perf_counter()-before
        writer.submit([(run/'outputs.npz',arrays),(run/'hybrid-outputs.npz',hybrid_arrays)]);files=writer.close();byfile={Path(r['path']).name:r for r in files}
        receipt={'status':'complete','utc':S.utc(),'job':job,'n':n,'protocol_sha256':protocol_hash,
            'dataset_sha256':p['datasets'][job['shard']]['sha256'] if 'datasets' in p else None,
            'checkpoint_sha256_by_arm':[p['models'][str(job['seed'])][arm]['sha256'] for arm in Q.ARMS],
            'outputs_sha256':byfile['outputs.npz']['sha256'],'hybrid_outputs_sha256':byfile['hybrid-outputs.npz']['sha256'],
            'input_transfer_seconds':input_seconds,'hidden_seconds_shared':hidden_seconds,
            'answer_seconds_by_arm':np.sum([x['charged_answer_seconds'] for x in timings],0).tolist(),
            'unique_answer_seconds_by_operation':np.sum([x['unique_operation_seconds'] for x in timings],0).tolist(),
            'cpu_return_seconds_by_arm':np.sum([x['cpu_return_seconds'] for x in timings],0).tolist(),
            'batch_timings':timings,'hybrid_batch_timings':ht,
            'main_hybrid_charged_seconds':sum(x['main_charged_seconds'] for x in ht),
            'strict_hybrid_charged_seconds':sum(x['strict_charged_seconds'] for x in ht),
            'actual_subset_inputs':sum(x['actual_subset_inputs'] for x in ht),
            'trace_archives':[{**r,'sha256':byfile[r['path']]['sha256']} for r in traces],
            'batch_archives':[{'path':n,'sha256':byfile[n]['sha256']} for n in batches],
            'subset_archives':[{**r,'sha256':byfile[r['path']]['sha256']} for r in subset_receipts],
            'hybrid_archives':[{**r,'sha256':byfile[r['path']]['sha256']} for r in hybrid_receipts],
            'diagnostic_seconds':diagnostic_seconds,'archive_seconds':writer.metrics['ownership_copy_seconds']+writer.metrics['backpressure_seconds']+writer.metrics['final_join_seconds'],
            'archive_writer_metrics':writer.metrics,'payload_build_seconds':payload_seconds,
            'job_seconds':time.perf_counter()-started,'all_writes_closed_hashed_and_joined':writer.closed}
        S.save(run/'result.json',receipt)
        return receipt
    finally:
        if not writer.closed:writer.close()
