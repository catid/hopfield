"""H2bi complete unchanged H2bh computation with sync versus bounded writes."""
import json
from pathlib import Path
import time
import numpy as np
import study as S
import confirmation96_batch as Q
from factorial_transfer import hidden
from bounded_archive_writer import ArchiveWriter

SOURCE=S.OUT/'loop80_confirmation'
KEYS=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path','hidden_steps','hidden_good','couplings']
AXIS0={'hidden_steps','hidden_good','couplings'}
REPORTED=(0,5,6,7)


def pipeline(tokens,models,doubles,output_directory,asynchronous,seeds=(0,1,2),watchdog=None):
    """Clock includes every owned copy, archive hash, final receipt and writer join."""
    torch,_,_=S.imports();output_directory=Path(output_directory);assert not output_directory.exists();source_protocol_sha=S.sha(SOURCE/'protocol.json');source_p=json.loads((SOURCE/'protocol.json').read_text());n=len(tokens);assert n in [32,256];torch.cuda.synchronize();started=time.perf_counter();writer=ArchiveWriter(asynchronous);metadata=[];answer=diagnostic=ownership_build=0.
    try:
        with torch.no_grad():
            for seed in seeds:
                job=next(j for j in source_p['jobs'] if j['shard']==0 and j['seed']==seed);run=output_directory/job['id'];run.mkdir(parents=True);saved={k:[] for k in KEYS};trace_names=[];batch_names=[];batch_timings=[]
                for lo in range(0,n,32):
                    before=time.perf_counter();t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');a,hs,good=hidden(models[seed][0],t);r=Q.solve(models[seed],doubles[seed],t,a,rotation=(seed+lo//32)%8,watchdog=watchdog);out=r['outputs'];prefix=r['prefixes'];block=r['blocks'];batch_timings.append(r['timing']);torch.cuda.synchronize();answer+=time.perf_counter()-before;before=time.perf_counter()
                    batch={k:np.stack([out[v]['cpu_output'][name] if v in REPORTED else out[v][name].cpu().numpy() for v in range(8)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                    for v in range(8):
                        model=doubles[seed][Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(model.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(model.energy(t,board,a.double()).cpu().numpy())
                    batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[seed][0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1]]),block_moves=np.stack([block[v]['block_moves'] for v in [2,3,4,5,6,7]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [2,3,4,5,6,7]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [2,3,4,5,6,7]]),block_cap=np.stack([block[v]['block_cap'] for v in [2,3,4,5,6,7]]),initial_simple_path=np.stack([block[v]['initial_simple_path'] for v in [2,3,4,5,6,7]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());torch.cuda.synchronize();diagnostic+=time.perf_counter()-before;before=time.perf_counter();bp=run/f'batch{lo//32:02d}.npz';items=[(bp,batch)];batch_names.append(bp.name)
                    for v in [2,3,4,5,6,7]:
                        raw=block[v]['raw'];path=run/f'batch{lo//32:02d}-{Q.ARMS[v]}-trace.npz';payload={'protocol_sha256':source_protocol_sha,'offset':lo,'arm':Q.ARMS[v],'trace':raw['trace'] if raw else [],'initial_terminal_checks':block[v]['initial_terminal_records'],'final_terminal_checks':raw.get('terminal_check_records',[]) if raw else []};arrays={'trace_utf8':np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8),'valid_prefix_indices':block[v]['valid_prefix_indices'],'states_before':raw['states_before'] if raw else np.zeros((0,0,180,180),bool),'initial_terminal':block[v]['initial_terminal'],'pre_terminal_stopped':raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool),'pre_terminal_block_cap':raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool)};items.append((path,arrays));trace_names.append({'offset':lo,'arm':Q.ARMS[v],'path':path.name})
                    ownership_build+=time.perf_counter()-before;writer.submit(items)
                    for k in KEYS:saved[k].append(batch[k])
                    if watchdog:watchdog()
                before=time.perf_counter();arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};ownership_build+=time.perf_counter()-before;writer.submit([(run/'outputs.npz',arrays)]);metadata.append({'job':job,'n':n,'source_protocol_sha256':source_protocol_sha,'batch_names':batch_names,'trace_names':trace_names,'batch_timings':batch_timings})
        file_receipts=writer.close();byfile={r['path']:r for r in file_receipts};receipts=[]
        for row in metadata:
            run=output_directory/row['job']['id'];receipt={'status':'complete','source_job':row['job'],'n':n,'source_protocol_sha256':source_protocol_sha,'batch_timings':row['batch_timings'],'answer_seconds_by_arm':np.sum([b['charged_answer_seconds'] for b in row['batch_timings']],0).tolist(),'unique_answer_seconds_by_operation':np.sum([b['unique_operation_seconds'] for b in row['batch_timings']],0).tolist(),'cpu_return_seconds_by_arm':np.sum([b['cpu_return_seconds'] for b in row['batch_timings']],0).tolist(),'outputs_sha256':byfile[str(run/'outputs.npz')]['sha256'],'trace_archives':[{**r,'sha256':byfile[str(run/r['path'])]['sha256']} for r in row['trace_names']],'batch_archives':[{'path':name,'sha256':byfile[str(run/name)]['sha256']} for name in row['batch_names']]};path=run/'result.json';S.save(path,receipt);receipts.append({'source_job':row['job'],'path':str(path),'sha256':S.sha(path),'bytes':path.stat().st_size})
        torch.cuda.synchronize();writer.check_memory();seconds=time.perf_counter()-started
        return {'seconds':seconds,'writer_metrics':writer.metrics,'compute_and_transfer_seconds':answer,'diagnostic_seconds':diagnostic,'payload_and_aggregate_build_seconds':ownership_build,'files':file_receipts,'receipts':receipts,'complete_artifact_files':len(file_receipts)+len(receipts),'complete_artifact_bytes':sum(r['bytes'] for r in file_receipts+receipts),'n':n,'seeds':list(seeds),'asynchronous':asynchronous,'all_writes_closed_hashed_and_joined':writer.closed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
    finally:
        if not writer.closed:writer.close()


def verify_source(directory,source_manifest):
    directory=Path(directory);checked=[]
    for job in source_manifest['jobs']:
        run=directory/job['id']
        if not run.exists():continue
        row=json.loads((run/'result.json').read_text());assert row['status']=='complete' and row['n']==256;assert row['source_protocol_sha256']==source_manifest['source_protocol_sha256'];actual={p.name:S.sha(p) for p in run.glob('*.npz')};assert actual==job['files_sha256'],job['id'];assert row['outputs_sha256']==actual['outputs.npz'];assert len(row['trace_archives'])==48 and len(row['batch_archives'])==8;checked.append({'job':job['id'],'byte_identical_npz_files':len(actual),'source_outputs_sha256':actual['outputs.npz']})
    assert checked
    return checked
