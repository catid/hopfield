"""H2bi frozen matched complete-pipeline artifact-overlap benchmark."""
import argparse
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
from artifact_overlap_pipeline import pipeline,verify_source
from confirmation96_study import load_models,protocol as source_protocol

DEST=S.OUT/'artifact_overlap';PP=S.HERE/'artifact-overlap-protocol.json'


def protocol():
    p=json.loads(PP.read_text())
    for path,h in {**p['source_sha256'],**p['source_evidence_sha256']}.items():assert S.sha(S.ROOT/path)==h,path
    return p


def setup(gpu,kind):
    started=time.perf_counter();lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();ds=p['source_cohort']['dataset'];assert S.sha(S.ROOT/ds['path'])==ds['sha256'];before=time.perf_counter()
    with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
    source=source_protocol();models={};doubles={}
    for seed in range(3):models[seed],doubles[seed]=load_models(source,seed)
    preload=time.perf_counter()-before;state={'status':'running','kind':kind,'gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'watchdog_checks':0,'completed_replays':[],'preload_seconds':preload};last=[0.]
    def watch():
        elapsed=time.perf_counter()-started;state['watchdog_checks']+=1
        if elapsed>(120 if kind=='gate' else 240):state.update(status='budget_exhausted',worker_seconds=elapsed);S.save(DEST/f'{kind}-worker{gpu}.json',state);raise TimeoutError('Frozen workerbudget; partialoutputs retained, no extension')
        if elapsed-last[0]>=1:state.update(worker_seconds=elapsed,updated_utc=S.utc());S.save(DEST/f'{kind}-worker{gpu}.json',state);last[0]=elapsed
    watch();return lock,torch,p,tokens,models,doubles,state,watch,started


def gate(gpu):
    lock,torch,p,tokens,models,doubles,state,watch,started=setup(gpu,'gate');seeds=[0,2] if gpu==0 else [1];rows=[]
    for mode in ['serialized','async']:
        state['phase']=mode;directory=DEST/'source-gates'/f'gpu{gpu}-{mode}';result=pipeline(tokens,models,doubles,directory,mode=='async',seeds=seeds,watchdog=watch);checked=verify_source(directory,p['source_cohort']);assert len(checked)==len(seeds);rows.append({'mode':mode,'pipeline':result,'source_checks':checked});state['completed_replays'].append(mode);watch()
    state.update(status='complete',worker_seconds=time.perf_counter()-started);S.save(DEST/f'gate-worker{gpu}.json',state);S.save(DEST/f'source-gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(PP),'gpu':gpu,'seeds':seeds,'model_inputs_per_mode':256*len(seeds),'rows':rows,'worker':state,'scope':'Both write implementations preserve every original H2bh shard00 batch/final/trace archive byte, split bymodelseed. Mandatory gate only; no speed endpoint or fresh accuracy.'})


def benchmark(gpu):
    p=protocol();gates=[json.loads((DEST/f'source-gate-gpu{g}.json').read_text()) for g in [0,1]];assert all(g['status']=='passed' and g['protocol_sha256']==S.sha(PP) for g in gates);assert sum(g['model_inputs_per_mode'] for g in gates)==768
    lock,torch,p,tokens,models,doubles,state,watch,started=setup(gpu,'timing');warmups=[]
    for mode in ['serialized','async']:
        state['phase']='warmup-'+mode;torch.cuda.reset_peak_memory_stats();r=pipeline(tokens[:32],models,doubles,DEST/'warmup'/f'gpu{gpu}-{mode}',mode=='async',watchdog=watch);warmups.append({'mode':mode,'pipeline':r});watch()
    rows=[]
    for rep in ([0,1] if gpu==0 else [2,3]):
        row={'repetition':rep,'gpu':gpu,'order':p['paired_order'][rep],'runs':{}}
        for mode in row['order']:
            state['phase']=f'repetition{rep}-{mode}';torch.cuda.reset_peak_memory_stats();directory=DEST/'runs'/f'gpu{gpu}-rep{rep}-{mode}';r=pipeline(tokens,models,doubles,directory,mode=='async',watchdog=watch);watch();before=time.perf_counter();checks=verify_source(directory,p['source_cohort']);assert len(checks)==3;row['runs'][mode]={'pipeline':r,'source_checks':checks,'source_verification_seconds':time.perf_counter()-before,'directory':str(directory.relative_to(S.ROOT))};state['completed_replays'].append(state['phase']);watch();S.emit('artifact_overlap_replay_complete',gpu=gpu,rep=rep,mode=mode,seconds=r['seconds'],worker_seconds=time.perf_counter()-started)
        row['paired_pipeline_speedup']=row['runs']['serialized']['pipeline']['seconds']/row['runs']['async']['pipeline']['seconds'];rows.append(row);S.save(DEST/f'pairs-gpu{gpu}.json',{'protocol_sha256':S.sha(PP),'rows':rows})
    state.update(status='complete',worker_seconds=time.perf_counter()-started);S.save(DEST/f'timing-worker{gpu}.json',state);S.save(DEST/f'benchmark-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(PP),'gpu':gpu,'rows':rows,'warmups':warmups,'worker':state,'mandatory_gate_sha256':{f'source-gate-gpu{g}.json':S.sha(DEST/f'source-gate-gpu{g}.json') for g in [0,1]}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['gate','benchmark']);ap.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args()
    try:globals()[a.command](a.gpu)
    except BaseException as error:
        import traceback
        S.save(DEST/f'failure-{a.command}-gpu{a.gpu}.json',{'status':'failed','command':a.command,'gpu':a.gpu,'error':repr(error),'traceback':traceback.format_exc(),'utc':S.utc(),'partial_outputs_retained':True});raise
