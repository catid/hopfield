"""H2bc fresh4096 size180 loop50 unified bounded decoder confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import unified_cycle_batch as Q
import fast_npz as Z
from factorial_transfer import hidden

PP=S.HERE/'unified-cycle-protocol.json'
DEST=S.OUT/'unified_cycle'
BASE=2026111525
ARMS=Q.ARMS
AXIS0={'hidden_steps','hidden_good','couplings'}


def freeze():
    from freeze_unified_cycle import freeze_protocol
    freeze_protocol()


def protocol():
    p=json.loads(PP.read_text())
    for path,h in {**p['source_sha256'],**p['parent_protocol_sha256'],**p['source_gate_files']}.items():assert S.sha(S.ROOT/path)==h,path
    for group in p['models'].values():
        for m in group.values():assert S.sha(S.ROOT/m['path'])==m['sha256']
    return p


def load_models(p,seed,device='cuda'):
    torch,_,_=S.imports();models=[]
    for arm in ARMS[:2]:
        m=S.model_for(seed,180,'original',device);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location=device,weights_only=True));models.append(m)
    for k,v in models[0].state_dict().items():
        if k not in ['T','b','kappa']:assert torch.equal(v,models[1].state_dict()[k]),k
    return models,[copy.deepcopy(m).double() for m in models]


def make_one(index):return S.maze(180,.5,np.random.SeedSequence([BASE,180,50,index]))


def prepare():
    protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP);started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz';indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));Z.save(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('unified_cycle_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started,'gate_sha256':S.sha(DEST/'numerical-gate.json')})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);inp=json.loads((DEST/'independent-input-audit.json').read_text());assert inp['status']=='passed' and inp['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={};doubles={}
    def watchdog():
        if time.perf_counter()-started>p['budget']['per_worker_wall_seconds']:
            state.update(status='budget_exhausted',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);raise TimeoutError('Frozen600sworkerbudget; completed batch files retained')
    for seed in range(3):models[seed],doubles[seed]=load_models(p,seed)
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        watchdog();run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens']
        keys=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path','hidden_steps','hidden_good','couplings'];saved={k:[] for k in keys};timings=[];hidden_seconds=diagnostic_seconds=archive_seconds=0.;job_start=time.perf_counter();seed=job['seed'];trace_receipts=[]
        with torch.no_grad():
            for lo in range(0,256,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');torch.cuda.synchronize();before=time.perf_counter();a,hs,good=hidden(models[seed][0],t);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                result=Q.solve(models[seed],doubles[seed],t,a,rotation=(ji+lo//32)%6,watchdog=watchdog);out=result['outputs'];prefix=result['prefixes'];block=result['blocks'];timings.append(result['timing']);before=time.perf_counter()
                batch={k:np.stack([out[v][name].cpu().numpy() for v in range(6)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                for v in range(6):
                    m=doubles[seed][Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(m.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(m.energy(t,board,a.double()).cpu().numpy())
                batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[seed][0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1]]),block_moves=np.stack([block[v]['block_moves'] for v in [2,3,4,5]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [2,3,4,5]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [2,3,4,5]]),block_cap=np.stack([block[v]['block_cap'] for v in [2,3,4,5]]),initial_simple_path=np.stack([block[v]['initial_simple_path'] for v in [2,3,4,5]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                before=time.perf_counter();bp=run/f'batch{lo//32:02d}.npz';Z.save(bp,**batch)
                for v in [2,3,4,5]:
                    raw=block[v]['raw'];path=run/f'batch{lo//32:02d}-{ARMS[v]}-trace.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'arm':ARMS[v],'trace':raw['trace'] if raw else [],'initial_terminal_checks':block[v]['initial_terminal_records'],'final_terminal_checks':raw.get('terminal_check_records',[]) if raw else []};arrays={'trace_utf8':np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8),'valid_prefix_indices':block[v]['valid_prefix_indices'],'states_before':raw['states_before'] if raw else np.zeros((0,0,180,180),bool),'initial_terminal':block[v]['initial_terminal'],'pre_terminal_stopped':raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool),'pre_terminal_block_cap':raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool)};Z.save(path,**arrays);trace_receipts.append({'offset':lo,'arm':ARMS[v],'path':path.name,'sha256':S.sha(path)})
                archive_seconds+=time.perf_counter()-before
                for k in keys:saved[k].append(batch[k])
                state.update(current_job=job,completed_inputs=lo+32,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);watchdog()
        before=time.perf_counter();arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};Z.save(run/'outputs.npz',**arrays);archive_seconds+=time.perf_counter()-before
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':np.sum([x['charged_answer_seconds'] for x in timings],0).tolist(),'unique_answer_seconds_by_operation':np.sum([x['unique_operation_seconds'] for x in timings],0).tolist(),'batch_timings':timings,'trace_archives':trace_receipts,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('unified_cycle_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started);watchdog()
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('unified_cycle_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
