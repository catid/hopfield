"""H2aj prospective source-fit by graph-block factorial on independent mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import factorial_batch as Q
import fast_npz as Z

PP=S.HERE/'factorial-transfer-protocol.json'
DEST=S.OUT/'factorial_transfer'
BASE=2026111516
ARMS=Q.ARMS
AXIS0={'hidden_steps','hidden_good','couplings'}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    old=json.loads((S.HERE/'second-negative-transfer-protocol.json').read_text());new=json.loads((S.HERE/'leaf-decoder-gate-protocol.json').read_text());models={}
    for seed in range(3):
        a=old['models'][str(seed)];b=new['models'][str(seed)];models[str(seed)]={'pl':a['pl'],'old_binary':a['second_negative_binary'],'new_binary':b,'old_blocks':a['second_negative_binary'],'new_blocks':b}
    source={str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['factorial_transfer.py','factorial_batch.py','gate_factorial_transfer.py','fast_npz.py','benchmark_archive_levels.py','graph_block_moves.py','binary_coordinate.py','larger_relaxation.py','study.py','audit.py']}
    parents={str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['second-negative-transfer-protocol.json','leaf-decoder-gate-protocol.json','graph-block-source-protocol.json']}
    source_gate={}
    for name in ['shard00.npz']+[f'runs/shard00-s{s}/outputs.npz' for s in range(3)]:
        path=S.OUT/'second_negative_transfer'/name;source_gate[str(path.relative_to(S.ROOT))]=S.sha(path)
    benchmark=S.OUT/'archive_benchmark/result.json'
    p={'id':'leaf-fit-by-label-free-block-fresh-factorial','hypothesis_id':'H2aj','status':'frozen_before_numerical_gate_and_fresh_generation','frozen_utc':S.utc(),'arms':list(ARMS),'models':models,'parent_protocol_sha256':parents,'source_sha256':source,'source_gate_files':source_gate,'archive_benchmark':{'path':str(benchmark.relative_to(S.ROOT)),'sha256':S.sha(benchmark)},'jobs':old['jobs'],'cohort':{'n_unique':4096,'length':120,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze SeedSequence([2026111516,120,20,index]), index0..4095','shards':16,'shard_size':256,'disjointness':'Exact token digest versus all earlier cohorts, including H2ae.'},'primary_endpoints':{'blocks_on_old':{'candidate':'old_blocks','control':'old_binary','target_gain_pp':1.},'training_without_blocks':{'candidate':'new_binary','control':'old_binary','target_gain_pp':1.},'combined_vs_bp':{'candidate':'new_blocks','control':'pl','noninferiority_loss_margin_pp':.1}},'statistics':'Each endpoint decided separately, no overall conjunction or simultaneous95claim. Gain endpoints: separately one-sided95 net bounds from CP marginalalpha.025; supportlower>=1pp, falsifyupper<1pp. NI: paired BPsuccess/candidatefailure CP one-sided95alpha.05; supportupper<=.1pp, falsifylower>.1pp. Otherwiseinconclusive. Joint-three-model success unit is independentmaze, n4096. Interaction descriptive.','inference':{'hidden':'One identical frozenfloat32 readout/model-input: tol1e-6 cap4000 check8. Assert every nonanswer state tensor identical.','bp':'BP AdamW continuedPL nativefloat32,u3,eta.5,cap8192,check16.','binary':'Frozen H2aa float64 four-color decoder, threshold-1e-8 cap512.','blocks':'Frozen H2ai helper, selectedfree degree1/2 components size2..256, fulloriginalenergy deletion<−1e−8, minimumdelta/lexicographic ties, max16rounds, unchanged nativebinary descent. No labels or validity oracle.','reuse':'Each block arm reuses its exact binary prefix. Failed prefixes remainfailed with no rescue. Block cost charges fullprefix plus incremental. Unique executedwork separate.','certificate':'Original float64 minimumflip>=−1e−6, finite, stopped, hiddenconverged, independent graphvalid shortest.'},'gate':'Before freshgeneration: deterministic first64 previouslyseen H2ae120x120inputs/model, allfivearms; exact retained readout/prefix parity, clamps, allcandidate literal fullenergy deltas, originalfinal certificates. Fixedrandom clamped boards additionally exercise moves. No labels read for policyselection.','budget':{'total_gpu_minute_envelope':20,'per_worker_wall_seconds':600,'source_gate_wall_seconds':180,'watchdog':'Eachbatch and blockround; preserve completedbatch archives and partialstate onexpiry; noextension. Gate excluded from freshworker envelope, bounded separately. Includes CPUcandidateconstruction, scoring, descent, diagnostics, archives and workerwall.'},'archive':'Uniform NumPy ZIP_DEFLATEDlevel1, allow_pickleFalse, exclusivecreate. Exactdtype/shape/bytes parity against retained sourcearchive; full traces and states retained.','accounting':{'primary_model_input_arm_evaluations':61440,'independent_inputs':4096,'reused_binary_prefixes':24576,'new_optimizer_updates':0}}
    S.save(PP,p);S.emit('factorial_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in {**p['source_sha256'],**p['parent_protocol_sha256'],**p['source_gate_files']}.items():assert S.sha(S.ROOT/path)==h,path
    for group in p['models'].values():
        for item in group.values():assert S.sha(S.ROOT/item['path'])==item['sha256']
    assert S.sha(S.ROOT/p['archive_benchmark']['path'])==p['archive_benchmark']['sha256']
    return p


def load_models(p,seed,device='cuda'):
    torch,_,_=S.imports();models=[]
    for arm in ARMS[:3]:
        model=S.model_for(seed,120,'original',device);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location=device,weights_only=True));models.append(model)
    for m in models[1:]:
        for k,v in models[0].state_dict().items():
            if k not in ['T','b','kappa']:assert torch.equal(v,m.state_dict()[k]),k
    return models,[copy.deepcopy(m).double() for m in models]


def hidden(base,t):
    torch,_,_=S.imports();from cram.maze_learned import gather
    psi,hs=base.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=base.parts(t,base.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,base.zoffs,math.inf),lc[:,None]],1),1);good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
    return base.couplings(t,psi),hs,good


def make_one(index):return S.maze(120,.2,np.random.SeedSequence([BASE,120,20,index]))


def prepare():
    protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP);started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz';indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));Z.save(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('factorial_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started,'gate_sha256':S.sha(DEST/'numerical-gate.json')})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);inp=json.loads((DEST/'independent-input-audit.json').read_text());assert inp['status']=='passed' and inp['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={};doubles={}
    def watchdog():
        if time.perf_counter()-started>600:
            state.update(status='budget_exhausted',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);raise TimeoutError('Frozen600sworkerbudget; completed batch files retained')
    for seed in range(3):models[seed],doubles[seed]=load_models(p,seed)
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        watchdog();run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens']
        keys=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','block_moves','post_block_sweeps','block_native_cap','block_cap','hidden_steps','hidden_good','couplings'];saved={k:[] for k in keys};timings=[];hidden_seconds=diagnostic_seconds=archive_seconds=0.;job_start=time.perf_counter();seed=job['seed'];trace_receipts=[]
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');torch.cuda.synchronize();before=time.perf_counter();a,hs,good=hidden(models[seed][0],t);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                result=Q.solve(models[seed],doubles[seed],t,a,rotation=(ji+lo//64)%5,watchdog=watchdog);out=result['outputs'];prefix=result['prefixes'];block=result['blocks'];timings.append(result['timing']);before=time.perf_counter()
                batch={k:np.stack([out[v][name].cpu().numpy() for v in range(5)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                for v in range(5):
                    m=doubles[seed][Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(m.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(m.energy(t,board,a.double()).cpu().numpy())
                batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[seed][0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1,2]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1,2]]),block_moves=np.stack([block[v]['block_moves'] for v in [3,4]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [3,4]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [3,4]]),block_cap=np.stack([block[v]['block_cap'] for v in [3,4]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                before=time.perf_counter();bp=run/f'batch{lo//64:02d}.npz';Z.save(bp,**batch)
                for v in [3,4]:
                    raw=block[v]['raw'];path=run/f'batch{lo//64:02d}-{ARMS[v]}-trace.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'arm':ARMS[v],'trace':raw['trace'] if raw else []};arrays={'trace_utf8':np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8),'valid_prefix_indices':block[v]['valid_prefix_indices'],'states_before':raw['states_before'] if raw else np.zeros((0,0,120,120),bool)};Z.save(path,**arrays);trace_receipts.append({'offset':lo,'arm':ARMS[v],'path':path.name,'sha256':S.sha(path)})
                archive_seconds+=time.perf_counter()-before
                for k in keys:saved[k].append(batch[k])
                state.update(current_job=job,completed_inputs=lo+64,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);watchdog()
        before=time.perf_counter();arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};Z.save(run/'outputs.npz',**arrays);archive_seconds+=time.perf_counter()-before
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':np.sum([x['charged_answer_seconds'] for x in timings],0).tolist(),'unique_answer_seconds_by_operation':np.sum([x['unique_operation_seconds'] for x in timings],0).tolist(),'batch_timings':timings,'trace_archives':trace_receipts,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('factorial_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started);watchdog()
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('factorial_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
