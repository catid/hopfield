"""H2al fresh180x180 transfer of fixed explicitly symbolic block policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import connectivity_batch as Q
import fast_npz as Z
from factorial_transfer import hidden

PP=S.HERE/'connectivity-transfer-protocol.json'
DEST=S.OUT/'connectivity_transfer'
BASE=2026111517
ARMS=Q.ARMS
AXIS0={'hidden_steps','hidden_good','couplings','initial_simple_path'}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    previous=json.loads((S.HERE/'factorial-transfer-protocol.json').read_text());screen=json.loads((S.OUT/'connectivity_screen/audit.json').read_text());assert screen['status']=='passed';models={}
    for seed in range(3):
        x=previous['models'][str(seed)];models[str(seed)]={'pl':x['pl'],'old_binary':x['old_binary'],'skip_simple_path':x['old_binary'],'connectivity_veto':x['old_binary']}
    names=['connectivity_transfer.py','connectivity_batch.py','gate_connectivity_transfer.py','connected_block_moves.py','graph_block_moves.py','binary_coordinate.py','larger_relaxation.py','factorial_transfer.py','fast_npz.py','study.py','audit.py']
    p={'id':'fresh180-loop20-symbolic-block-transfer','hypothesis_id':'H2al','frozen_utc':S.utc(),'status':'frozen_before_sizegate_and_freshgeneration','arms':list(ARMS),'models':models,'jobs':previous['jobs'],'parent_protocol_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['factorial-transfer-protocol.json','connectivity-screen-protocol.json']},'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names},'screen_audit_sha256':S.sha(S.OUT/'connectivity_screen/audit.json'),'cohort':{'n_unique':4096,'length':180,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze SeedSequence([2026111517,180,20,index]), index0..4095','shards':16,'shard_size':256,'prior_exclusions':['native_relaxation','larger_relaxation','selective_fallback','window_fallback','contrastive_training','lp_transfer','lp_relaxation','binary_coordinate','hard_negative_transfer','second_negative_transfer','factorial_transfer'],'rationale':'Gridlength1.5timesH2aj, same looprate, fixedweights/caps/policies; test spatialscale transfer rather than repeatthe120distribution.'},'primary_endpoints':{'veto_gain':{'candidate':'connectivity_veto','control':'old_binary','target_gain_pp':1.},'veto_vs_bp':{'candidate':'connectivity_veto','control':'pl','noninferiority_loss_margin_pp':.1}},'statistics':'Separately decided endpoints, nooverallconjunction/simultaneous95claim. Gain net one-sided95bounds from CPmarginalalpha.025: supportlower>=1pp,falsifyupper<1pp. NI pairedBPsuccess/vetofailure lossCPone-sided95alpha.05: supportupper<=.1pp,falsifylower>.1pp. Otherwiseinconclusive.4096independentmazeunits; all3modeljointsuccess. Skipcomparisons/descriptivezero-loss separately.','policy':'FrozenH2ak. BPnativefloat32u3eta.5cap8192check16. LPbinaryfloat64fourcolor threshold-1e-8cap512. Blocksdegree1/2components2..256cells,max16rounds,fullenergyminimumdelta then unchangedbinarydescent. Skip returnsinitial answer exactly if selectedcellsformone simpleendpointpath, elseoriginalblocks. Veto admits deletionsonly ifselectedendpointsremainconnected before native descent. Explicit symbolicgraphknowledge; no targetlabels/shortestdistance in selection.','proof_scope':'Skip cannotregress an initiallycorrectsimplepath because itreturns thatsameanswer/flags. This is a conditional identityproof, not measured statisticalsupport. Veto zero losses remain empirical; auditnativedisconnections.','hidden':'Sharedimmutablefloat32readout tol1e-6cap4000check8, every nonanswer tensor identical. Exact retained binaryprefix reused bybothpolicies; failedprefixesretained, norescue.','sizegate':'Beforefreshgeneration: first64alreadyseenH2aj120inputs/model wallpaddedto180; verify unchangedinterior hidden/readouts/native/binaryprefix,clamps andliteralcosts. No newmazeinputs or labelselection.','budget':{'inference_total_gpu_minutes':10,'per_worker_wall_seconds':300,'source_size_gate_wall_seconds':120,'input_cpu_audit_wall_seconds':120,'input_cpu_audit_max_processes':8,'full_cpu_audit_wall_seconds':300,'full_cpu_audit_max_processes':16,'full_cpu_audit_aggregate_process_wall_minutes':80,'statistics_wall_seconds':120,'enforcement':'Inferencewatchdogs eachbatch/blockround; auditexternal walltimeouts plus retained partialreceipts. Allcaps/failures retained; noextensions.'},'archive':'Uniformexclusive NumPyZIP_DEFLATEDlevel1,fulltraces/states/counters andtimings. Charge completeprefix+increment eachpolicy; uniqueexecutedworkseparate.','accounting':{'fresh_inputs':4096,'primary_model_input_arm_evaluations':49152,'reused_binary_prefixes':24576,'new_optimizer_updates':0},'source_gate_files':{}}
    for n in ['shard00.npz']+[f'runs/shard00-s{s}/outputs.npz' for s in range(3)]:
        path=S.OUT/'factorial_transfer'/n;p['source_gate_files'][str(path.relative_to(S.ROOT))]=S.sha(path)
    S.save(PP,p);S.emit('connectivity_transfer_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in {**p['source_sha256'],**p['parent_protocol_sha256'],**p['source_gate_files']}.items():assert S.sha(S.ROOT/path)==h,path
    for group in p['models'].values():
        for item in group.values():assert S.sha(S.ROOT/item['path'])==item['sha256']
    assert S.sha(S.OUT/'connectivity_screen/audit.json')==p['screen_audit_sha256']
    return p


def load_models(p,seed,device='cuda'):
    torch,_,_=S.imports();models=[]
    for arm in ARMS[:2]:
        model=S.model_for(seed,180,'original',device);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location=device,weights_only=True));models.append(model)
    for k,v in models[0].state_dict().items():
        if k not in ['T','b','kappa']:assert torch.equal(v,models[1].state_dict()[k]),k
    return models,[copy.deepcopy(m).double() for m in models]


def make_one(index):return S.maze(180,.2,np.random.SeedSequence([BASE,180,20,index]))


def prepare():
    protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP);started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz';indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));Z.save(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('connectivity_transfer_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started,'gate_sha256':S.sha(DEST/'numerical-gate.json')})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);inp=json.loads((DEST/'independent-input-audit.json').read_text());assert inp['status']=='passed' and inp['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={};doubles={}
    def watchdog():
        if time.perf_counter()-started>300:
            state.update(status='budget_exhausted',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);raise TimeoutError('Frozen300sworkerbudget; completed batch files retained')
    for seed in range(3):models[seed],doubles[seed]=load_models(p,seed)
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        watchdog();run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens']
        keys=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','initial_simple_path','block_moves','post_block_sweeps','block_native_cap','block_cap','hidden_steps','hidden_good','couplings'];saved={k:[] for k in keys};timings=[];hidden_seconds=diagnostic_seconds=archive_seconds=0.;job_start=time.perf_counter();seed=job['seed'];trace_receipts=[]
        with torch.no_grad():
            for lo in range(0,256,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');torch.cuda.synchronize();before=time.perf_counter();a,hs,good=hidden(models[seed][0],t);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                result=Q.solve(models[seed],doubles[seed],t,a,rotation=(ji+lo//32)%4,watchdog=watchdog);out=result['outputs'];prefix=result['prefixes'];block=result['blocks'];timings.append(result['timing']);before=time.perf_counter()
                batch={k:np.stack([out[v][name].cpu().numpy() for v in range(4)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                for v in range(4):
                    m=doubles[seed][Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(m.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(m.energy(t,board,a.double()).cpu().numpy())
                batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[seed][0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1]]),initial_simple_path=block[2]['initial_simple_path'],block_moves=np.stack([block[v]['block_moves'] for v in [2,3]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [2,3]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [2,3]]),block_cap=np.stack([block[v]['block_cap'] for v in [2,3]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                before=time.perf_counter();bp=run/f'batch{lo//32:02d}.npz';Z.save(bp,**batch)
                for v in [2,3]:
                    raw=block[v]['raw'];path=run/f'batch{lo//32:02d}-{ARMS[v]}-trace.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'arm':ARMS[v],'trace':raw['trace'] if raw else []};arrays={'trace_utf8':np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8),'valid_prefix_indices':block[v]['valid_prefix_indices'],'states_before':raw['states_before'] if raw else np.zeros((0,0,180,180),bool)};Z.save(path,**arrays);trace_receipts.append({'offset':lo,'arm':ARMS[v],'path':path.name,'sha256':S.sha(path)})
                archive_seconds+=time.perf_counter()-before
                for k in keys:saved[k].append(batch[k])
                state.update(current_job=job,completed_inputs=lo+32,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);watchdog()
        before=time.perf_counter();arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};Z.save(run/'outputs.npz',**arrays);archive_seconds+=time.perf_counter()-before
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':np.sum([x['charged_answer_seconds'] for x in timings],0).tolist(),'unique_answer_seconds_by_operation':np.sum([x['unique_operation_seconds'] for x in timings],0).tolist(),'batch_timings':timings,'trace_archives':trace_receipts,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('connectivity_transfer_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started);watchdog()
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('connectivity_transfer_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
