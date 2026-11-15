"""Fixed1024-input fallback windows versus cohort-end batching on fresh inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import platform
import time
import numpy as np
import study as S
import larger_relaxation as L
import selective_fallback as F
from fast_path_validator import Validator

DEST=S.OUT/'window_fallback'
PP=S.HERE/'window_fallback-protocol.json'
BASE=2026111507
WINDOW=1024
REPS=3


def freeze():
    p=json.loads(F.PP.read_text());pilots=[json.loads((DEST/f'prior-cohort-pilot-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in pilots)
    p.update(id='fixed1024-window-graph-informed-fallback',frozen_utc=S.utc(),execution_script_sha256=S.sha(__file__))
    p['cohort']=dict(p['cohort'],base_seed=BASE);p['initialization']=dict(p['initialization'],arms=['native','bounded1024','cohort_end'])
    p['primary']={'id':'H2u','hypothesis':'Fresh120grid loop20 bounded1024 graph-informed lambda1 fallback gains at least0.05pp joint success over native and has at most2% median measured answer overhead.','accuracy_target_pp':.05,'overhead_target_percent':2.,'unit':'Independent input;16384unique inputs,three fixed models.','accuracy_decision':'Separately one-sided95% net bounds from paired CP marginal tails alpha=.025. Support lower>=0.05pp; falsify upper<0.05pp; otherwise inconclusive.','runtime_decision':'Three full repetitions of each policy, all retained. Per bounded repetition sum both GPUs incremental validation/queue/setup/fallback/output seconds, divide by common native answer seconds. Support median<=2%; otherwise falsify. No timing CI inferred.','conjunction':'Support only when both pass; falsify when either falsifies; otherwise inconclusive. Fixed2% remains unchanged after pilot and fresh outcomes.'}
    p['decoder']=dict(p['decoder'],batching='Bounded policy flushes each model queue after exactly1024native inputs on its GPU, immediately after the16th64-input batch. Cohort control flushes at8192inputs/model/GPU. Return each native accepted answer immediately. Fixed lambda1 only for exact graph-feasibility rejects. No shortestness oracle.',controls='Native prefix shared exactly within each policy execution; both scheduling policies execute3complete timed passes with exact native-output and policy-specific fallback parity checks. No always-lambda full-cohort arm. Both policies use the same fixed lambda and native hidden couplings.')
    p['timing']={'repetitions':3,'policy_order':'Rep0 bounded then cohort; rep1 cohort then bounded; rep2 bounded then cohort. Fixed before fresh inputs.','native_scope':'Model setup, CPU-to-GPU input transfer, hidden settle/readout/convergence, native solve, CPU board/step/flag return. Native prefix serves paired control.','incremental_scope':'Exact compiled validator/conversion, queue index creation/gathers/metadata, queue concatenation and fallback initial state, lambda1 solve and CPU return. All incremental costs charged, including native batches with no reject.','latency':'For every rejected model-input retain raw enqueue/end times in perf_counter seconds relative to each independently executed policy pass; wait includes intervening evidence/audit-I/O work. Also retain component-summed decoder-clock enqueue/end times excluding evidence work, and source ordinal, native-completion count and flush count. Delay in native-input count is flush_count minus native_completion_count; source-ordinal delay also retained. Report quantiles/max for both clocks and count delays.','excluded_evidence_costs':'Archive loading, one warmup, full-logit diagnostic copying, hashing, serialization and logging are excluded from component decoder timings, included in actual wall latency and total worker wall. Each policy separately runs the entire native prefix so real wall delays do not contain the competing policy.','reuse':'Native arrays recorded once; subsequent native passes are required timing repeats with exact board/step/flag checks. Fallback arrays recorded once per policy. Timing repeats kept separate from scientific unique-input evidence.'}
    p['retained_evidence']='One full native output/logit/diagnostic archive per model-input; actual bounded and cohort-end rejected-subset outputs/logits/diagnostics; exact acceptance masks; every fallback batch and repeat hash; all per-case latency clocks/counts and complete timing components. Native accepted outputs reused in both decoder arms.'
    p['budget']={'total_gpu_minute_envelope':30,'per_worker_wall_seconds':900,'watchdog':'Check each64-input batch; preserve completed artifacts if exceeded; no protocol changes.'}
    p['window_choice']={'fixed_window':1024,'authority':'Coordinator chose1024before these prior-cohort timing outputs, retained regardless of pilot cost; no fresh-outcome tuning.','pilot_scope':'Explicit prior-H2t rejected-input development replays for256,1024and8192windows; excluded from new confirmation.','receipts':{str((DEST/f'prior-cohort-pilot-gpu{g}.json').relative_to(S.ROOT)):S.sha(DEST/f'prior-cohort-pilot-gpu{g}.json') for g in [0,1]}}
    p['implementation_sha256'].update({str((S.HERE/'selective_fallback.py').relative_to(S.ROOT)):S.sha(F.__file__)})
    p['confirmation']={'parent_protocol_sha256':S.sha(F.PP),'parent_result_sha256':S.sha(F.DEST/'audit.json'),'change':'Fresh cohort; fixed1024flush vs separately executed cohort-end control; actual per-case latency and throughput measured.'}
    if PP.exists():raise FileExistsError(PP)
    S.save(PP,p);S.emit('window_protocol_frozen',sha256=S.sha(PP),window=WINDOW,inputs=16384)


def make_one(index):return S.maze(120,.2,np.random.SeedSequence([BASE,120,20,index]))


def prepare():
    p=json.loads(PP.read_text());assert p['execution_script_sha256']==S.sha(__file__);datasets=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(64):
            path=DEST/f'shard{shard:02d}.npz'
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('window_data',shard=shard,inputs_done=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started})


def preflight(gpu):
    lock,inventory=F.acquire(gpu);torch,_,M=S.imports();model=S.model_for(0,120,'original','cuda');items=[S.maze(120,.2,np.random.SeedSequence([2026111597,i])) for i in range(64)];tokens=np.stack([x[0] for x in items]);t=torch.tensor(tokens.astype(np.int64),device='cuda')
    with torch.no_grad():
        psi,_=model.settle(t);a=model.couplings(t,psi);u=torch.full_like(a[:,0],3.);native=M.descend(model,t,a,steps=8192,eta=.5,margin=3.,certify_every=16);new=L.solve(model,t,a,u,8192,0.);assert all(torch.equal(native[i],new[i]) for i in range(3))
    pred=new[0].bool().cpu().numpy();assert np.array_equal(Validator()(tokens,pred),np.array([L.feasible(tok,b) for tok,b in zip(tokens,pred)]))
    S.save(DEST/'preflight.json',{'status':'passed','utc':S.utc(),'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'native_parity':True,'validator_parity':True,'reserved_inputs':64,'inventory':inventory});S.emit('window_preflight_passed')


def worker(gpu):
    lock,inventory=F.acquire(gpu);started=time.perf_counter();torch,_,_=S.imports();from cram.maze_learned import gather
    p=json.loads(PP.read_text());assert p['execution_script_sha256']==S.sha(__file__)
    for path,h in p['implementation_sha256'].items():assert S.sha(S.ROOT/path)==h
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h
    for m in p['models'].values():assert S.sha(S.ROOT/m['path'])==m['sha256']
    assert S.sha(F.LIBRARY)==p['validator_gate']['binary_sha256']
    pre=json.loads((DEST/'preflight.json').read_text());assert pre['status']=='passed' and pre['protocol_sha256']==S.sha(PP)
    manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);data={}
    for ds in manifest['datasets']:
        assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:data[ds['shard']]={k:x[k] for k in x.files}
    jobs=[j for j in p['jobs'] if j['gpu']==gpu];validator=Validator();state={'status':'running','pid':os.getpid(),'host':platform.node(),'gpu':gpu,'started_utc':S.utc(),'jobs_complete':[]};timing=[];native_refs={};fallback_refs={}
    warm=S.model_for(0,120,'original','cuda');t=torch.tensor(data[0]['tokens'][:64].astype(np.int64),device='cuda')
    with torch.no_grad():psi,_=warm.settle(t);a=warm.couplings(t,psi);L.solve(warm,t,a,torch.full_like(a[:,0],3.),32,0.);L.solve(warm,t,a,torch.full_like(a[:,0],3.),32,1.)
    del warm,t,a,psi
    common_keys=['model_setup','input_setup','hidden','native_answer','native_output'];extra_keys=['validator','queue','fallback_setup','fallback_answer','fallback_output']
    for rep in range(REPS):
        for policy in (['bounded1024','cohort_end'] if rep%2==0 else ['cohort_end','bounded1024']):
            pass_started=time.perf_counter();torch.cuda.synchronize();begin=time.perf_counter();models={s:S.model_for(s,120,'original','cuda') for s in range(3)};torch.cuda.synchronize();costs={k:0. for k in common_keys+extra_keys};costs['model_setup']=time.perf_counter()-begin
            queues={s:[] for s in range(3)};counts={s:0 for s in range(3)};latencies=[];fallback_count=0;batch_rows=[]
            def flush(seed):
                nonlocal fallback_count
                if not queues[seed]:return
                torch.cuda.synchronize();begin=time.perf_counter();parts=queues[seed];qt=torch.cat([x[0] for x in parts]);qa=torch.cat([x[1] for x in parts]);metadata=[m for x in parts for m in x[2]];window=(counts[seed]-1)//(WINDOW if policy=='bounded1024' else 8192);torch.cuda.synchronize();costs['fallback_setup']+=time.perf_counter()-begin;values=[];hashes=[]
                for lo in range(0,len(metadata),64):
                    torch.cuda.synchronize();begin=time.perf_counter();t=qt[lo:lo+64];a=qa[lo:lo+64];u=torch.full_like(a[:,0],3.);torch.cuda.synchronize();costs['fallback_setup']+=time.perf_counter()-begin
                    begin=time.perf_counter();output=L.solve(models[seed],t,a,u,8192,1.);torch.cuda.synchronize();costs['fallback_answer']+=time.perf_counter()-begin
                    begin=time.perf_counter();cpu=F.cpu_answer(output);torch.cuda.synchronize();costs['fallback_output']+=time.perf_counter()-begin;ended_wall=time.perf_counter()-pass_started;ended_decoder=sum(costs.values());hashes.append(F.digest(cpu))
                    for meta in metadata[lo:lo+64]:latencies.append(dict(meta,model_seed=seed,policy=policy,repetition=rep,flush_native_count=counts[seed],completed_wall_seconds=ended_wall,completed_decoder_seconds=ended_decoder,input_delay_from_native_completion=counts[seed]-meta['enqueue_native_count'],input_delay_from_source_ordinal=counts[seed]-meta['source_ordinal'],wall_delay_seconds=ended_wall-meta['enqueue_wall_seconds'],decoder_delay_seconds=ended_decoder-meta['enqueue_decoder_seconds']))
                    if rep==0:values.append(F.diagnostics(torch,models[seed],t,a,output,1.,cpu))
                key=(policy,seed,window)
                if rep==0:
                    arrays={k:np.concatenate([v[k] for v in values]) for k in values[0]};arrays['indices']=np.array([m['input_index'] for m in metadata]);arrays['hidden_good']=np.array([m['hidden_good'] for m in metadata]);path=DEST/f'fallback-{policy}-gpu{gpu}-seed{seed}-window{window:02d}.npz';np.savez_compressed(path,**arrays);fallback_refs[key]=hashes
                else:assert hashes==fallback_refs[key],(rep,key)
                fallback_count+=len(metadata);batch_rows.append({'seed':seed,'window':window,'n':len(metadata),'flush_native_count':counts[seed],'board_step_flag_hashes':hashes,'repeat_parity':True});queues[seed]=[]
            for job in jobs:
                seed=job['seed'];model=models[seed];d=data[job['shard']];tokens=d['tokens'];record_native=rep==0 and policy=='bounded1024';saved=[];accepted=[];native_hash=hashlib.sha256();run=DEST/f"runs/{job['id']}"
                state.update(repetition=rep,policy=policy,current_job=job,completed_inputs=0,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state)
                for lo in range(0,256,64):
                    token=tokens[lo:lo+64];torch.cuda.synchronize();begin=time.perf_counter();t=torch.tensor(token.astype(np.int64),device='cuda');torch.cuda.synchronize();costs['input_setup']+=time.perf_counter()-begin
                    with torch.no_grad():
                        torch.cuda.synchronize();begin=time.perf_counter();psi,hs=model.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1);hg=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1));a=model.couplings(t,psi);u=torch.full_like(a[:,0],3.);torch.cuda.synchronize();costs['hidden']+=time.perf_counter()-begin
                        begin=time.perf_counter();output=L.solve(model,t,a,u,8192,0.);torch.cuda.synchronize();costs['native_answer']+=time.perf_counter()-begin
                        begin=time.perf_counter();cpu=F.cpu_answer(output);torch.cuda.synchronize();costs['native_output']+=time.perf_counter()-begin;enqueue_wall=time.perf_counter()-pass_started;enqueue_decoder=sum(costs.values())
                        begin=time.perf_counter();good=validator(token,cpu['predictions']);hidden_cpu=hg.cpu().numpy();good&=cpu['finite']&cpu['stopped']&hidden_cpu;reject=np.flatnonzero(~good);costs['validator']+=time.perf_counter()-begin
                        torch.cuda.synchronize();begin=time.perf_counter()
                        if len(reject):
                            idx=torch.tensor(reject,device='cuda');metadata=[{'input_index':int(d['indices'][lo+i]),'source_ordinal':counts[seed]+int(i)+1,'enqueue_native_count':counts[seed]+64,'enqueue_wall_seconds':enqueue_wall,'enqueue_decoder_seconds':enqueue_decoder,'hidden_good':bool(hidden_cpu[i])} for i in reject];queues[seed].append((t[idx],a[idx],metadata))
                        torch.cuda.synchronize();costs['queue']+=time.perf_counter()-begin;counts[seed]+=64;native_hash.update(F.digest(cpu).encode())
                        if record_native:
                            F.diagnostics(torch,model,t,a,output,0.,cpu);saved.append(cpu|{'hidden_good':hidden_cpu,'hidden_steps':hs.cpu().numpy()});accepted.append(good)
                    if policy=='bounded1024' and counts[seed]%WINDOW==0:flush(seed)
                    state.update(completed_inputs=lo+64,updated_utc=S.utc(),worker_elapsed_s=time.perf_counter()-started);S.save(DEST/f'worker{gpu}.json',state)
                    if time.perf_counter()-started>900:raise TimeoutError('Frozen900s budget reached; completed artifacts retained')
                if record_native:
                    run.mkdir(parents=True,exist_ok=True)
                    if (run/'result.json').exists():raise FileExistsError(run)
                    arrays={k:np.concatenate([x[k] for x in saved]) for k in saved[0]};arrays['native_accepted']=np.concatenate(accepted);np.savez_compressed(run/'outputs.npz',**arrays);native_refs[job['id']]=native_hash.hexdigest();S.save(run/'result.json',{'status':'complete','job':job,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'outputs_sha256':S.sha(run/'outputs.npz'),'dataset_sha256':manifest['datasets'][job['shard']]['sha256'],'checkpoint_sha256':p['models'][str(seed)]['sha256'],'native_board_step_flag_digest':native_refs[job['id']]})
                else:assert native_hash.hexdigest()==native_refs[job['id']],(rep,policy,job)
                state['jobs_complete'].append(f"rep{rep}/{policy}/{job['id']}");S.emit('window_job_complete',gpu=gpu,repetition=rep,policy=policy,job=job['id'])
            if policy=='cohort_end':
                for seed in range(3):flush(seed)
            assert all(counts[s]==8192 and not queues[s] for s in range(3));common=sum(costs[k] for k in common_keys);extra=sum(costs[k] for k in extra_keys)
            timing.append({'repetition':rep,'policy':policy,'cost_seconds':costs,'native_common_seconds':common,'incremental_seconds':extra,'answer_overhead_percent':100*extra/costs['native_answer'],'end_to_end_overhead_percent':100*extra/common,'fallback_model_inputs':fallback_count,'exact_native_repeat_parity':True,'fallback_batches':batch_rows,'pass_wall_seconds':time.perf_counter()-pass_started})
            S.save(DEST/f'latencies-gpu{gpu}-rep{rep}-{policy}.json',{'protocol_sha256':S.sha(PP),'rows':latencies});S.save(DEST/f'timing-gpu{gpu}.json',{'status':'complete' if rep==2 and policy=='cohort_end' else 'partial','protocol_sha256':S.sha(PP),'gpu':gpu,'inventory':inventory,'repetitions':timing});S.emit('window_pass_complete',gpu=gpu,repetition=rep,policy=policy,answer_overhead_percent=timing[-1]['answer_overhead_percent'],cases=fallback_count)
    state.update(status='complete',completed_utc=S.utc(),worker_elapsed_s=time.perf_counter()-started);S.save(DEST/f'worker{gpu}.json',state);S.emit('window_worker_complete',gpu=gpu,seconds=state['worker_elapsed_s'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','preflight','worker']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='prepare':prepare()
    elif args.command=='preflight':preflight(args.gpu)
    else:worker(args.gpu)
