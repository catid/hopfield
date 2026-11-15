"""H2av fresh4096 size180 fixed-policy cycle-cut transfer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import cycle_transfer_batch as Q
import fast_npz as Z
from factorial_transfer import hidden

PP=S.HERE/'cycle-transfer-180-protocol.json'
DEST=S.OUT/'cycle_transfer_180'
BASE=2026111521
ARMS=Q.ARMS
AXIS0={'hidden_steps','hidden_good','couplings'}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=json.loads((S.HERE/'safe-block-transfer-protocol.json').read_text());models={}
    for seed in range(3):
        old=parent['models'][str(seed)];models[str(seed)]={arm:old['pl' if arm=='pl' else 'old_binary'] for arm in ARMS}
    gate_path=S.OUT/'cycle_transfer_180_preflight/numerical-gate.json';gate=json.loads(gate_path.read_text());registry_path=DEST/'seed-registry.json';registry=json.loads(registry_path.read_text());assert gate['status']=='passed' and gate['batch_helper_sha256']==S.sha(S.HERE/'cycle_transfer_batch.py') and gate['script_sha256']==S.sha(S.HERE/'preflight_cycle_transfer_180.py');assert registry['status']=='passed' and registry['proposed_seed_base']==BASE and registry['fresh_inputs_drawn']==0
    names=['cycle_transfer_180.py','cycle_transfer_batch.py','preflight_cycle_transfer_180.py','audit_cycle180_seed_registry.py','audit_cycle_transfer180_inputs.py','connectivity_transfer_180.py','connected_block_moves.py','cycle_cut_moves.py','lexical_cycle_cut_moves.py','graph_block_moves.py','binary_coordinate.py','larger_relaxation.py','factorial_transfer.py','fast_npz.py','study.py','audit.py','safe_block_transfer.py']
    p={'id':'fresh4096-cycle-cut-transfer180','hypothesis_id':'H2av','status':'frozen_after_complete_preflight_before_fresh_draws','frozen_utc':S.utc(),'arms':list(ARMS),'models':models,'jobs':[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':(3*shard+seed)%2} for shard in range(16) for seed in range(3)],'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names},'parent_protocol_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['safe-block-transfer-protocol.json','cycle-cut-rescue-checked-protocol.json','lexical-cycle-cut-protocol.json','cycle-transfer-protocol.json','connectivity-transfer-180-protocol.json']},'source_gate_files':{str(gate_path.relative_to(S.ROOT)):S.sha(gate_path),str(registry_path.relative_to(S.ROOT)):S.sha(registry_path)},'cohort':{'n_unique':4096,'length':180,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze SeedSequence([2026111521,180,20,index]),index0..4095','shards':16,'shard_size':256,'prior_registry':registry['prior_data_manifests'],'exclusion':'Exacttokendigest uniquenessandnonoverlap againstall19registeredpriormanifests;allshortestlabels independentlycheckedbeforeruns.'},'primary_endpoints':{'lexical_gain_vs_veto':{'candidate':'lexical_cycle_cut','control':'connectivity_veto','minimum_net_gain_pp':.5},'lexical_noninferiority_vs_bp':{'candidate':'lexical_cycle_cut','control':'pl','minimum_net_gain_pp':-.1}},'paired_inference':'Independentunit=maze;successrequiresall3fixedmodels. LetG=candidatesuccess/controlfailure,L=controlsuc/candidatefail. ComputeCPmarginal boundsforG/n,L/n withone-sidedalpha.025each. Separatelyone-sided95netlower=100*(G_lower-L_upper);netupper=100*(G_upper-L_lower). Foreachdeclaredthreshold,supportiflower>=threshold,falsifyifupper<threshold,otherwiseinconclusive. Twoprimaryendpoints reportedseparately;nosimultaneous95claim. NImeansnetaccuracydifference>=-.1pp,notstandalonelossprobability.', 'secondary':'Energy-vs-lexicalpairedgains/lossesdescriptive;no learnedrankingbenefitclaim. BPfirst. Allcaps,nonfinite,hiddenfailuresandnativecyclerestoration retained. Initialcorrectsimplepathidentity proof distinctfrom measuredfreshgain.', 'inference':'FixedH2aeLPandcontinuedBPfits. BPnativefloat32u3eta.5cap8192check16. Sharedfloat32hidden tol1e-6cap4000check8. LPfloat64fourcolor threshold-1e-8cap512. Oneexactbinaryprefixandoneexactconnectivity-vetoprefix shared. Vetooriginalnegativeenergychainmoves. Cyclepolicies: initialsimplepathidentity; oldchaincomponents2..256,endpointconnectivityandstrict E-V+component_count reduction,acceptminimumoriginaldelta versuslexicographicminimum evenpositive;unchangednativeaftereachcut,max16rounds. Explicitgraphprior,energyincreasesallowed; nolabel/shortestdistance choice.', 'gate':'Completeactual180objectivepreflightbeforefreeze: first64alreadyseenH2aminputs/model, exactretainedBP/binary/vetoboards/iterations/flags/readouts, upstreamnativeanduncachedbinaryparity, independentoriginalenergies/flips/continuousgradients/readouts, allfivepolicyclamps/prefixes/costs. Allfourouteredgesandinternalwalls. H2alpaddinginvariancefailurepreserved;nosizeinvarianceclaim. BothselectorpoliciesandpositiveacceptancealreadygatedbyH2au.', 'budget':{'inference_total_gpu_minutes':10,'per_worker_wall_seconds':300,'source_gate_wall_seconds':120,'input_cpu_audit_wall_seconds':120,'input_cpu_audit_max_processes':8,'full_cpu_audit_wall_seconds':300,'full_cpu_audit_max_processes':16,'statistics_wall_seconds':120,'enforcement':'Perbatch/blockroundwatchdogandexternalwalltimeouts;retaincompletedarchives,noextensions.'},'archive':'UniformexclusiveNPZdeflate1,allstates/traces/counters;fullbinary+vetoprefixchargedtoeachcyclearm butexecutedonce. Fullcostscoringretainedinlexicalforaudit;notoptimizedlexicalruntimeclaim.', 'accounting':{'fresh_inputs':4096,'primary_model_input_arm_evaluations':61440,'reused_binary_prefixes':12288,'reused_full_veto_prefixes':24576,'new_optimizer_updates':0}}
    p['failed_padding_lineage']=json.loads((S.HERE/'connectivity-transfer-180-protocol.json').read_text())['failed_H2al'];p['same_fixed_policy_source_protocol_sha256']=S.sha(S.HERE/'cycle-transfer-protocol.json')
    S.save(PP,p);S.save(DEST/'protocol.json',p);gate.update(protocol_sha256=S.sha(PP),source_preflight_sha256=S.sha(gate_path),new_gate_runs=0);S.save(DEST/'numerical-gate.json',gate);S.emit('cycle_transfer_frozen',sha256=S.sha(PP))


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


def make_one(index):return S.maze(180,.2,np.random.SeedSequence([BASE,180,20,index]))


def prepare():
    protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP);started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz';indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));Z.save(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('cycle_transfer_data',shard=shard,inputs=(shard+1)*256)
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
        keys=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path','hidden_steps','hidden_good','couplings'];saved={k:[] for k in keys};timings=[];hidden_seconds=diagnostic_seconds=archive_seconds=0.;job_start=time.perf_counter();seed=job['seed'];trace_receipts=[]
        with torch.no_grad():
            for lo in range(0,256,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');torch.cuda.synchronize();before=time.perf_counter();a,hs,good=hidden(models[seed][0],t);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                result=Q.solve(models[seed],doubles[seed],t,a,rotation=(ji+lo//32)%5,watchdog=watchdog);out=result['outputs'];prefix=result['prefixes'];block=result['blocks'];timings.append(result['timing']);before=time.perf_counter()
                batch={k:np.stack([out[v][name].cpu().numpy() for v in range(5)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                for v in range(5):
                    m=doubles[seed][Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(m.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(m.energy(t,board,a.double()).cpu().numpy())
                batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[seed][0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1]]),block_moves=np.stack([block[v]['block_moves'] for v in [2,3,4]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [2,3,4]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [2,3,4]]),block_cap=np.stack([block[v]['block_cap'] for v in [2,3,4]]),initial_simple_path=np.stack([block[v]['initial_simple_path'] for v in [2,3,4]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                before=time.perf_counter();bp=run/f'batch{lo//32:02d}.npz';Z.save(bp,**batch)
                for v in [2,3,4]:
                    raw=block[v]['raw'];path=run/f'batch{lo//32:02d}-{ARMS[v]}-trace.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'arm':ARMS[v],'trace':raw['trace'] if raw else []};arrays={'trace_utf8':np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8),'valid_prefix_indices':block[v]['valid_prefix_indices'],'states_before':raw['states_before'] if raw else np.zeros((0,0,180,180),bool)};Z.save(path,**arrays);trace_receipts.append({'offset':lo,'arm':ARMS[v],'path':path.name,'sha256':S.sha(path)})
                archive_seconds+=time.perf_counter()-before
                for k in keys:saved[k].append(batch[k])
                state.update(current_job=job,completed_inputs=lo+32,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);watchdog()
        before=time.perf_counter();arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};Z.save(run/'outputs.npz',**arrays);archive_seconds+=time.perf_counter()-before
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':np.sum([x['charged_answer_seconds'] for x in timings],0).tolist(),'unique_answer_seconds_by_operation':np.sum([x['unique_operation_seconds'] for x in timings],0).tolist(),'batch_timings':timings,'trace_archives':trace_receipts,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('cycle_transfer_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started);watchdog()
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('cycle_transfer_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
