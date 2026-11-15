"""H2ay fresh4096 size180 loop50 density transfer of the fixed cap composition."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import cap_transfer_batch as Q
import fast_npz as Z
from factorial_transfer import hidden

PP=S.HERE/'density-transfer-protocol.json'
DEST=S.OUT/'density_transfer'
BASE=2026111523
ARMS=Q.ARMS
AXIS0={'hidden_steps','hidden_good','couplings'}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=json.loads((S.HERE/'cap-transfer-protocol.json').read_text());models={}
    for seed in range(3):models[str(seed)]={arm:parent['models'][str(seed)]['pl' if arm=='pl' else 'old_binary'] for arm in ARMS}
    gate_path=S.OUT/'cap_transfer_preflight/numerical-gate.json';gate=json.loads(gate_path.read_text());registry_path=DEST/'seed-registry.json';registry=json.loads(registry_path.read_text());assert gate['status']=='passed' and gate['batch_helper_sha256']==S.sha(S.HERE/'cap_transfer_batch.py') and gate['script_sha256']==S.sha(S.HERE/'preflight_cap_transfer.py');assert gate['all_frozen_nonsimple_starts']==84;assert registry['status']=='passed' and registry['proposed_seed_base']==BASE and registry['fresh_inputs_drawn']==0
    density_gate_path=S.OUT/'density_transfer_preflight/numerical-gate.json';density_gate=json.loads(density_gate_path.read_text());assert density_gate['status']=='passed' and density_gate['driver_sha256']==S.sha(S.HERE/'density_transfer.py') and density_gate['script_sha256']==S.sha(S.HERE/'preflight_density_transfer.py') and density_gate['new_study_inputs']==0
    names=['density_transfer.py','preflight_density_transfer.py','cap_transfer_batch.py','preflight_cap_transfer.py','audit_density_seed_registry.py','audit_density_transfer_inputs.py','cycle_transfer_180.py','large_cycle_cut_moves.py','verify_large_cycle_geometry.py','connected_block_moves.py','cycle_cut_moves.py','lexical_cycle_cut_moves.py','graph_block_moves.py','binary_coordinate.py','larger_relaxation.py','factorial_transfer.py','fast_npz.py','study.py','audit.py','safe_block_transfer.py']
    p={'id':'fresh4096-density50-cap-composition180','hypothesis_id':'H2ay','status':'frozen_after_complete_source_preflight_before_fresh_draws','frozen_utc':S.utc(),'arms':list(ARMS),'models':models,'jobs':[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':(3*shard+seed)%2} for shard in range(16) for seed in range(3)],'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names},'parent_protocol_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['cap-transfer-protocol.json','cycle-transfer-180-protocol.json','large-cycle-cut-protocol.json']},'source_gate_files':{str(gate_path.relative_to(S.ROOT)):S.sha(gate_path),str(registry_path.relative_to(S.ROOT)):S.sha(registry_path),str(density_gate_path.relative_to(S.ROOT)):S.sha(density_gate_path)},'cohort':{'n_unique':4096,'length':180,'loop_rate':.5,'base_seed':BASE,'generator':'study.maze SeedSequence([2026111523,180,50,index]), index0..4095','shards':16,'shard_size':256,'prior_registry':registry['prior_data_manifests'],'exclusion':'Exact token-digest uniqueness and nonoverlap against all registered prior Work dataset manifests; every shortest label independently checked before inference.'},'primary_endpoints':{'lexical512_gain_vs_lexical256':{'kind':'paired_net','candidate':'lexical512_from_lexical256','control':'lexical256','minimum_net_gain_pp':.2},'lexical512_regression_probability':{'kind':'paired_loss_probability','candidate':'lexical512_from_lexical256','control':'lexical256','maximum_regression_probability_pp':.1},'lexical512_noninferiority_vs_bp':{'kind':'paired_net','candidate':'lexical512_from_lexical256','control':'pl','minimum_net_gain_pp':-.1}},'paired_inference':'Independent unit is the maze; joint success requires all three fixed models. For each paired-net endpoint use exact Clopper-Pearson gain and loss marginal one-sided alpha .025 each, lower=100*(gain_lower-loss_upper), upper=100*(gain_upper-loss_lower). Support if lower >= declared threshold; falsify if upper < threshold; else inconclusive. Regression-probability endpoint uses exact paired-loss count over4096 with one-sided alpha .05: support if upper<=.1pp, falsify if lower>.1pp, else inconclusive. Three separate endpoint decisions; no simultaneous familywise95 claim. With zero losses/4096 the upper probability bound is .0731113pp. NI concerns net accuracy, not standalone loss probability.','density_hypothesis_justification':'Retain the H2ax minimum-gain/safety/BP NI targets unchanged to test transport under increased loop density. Additional junctions may shorten candidate chains and diminish the cap remedy; this possibility does not justify lowering the gain target. No new training or retuning. Interpret endpoints within the new loop50 distribution; do not pool with loop20.', 'secondary':'Energy512 from identical lexical256 starts descriptive paired control; no learned-ranking advantage endpoint. BP AdamW first; all failures/caps retained.','inference':'Fixed H2ae LP and continued BP models. BP native float32 u3 eta.5 cap8192 check16; shared hidden float32 tol1e-6 cap4000 check8. Binary float64 four-color threshold -1e-8 cap512. One binary, veto and lexical256 prefix per input, reused exactly. Veto uses negative original-energy chain moves. Lexical256: chain size2..256, reachability and strict cycle-rank decrease, lexical selector, permits positive cost. Both512 continuations start from lexical256 with chain size2..512, same geometry and native suffix; lexical versus minimum original energy/exact lexical tie. All block solvers16 rounds, initial simple-path identity for cycle policies; no labels or shortest distances used for choices. Batch32.','gate':'Same six-policy helper, models, size, precision and caps as H2ax; reuse complete hash-bound preflight without redundant GPU trajectories. The generator remains the same study.maze function; only loop parameter/seed differ, and all fresh labels/uniqueness/clamps are independently audited before inference. Complete all-six-arm actual180 source replay on first two inputs per shard/model plus all84 previously nonsimple starts, exact predictions/iterations/flags/readouts/block flags, independent energies/flips and literal candidate costs. Hash-bound earlier gradient/readout/native/uncached objective checks and size-boundary/positive-restoration/selector synthetic gates. No padding invariance claim.','budget':{'inference_total_gpu_minutes':10,'per_worker_wall_seconds':300,'source_gate_wall_seconds':120,'input_cpu_audit_wall_seconds':120,'input_cpu_audit_max_processes':8,'full_cpu_audit_wall_seconds':300,'full_cpu_audit_max_processes':16,'statistics_wall_seconds':120,'enforcement':'Per batch and block-round watchdog plus external timeout. Preserve partial archives; no extensions.'},'archive':'Exclusive NPZ deflate1. Every final state, trace and counter retained. All six arms charged full dependency prefixes though computed once; lexical candidate scoring retained for audit. Hidden/diagnostics/archive separate.','accounting':{'fresh_inputs':4096,'planned_jobs':48,'primary_model_input_arm_evaluations':73728,'reused_binary_prefixes':12288,'reused_veto_prefixes':12288,'reused_lexical256_prefixes':24576,'new_optimizer_updates':0}}
    S.save(PP,p);S.save(DEST/'protocol.json',p);gate.update(protocol_sha256=S.sha(PP),source_preflight_sha256=S.sha(gate_path),new_gate_runs=0);S.save(DEST/'numerical-gate.json',gate);S.emit('density_transfer_frozen',sha256=S.sha(PP))


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
                result=Q.solve(models[seed],doubles[seed],t,a,rotation=(ji+lo//32)%6,watchdog=watchdog);out=result['outputs'];prefix=result['prefixes'];block=result['blocks'];timings.append(result['timing']);before=time.perf_counter()
                batch={k:np.stack([out[v][name].cpu().numpy() for v in range(6)]) for k,name in [('predictions','prediction'),('iterations','iterations'),('stopped','stopped'),('finite','finite')]};batch['predictions']=batch['predictions'].astype(bool);batch['native_final_logits']=out[0]['final_logits'].cpu().numpy()[None];margins=[];energies=[]
                for v in range(6):
                    m=doubles[seed][Q.MODEL_INDEX[v]];board=out[v]['prediction'].double();margins.append(m.flip_delta(t,board,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(m.energy(t,board,a.double()).cpu().numpy())
                batch.update(minimum_flip_margin64=np.stack(margins),energy64=np.stack(energies),native_minimum_flip_margin32=models[seed][0].flip_delta(t,out[0]['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy()[None],binary_flip_count=np.stack([prefix[v]['flip_count'].cpu().numpy() for v in [1]]),binary_cache_margin=np.stack([prefix[v]['minimum_flip_margin'].cpu().numpy() for v in [1]]),block_moves=np.stack([block[v]['block_moves'] for v in [2,3,4,5]]),post_block_sweeps=np.stack([block[v]['post_block_sweeps'] for v in [2,3,4,5]]),block_native_cap=np.stack([block[v]['native_cap'] for v in [2,3,4,5]]),block_cap=np.stack([block[v]['block_cap'] for v in [2,3,4,5]]),initial_simple_path=np.stack([block[v]['initial_simple_path'] for v in [2,3,4,5]]),hidden_steps=hs.cpu().numpy(),hidden_good=good.cpu().numpy(),couplings=a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                before=time.perf_counter();bp=run/f'batch{lo//32:02d}.npz';Z.save(bp,**batch)
                for v in [2,3,4,5]:
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
