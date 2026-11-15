"""Frozen answer-only AdamW comparison with a fresh native-decoder evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
import os
import platform
import time
import numpy as np
import study as S
import selective_fallback as F
import larger_relaxation as L
import contrastive_objective as C
from prepare_contrastive import DEST

PP=S.HERE/'contrastive-training-protocol.json'
BASE=2026111508
N=4096
ARMS=('unchanged','pl','contrastive')


def freeze():
    if PP.exists():raise FileExistsError(PP)
    gates={}
    for name in ['numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json']:
        path=DEST/name;gate=json.loads(path.read_text());assert gate['status']=='passed' and gate['objective_sha256']==S.sha(C.__file__)
        gates[str(path.relative_to(S.ROOT))]=S.sha(path)
    training=json.loads((DEST/'training-manifest.json').read_text());assert S.sha(DEST/'training-arrays.npz')==training['arrays_sha256']
    schedules={}
    for seed in range(3):
        rng=np.random.default_rng(np.random.SeedSequence([BASE,30,seed]));indices=rng.integers(0,1000,(C.UPDATES,C.BATCH),dtype=np.int64)
        path=DEST/f'minibatches-seed{seed}.npz';np.savez_compressed(path,indices=indices);schedules[str(seed)]={'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path)}
    parent=json.loads((S.HERE/'larger_relaxation-protocol.json').read_text())
    sources={str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['contrastive_objective.py','contrastive_training.py','study.py','larger_relaxation.py','binary_preserving.py','selective_fallback.py','gate_contrastive.py']}
    sources.update(parent['source_sha256'])
    p={'id':'answer-only-endpoint-contrastive-training','frozen_utc':S.utc(),'status':'frozen_before_scientific_fits_and_fresh_cohort','hypothesis_id':'H2v','arms':list(ARMS),'models':parent['models'],'source_sha256':sources,'gates':gates,'training_manifest':training,'training_manifest_sha256':S.sha(DEST/'training-manifest.json'),'readout_manifests':{str(seed):{'manifest':json.loads((DEST/f'train-readout-seed{seed}.json').read_text()),'sha256':S.sha(DEST/f'train-readout-seed{seed}.json')} for seed in range(3)},'minibatch_schedules':schedules,
       'training':{'data':'Actual original1000train.csv30x30mazes. Not regenerated. Frozen training arrays/readouts are retained.','trainable':['T','b','kappa'],'n_trainable':23,'frozen':['mu','beta','all nontrainable buffers'],'readout':'Fixed original-model couplings, float32, batches64, hidden tol1e-6 cap4000 check8. Both fits use identical arrays.','optimizer':'BP AdamW','updates':400,'lr':.01,'weight_decay':0.,'betas':[.9,.999],'eps':1e-8,'foreach':False,'batch_size':64,'minibatches':'Identical seed-specific400x64fixed schedule sampled uniformly with replacement; both arms start from identical immutable checkpoint and optimizer state.','loss':'PL + weight*mean(softplus(1+(E(label)-E(endpoints))/max(1,selected free label cells)))','contrastive_weight':.1,'control_weight':0.,'matched_compute':'Both arms construct PL and contrastive computational graphs; only fixed scalar weight differs. No hidden-layer training or implicit adjoint.','scope':'Answer-only continued fine-tuning, intentionally different from original full-model training. No target/development accuracy, margin, weight, update or learning-rate tuning.'},
       'cohort':{'n_unique':N,'length':120,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze; SeedSequence([2026111508,120,20,index]), index0..4095','shards':16,'shard_size':256,'prior_disjointness':'Distinct fixed base seed; additionally SHA256 tokens must not overlap prior H2r/H2s/H2t/H2ucohorts. No pooling.'},
       'inference':{'decoder':'Native CRAM only; no graph-informed selection, lambda, BFS repair or label feedback.','dtype':'float32','hidden_tol':1e-6,'hidden_cap':4000,'hidden_check':8,'answer_initial_u':3.,'answer_eta':.5,'answer_cap':8192,'answer_check':16,'batch_size':64,'certificate':'Original model flip_delta >= -1e-6, finite answer and converged hidden layer','shared_readout':'One unchanged hidden solve per input/model shared identically by all three arms. Fixed mu/beta are audited unchanged.','arm_order':'Rotate arm order by job index plus minibatch index.'},
       'primary_ranking':{'hypothesis':'Contrastive fine-tuning improves joint-three-model label-over-endpoints ranking by at least90percentage points relative to continued PL.','target_pp':90.,'success':'All three fixed models have float64[E(label)-E(endpoints)] / max(1,selected free label cells) < -1e-6. Ambiguous/zero/nonfinite gaps count false.','decision':'Paired gains and losses over4096 independent inputs. CPgainlower(alpha.025)-CPlossupper(alpha.025) gives separately one-sided95net lower; upper analogously. Support if lower>=90pp; falsify if upper<90pp; otherwise inconclusive.','scope':'Pairwise ranking only; not a globalminimum certificate.'},
       'primary_accuracy':{'hypothesis':'Contrastive joint-three-model valid-shortest accuracy is noninferior to continued PL within0.1percentage point.','margin_pp':.1,'success':'All three outputs independently graph-valid shortest simple paths, finite, certified and hidden-converged.','decision':'Paired loss event = PLjointsuccess and contrastivejointfailure. Conservative noninferiority support if exactCPone-sided95lossprobability upper(alpha.05)<=0.1pp; falsify if corresponding lower>0.1pp; otherwise inconclusive. Report gains/losses and net accuracy separately.','unit':'Independent input; three fixed models are not independent repeats.'},
       'overall':'Supported only if both separately frozen endpoints support; falsified if either falsifies; otherwise inconclusive. Ranking cannot substitute for solvequality. No multiplicity-adjusted jointconfidence claim.',
       'secondary':'Per-model ranking, solves, certification, endpoint outputs, path length, training loss/parameter change and measured runtime descriptive. Unchanged control retained.',
       'budget':{'total_gpu_minute_envelope':30,'training_worker_seconds':300,'evaluation_worker_seconds':600,'watchdog':'Check every training update and evaluation batch. Preserve partial checkpoints/results on timeout; no cap/sample changes or resumed budget extension after outcomes. Preparation/numerical gates recorded separately.'},
       'fit_jobs':[{'id':f's{seed}-{arm}','seed':seed,'arm':arm,'weight':weight,'gpu':(seed+index)%2} for seed in range(3) for index,(arm,weight) in enumerate([('pl',0.),('contrastive',.1)])],
       'jobs':[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':(shard*3+seed)%2} for shard in range(16) for seed in range(3)],
       'retained':'Source/model/data/schedule hashes; all400updatesloss traces; final checkpoint and optimizer state; frozen-readout caches; freshinput graphs/labels; all final logits/boards/flags/steps, couplings, float64ranking gaps, nativeflipmargins, energy, per-worker timings/progress and failures.'}
    S.save(PP,p);S.emit('contrastive_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for model in p['models'].values():assert S.sha(S.ROOT/model['path'])==model['sha256']
    return p


def train(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter()
    for path,h in p['gates'].items():assert S.sha(S.ROOT/path)==h
    assert S.sha(DEST/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
    with np.load(DEST/'training-arrays.npz') as data:t=torch.tensor(data['tokens'],device='cuda');y=torch.tensor(data['labels'],dtype=torch.float32,device='cuda')
    state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};S.save(DEST/f'train-worker{gpu}.json',state)
    for job in p['fit_jobs']:
        if job['gpu']!=gpu:continue
        run=DEST/'fits'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        seed=job['seed'];source=DEST/f'train-readout-seed{seed}.npz';assert S.sha(source)==p['readout_manifests'][str(seed)]['manifest']['readout_sha256']
        with np.load(source) as x:a=torch.tensor(x['couplings'],device='cuda')
        sched=p['minibatch_schedules'][str(seed)];assert S.sha(S.ROOT/sched['path'])==sched['sha256']
        with np.load(S.ROOT/sched['path']) as x:indices=torch.tensor(x['indices'],device='cuda')
        model=S.model_for(seed,30,'original','cuda');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};opt=C.optimizer(torch,model);trace=[];torch.cuda.synchronize();fit_start=time.perf_counter()
        for step,idx in enumerate(indices):
            opt.zero_grad(set_to_none=True);loss,pl,contrast,gap=C.losses(model,t[idx],y[idx],a[idx],job['weight']);loss.backward();opt.step()
            trace.append({'step':step+1,'loss':float(loss.detach()),'pl':float(pl.detach()),'contrast':float(contrast.detach()),'mean_normalized_label_minus_endpoint':float(gap.detach().mean())})
            if not all(math.isfinite(v) for v in trace[-1].values()):raise FloatingPointError(trace[-1])
            if step%25==24:state.update(current_job=job,updates_complete=step+1,elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'train-worker{gpu}.json',state)
            if time.perf_counter()-started>300:
                torch.save(model.state_dict(),run/'partial.pt');S.save(run/'partial-trace.json',trace);raise TimeoutError('Frozen300s training worker budget')
        torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit_start
        final={k:v.detach().cpu() for k,v in model.state_dict().items()};frozen=[k for k in final if k not in C.PARAMETERS];assert all(torch.equal(initial[k],final[k]) for k in frozen)
        torch.save(final,run/'checkpoint.pt');torch.save(opt.state_dict(),run/'optimizer.pt');S.save(run/'trace.json',trace)
        r={'status':'complete','utc':S.utc(),'job':job,'protocol_sha256':S.sha(PP),'initial_checkpoint_sha256':p['models'][str(seed)]['sha256'],'checkpoint_sha256':S.sha(run/'checkpoint.pt'),'optimizer_sha256':S.sha(run/'optimizer.pt'),'trace_sha256':S.sha(run/'trace.json'),'schedule_sha256':sched['sha256'],'readout_sha256':S.sha(source),'fit_seconds':fit_seconds,'updates':len(trace),'unchanged_frozen_tensors':frozen,'parameter_changes':{k:float((final[k]-initial[k]).abs().max()) for k in C.PARAMETERS},'torch':torch.__version__,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('contrastive_fit_complete',**r)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'train-worker{gpu}.json',state)


def make_one(index):return S.maze(120,.2,np.random.SeedSequence([BASE,120,20,index]))


def prepare():
    p=protocol();started=time.perf_counter();manifest=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz'
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));manifest.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('contrastive_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':manifest,'seconds':time.perf_counter()-started})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();from cram.maze_learned import gather
    p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP)
    state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};started=time.perf_counter();models={};doubles={};checkpoints={}
    for seed in range(3):
        models[seed]=[];doubles[seed]=[];checkpoints[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cuda')
            cp=S.OUT/f'models/seed{seed}.pt'
            if arm!='unchanged':
                cp=DEST/f'fits/s{seed}-{arm}/checkpoint.pt';r=json.loads(cp.with_name('result.json').read_text());assert r['status']=='complete' and S.sha(cp)==r['checkpoint_sha256'];model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True))
            models[seed].append(model);doubles[seed].append(copy.deepcopy(model).double());checkpoints[seed].append(S.sha(cp))
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens'];labels=x['labels']
        saved={k:[] for k in ['predictions','answer_steps','stopped','finite','final_logits','minimum_flip_margin','energy','normalized_rank_gap','hidden_steps','hidden_good','couplings']};answer_seconds=np.zeros(3);hidden_seconds=diagnostic_seconds=0.;job_start=time.perf_counter();seed=job['seed']
        for lo in range(0,256,64):
            t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');y=torch.tensor(labels[lo:lo+64],device='cuda',dtype=torch.float64);base=models[seed][0]
            with torch.no_grad():
                torch.cuda.synchronize();start=time.perf_counter();psi,steps=base.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=base.parts(t,base.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,base.zoffs,math.inf),lc[:,None]],1),1);good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1));a=base.couplings(t,psi);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-start
                out={};offset=(ji+lo//64)%3
                for v in list(range(3))[offset:]+list(range(3))[:offset]:
                    torch.cuda.synchronize();start=time.perf_counter();out[v]=L.solve(models[seed][v],t,a,torch.full_like(a[:,0],3.),8192,0.);torch.cuda.synchronize();answer_seconds[v]+=time.perf_counter()-start
                start=time.perf_counter()
                for k,key in enumerate(['predictions','answer_steps','stopped','finite','final_logits']):saved[key].append(np.stack([out[v][k].cpu().numpy() for v in range(3)]))
                saved['minimum_flip_margin'].append(np.stack([models[seed][v].flip_delta(t,out[v][0],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy() for v in range(3)]));saved['energy'].append(np.stack([models[seed][v].energy(t,out[v][0],a).cpu().numpy() for v in range(3)]))
                count=(y*(t==1)).sum((1,2)).clamp_min(1);ad=a.double();zero=torch.zeros_like(y);saved['normalized_rank_gap'].append(np.stack([((m.energy(t,y,ad)-m.energy(t,zero,ad))/count).cpu().numpy() for m in doubles[seed]]))
                saved['hidden_steps'].append(steps.cpu().numpy());saved['hidden_good'].append(good.cpu().numpy());saved['couplings'].append(a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-start
            state.update(current_job=job,completed_inputs=lo+64,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state)
            if time.perf_counter()-started>600:
                np.savez_compressed(run/'partial.npz',**{k:np.concatenate(v,axis=0 if k in ['hidden_steps','hidden_good','couplings'] else 1) for k,v in saved.items()});raise TimeoutError('Frozen600s evaluation worker budget')
        arrays={k:np.concatenate(v,axis=0 if k in ['hidden_steps','hidden_good','couplings'] else 1) for k,v in saved.items()};arrays['predictions']=arrays['predictions'].astype(bool);np.savez_compressed(run/'outputs.npz',**arrays)
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':checkpoints[seed],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':answer_seconds.tolist(),'diagnostic_seconds':diagnostic_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('contrastive_eval_job_complete',job=job['id'],gpu=gpu,answer_seconds=answer_seconds.tolist(),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('contrastive_eval_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','train','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command in ['train','evaluate']:globals()[args.command](args.gpu)
    else:globals()[args.command]()
