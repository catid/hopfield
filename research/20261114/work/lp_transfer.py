"""Fresh native inference of positive-scaled LPmodels versus frozenBPAdamWPL."""
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
import larger_relaxation as L
from prepare_lp_transfer import DEST,TRAIN,NORM

PP=S.HERE/'lp-transfer-protocol.json'
BASE=2026111509
N=4096
ARMS=('pl','lp_scaled','unchanged')


def freeze():
    if PP.exists():raise FileExistsError(PP)
    cast=json.loads((DEST/'cast-gate.json').read_text());assert cast['status']=='passed';gates={}
    for name in ['cast-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json']:
        path=DEST/name;r=json.loads(path.read_text());assert r['status']=='passed';gates[str(path.relative_to(S.ROOT))]=S.sha(path)
    models={}
    for seed in range(3):
        models[str(seed)]={}
        for arm in ARMS:
            path=TRAIN/f'fits/s{seed}-pl/checkpoint.pt' if arm=='pl' else DEST/f'lp-scaled-seed{seed}.pt' if arm=='lp_scaled' else S.OUT/f'models/seed{seed}.pt'
            models[str(seed)][arm]={'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path)}
    parent=json.loads((S.HERE/'contrastive-training-protocol.json').read_text());sources={k:v for k,v in parent['source_sha256'].items() if k.startswith('upstream/')};sources.update({str(path.relative_to(S.ROOT)):S.sha(path) for path in [S.HERE/'lp_transfer.py',S.HERE/'prepare_lp_transfer.py',S.HERE/'study.py',S.HERE/'larger_relaxation.py',S.HERE/'binary_preserving.py',S.HERE/'selective_fallback.py']})
    p={'id':'norm-matched-LP-native-size-transfer','hypothesis_id':'H2y','frozen_utc':S.utc(),'status':'frozen_before_fresh_cohort_and_inference','arms':list(ARMS),'models':models,'gates':gates,'source_sha256':sources,'cast_receipt':cast,'lp_source':{'min_norm_protocol_sha256':S.sha(S.HERE/'training-min-norm-protocol.json'),'verification_sha256':S.sha(NORM/'independent-verification.json'),'choice':'One fixed positive uniformscale permodel: original maxabsoluteanswerparameter divided by savedverifiedH2xprimal maxabsoluteparameter. Onefloat32cast; no retuning.','scope':'LPparameters solve allsource-label localflip and two negative-ranking constraints, then undergo uniformscale. LP andBP optimize differentobjectives with differentcomputebudgets; this is a model/objectivecomparison, not equal-computeoptimizerbenchmark.'},
       'controls':{'pl':'Frozen H2v BPAdamWcontinuedPL: sameoriginal1000trainingarrays, answeronly23parameters,400updates,batch64,lr.01,wd0; no newfit.','unchanged':'Immutableoriginalcheckpoint; same frozenmu/beta/readout asotherarms.','added_training_information':'LP uses originaltraininglabels and explicitly constructed endpoint/allopennnegativeboards. At inference everyarm gets onlysuppliedgraphtokens/fixedhiddenreadout, never labels.'},
       'cohort':{'n_unique':N,'length':120,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze; SeedSequence([2026111509,120,20,index]), index0..4095','shards':16,'shard_size':256,'disjointness':'Freshbase plus exactcanonicaltokendigestcheck against H2r/H2s/H2t/H2u/H2v. No pooling or newsourcefitting.'},
       'inference':{'decoder':'Nativeonly, no fallback, graphvalidationselector, lambda or repair.','dtype':'float32','hidden_tol':1e-6,'hidden_cap':4000,'hidden_check':8,'answer_initial_u':3.,'answer_eta':.5,'answer_cap':8192,'answer_check':16,'batch_size':64,'certificate':'Originalflip_delta>=-1e-6 and finiteanswer; success also requires hiddenconvergence.','readout':'One unchangedhidden solve perinput/model, sharedidentically acrossall3arms.','arm_order':'Rotate byjobindex plusminibatchindex.','no_retuning':'No cap, eta, scale, dtype, batch, sourceobjective, modelselection or target-dependentrescue changes.'},
       'primary_ranking':{'hypothesis':'Norm-matchedLP improves joint-three-model label-below-BOTH-negative ranking by at least90percentagepoints versus frozencontinuedPL.','target_pp':90.,'success':'For all3fixedmodels, bothfloat64normalizedgaps[E(label)-E(endpoints)]/selectedfreelabelcount and [E(label)-E(allopen)]/unselectedfreecount are strictly<-1e-6. Counts clampedminimum1; ambiguity/nonfinite fail.','decision':'Pairedgain/loss events over4096independentinputs. Separately one-sided95net bounds use CPmarginaltailsalpha.025. Support lower>=90pp, falsify upper<90pp, elseinconclusive.','scope':'Twoexplicitbinarynegative comparisons, not all-boardglobaloptimality.'},
       'primary_accuracy':{'hypothesis':'Norm-matchedLP joint valid-shortest accuracy is noninferior within0.1percentagepoint to frozencontinuedPL.','margin_pp':.1,'success':'All3outputs independentgraph-valid shortestpaths, finite/certified/hiddenconverged.','decision':'Pairedloss event=PLjointsuccess and LPjointfailure. Support if exactCPone-sided95loss upper(alpha.05)<=.1pp; falsify if lower>.1pp; otherwiseinconclusive. Report gains/losses/net separately.'},
       'overall':'Supported only if both endpoints support; falsified if eitherfalsifies; otherwiseinconclusive. Rankingcannot replace solvequality. No simultaneous95confidenceclaim across endpoints.','secondary':'Per-model bothrankingcomponents, validity/shortestness, margins, selectedcells, wrongcertificates, timeouts, timing and unchangedcomparison descriptive.',
       'budget':{'total_gpu_minute_envelope':30,'per_worker_wall_seconds':900,'watchdog':'Check everybatch; preserve partialoutputonbudgetexpiry, no extension or resume underchangedcap.'},'timing':'Synchronized commonhiddentime and perarmanswersolve time; diagnostics/CPUtransfer/archivewriting separate in workerwall. One fullscientificpass, no runtimeCI. Source-onlycast and originaltraining64inputthroughput gates are separate preparation, not freshevidence.',
       'retained':'Allfreshgraphs/labels/digests, sharedcouplings, finalfloat32logits/predictions/steps/flags, energies/margins, bothfloat64rankinggaps, perjobtimings, source/model/protocolhashes and failures.','jobs':[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':(shard*3+seed)%2} for shard in range(16) for seed in range(3)]}
    S.save(PP,p);S.emit('lp_transfer_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for path,h in p['gates'].items():assert S.sha(S.ROOT/path)==h,path
    for group in p['models'].values():
        for item in group.values():assert S.sha(S.ROOT/item['path'])==item['sha256']
    return p


def make_one(index):return S.maze(120,.2,np.random.SeedSequence([BASE,120,20,index]))


def prepare():
    p=protocol();started=time.perf_counter();manifest=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz'
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));manifest.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('lp_transfer_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':manifest,'seconds':time.perf_counter()-started})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();from cram.maze_learned import gather
    p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={};doubles={}
    for seed in range(3):
        models[seed]=[];doubles[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cuda');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location='cuda',weights_only=True));models[seed].append(model);doubles[seed].append(copy.deepcopy(model).double())
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens'];labels=x['labels']
        saved={k:[] for k in ['predictions','answer_steps','stopped','finite','final_logits','minimum_flip_margin','energy','normalized_endpoint_gap','normalized_dense_gap','hidden_steps','hidden_good','couplings']};answer_seconds=np.zeros(3);hidden_seconds=diagnostic_seconds=0.;job_start=time.perf_counter();seed=job['seed']
        for lo in range(0,256,64):
            t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');y=torch.tensor(labels[lo:lo+64],device='cuda',dtype=torch.float64);base=models[seed][2]
            with torch.no_grad():
                torch.cuda.synchronize();start=time.perf_counter();psi,steps=base.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=base.parts(t,base.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,base.zoffs,math.inf),lc[:,None]],1),1);good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1));a=base.couplings(t,psi);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-start
                out={};offset=(ji+lo//64)%3
                for v in list(range(3))[offset:]+list(range(3))[:offset]:
                    torch.cuda.synchronize();start=time.perf_counter();out[v]=L.solve(models[seed][v],t,a,torch.full_like(a[:,0],3.),8192,0.);torch.cuda.synchronize();answer_seconds[v]+=time.perf_counter()-start
                start=time.perf_counter()
                for k,key in enumerate(['predictions','answer_steps','stopped','finite','final_logits']):saved[key].append(np.stack([out[v][k].cpu().numpy() for v in range(3)]))
                saved['minimum_flip_margin'].append(np.stack([models[seed][v].flip_delta(t,out[v][0],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy() for v in range(3)]));saved['energy'].append(np.stack([models[seed][v].energy(t,out[v][0],a).cpu().numpy() for v in range(3)]))
                on=(y*(t==1)).sum((1,2)).clamp_min(1);off=((1-y)*(t==1)).sum((1,2)).clamp_min(1);ad=a.double();zero=torch.zeros_like(y);ones=torch.ones_like(y);endpoint=[];dense=[]
                for model in doubles[seed]:
                    energy=model.energy(t,y,ad);endpoint.append(((energy-model.energy(t,zero,ad))/on).cpu().numpy());dense.append(((energy-model.energy(t,ones,ad))/off).cpu().numpy())
                saved['normalized_endpoint_gap'].append(np.stack(endpoint));saved['normalized_dense_gap'].append(np.stack(dense));saved['hidden_steps'].append(steps.cpu().numpy());saved['hidden_good'].append(good.cpu().numpy());saved['couplings'].append(a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-start
            state.update(current_job=job,completed_inputs=lo+64,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state)
            if time.perf_counter()-started>900:
                np.savez_compressed(run/'partial.npz',**{k:np.concatenate(v,axis=0 if k in ['hidden_steps','hidden_good','couplings'] else 1) for k,v in saved.items()});raise TimeoutError('Frozen900sworkerbudget')
        arrays={k:np.concatenate(v,axis=0 if k in ['hidden_steps','hidden_good','couplings'] else 1) for k,v in saved.items()};arrays['predictions']=arrays['predictions'].astype(bool);np.savez_compressed(run/'outputs.npz',**arrays)
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':answer_seconds.tolist(),'diagnostic_seconds':diagnostic_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('lp_transfer_eval_job_complete',job=job['id'],gpu=gpu,answer_seconds=answer_seconds.tolist(),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('lp_transfer_eval_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','prepare','evaluate']);p.add_argument('--gpu',type=int,choices=[0,1]);args=p.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
