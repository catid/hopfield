"""Frozen H2z: fixed binary-preserving lambda4 on norm-matched LP models."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import larger_relaxation as L
import binary_preserving as B
from gate_lp_relaxation import DEST,STRENGTH
from lp_transfer import PP as PARENT,protocol as parent_protocol

PP=S.HERE/'lp-relaxation-protocol.json'
BASE=2026111510
N=4096
ARMS=('pl','lp_native','lp_lambda4')
LAMBDAS=(0.,0.,STRENGTH)


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=parent_protocol();gates={}
    for name in ['numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json']:
        path=DEST/name;r=json.loads(path.read_text());assert r['status']=='passed' and r['lambda_fixed']==4.
        assert r['script_sha256']==S.sha(S.HERE/'gate_lp_relaxation.py')
        gates[str(path.relative_to(S.ROOT))]=S.sha(path)
    models={str(seed):{arm:parent['models'][str(seed)]['pl' if arm=='pl' else 'lp_scaled'] for arm in ARMS} for seed in range(3)}
    sources=dict(parent['source_sha256']);sources.update({str(path.relative_to(S.ROOT)):S.sha(path) for path in [S.HERE/'lp_relaxation.py',S.HERE/'gate_lp_relaxation.py']})
    p={'id':'norm-matched-LP-binary-preserving-lambda4','hypothesis_id':'H2z','status':'frozen_before_fresh_generation_and_inference','frozen_utc':S.utc(),'parent_protocol_sha256':S.sha(PARENT),'arms':list(ARMS),'lambdas':list(LAMBDAS),'models':models,'source_sha256':sources,'gates':gates,
      'intervention':{'fixed_lambda':4.,'energy':'Native E(p)+4 sum_free p_i(1-p_i)','field_change':'4(2p_i-1) onfreecells, zeroelsewhere','binary_energy':'Exactlyzero addedterm on everybinaryboard; originalbinaryflipcertificate unchanged.','rationale':'Native self-coefficients kappa0>0 and attention>=0. Coordinatecurvature1/[p(1-p)]-2*kappa0*a0-8 is strictlynegative for p in(.45,.55), upperbound4.040405-8. This excludes such states from continuouslocalminima, but guarantees neither convergence nor validrounding.','selection':'One theory-chosen lambda4, fixedbefore sourcegate; no sweep, targettuning or postgate retuning.'},
      'controls':{'pl':'FrozenH2v BPAdamWcontinuedPL native inference; differentobjective/computebudget fromLP, not fairoptimizerbenchmark.','lp_native':'Same norm-matchedLPcheckpoint as intervention, lambda0; independent native trajectory on everyfreshinput.','hidden':'Sameunchangedmu/beta and couplings sharedacrossallarms.','labels':'Freshlabels used only byofflineindependentgraph evaluation; no training or inference labelaccess.'},
      'cohort':{'n_unique':N,'length':120,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze with SeedSequence([2026111510,120,20,index]),index0..4095','shards':16,'shard_size':256,'disjointness':'Canonicaltoken hashes checkedagainst H2r/H2s/H2t/H2u/H2v/H2y; no pooling.'},
      'inference':{'dtype':'float32','hidden_tol':1e-6,'hidden_cap':4000,'hidden_check':8,'answer_initial_u':3.,'answer_eta':.5,'answer_cap':8192,'answer_check':16,'batch_size':64,'certificate':'Originalbinaryflip_delta>=-1e-6, finiteanswer and convergedhidden; independentgraph validshortest for success.','arm_order':'Rotate byjobindex andminibatchindex.','forbidden':'No graphfallback, repair, lambda/eta/capchanges, targettuning or postcaprescue. Oscillation/nonconvergence is retained asfailure.'},
      'primary_gain':{'hypothesis':'lambda4 improves joint-three-model validshortest success by at least90percentagepoints versusLPnative.','target_pp':90.,'decision':'Pairedgains/losses over4096independentinputs; separatelyone-sided95netbounds fromCPmarginalalpha.025. Supportlower>=90pp; falsifyupper<90pp; elseinconclusive.'},
      'primary_noninferiority':{'hypothesis':'lambda4 joint-three-model validshortest accuracy is noninferior within0.1percentagepoint versusfrozenBPPLnative.','margin_pp':.1,'decision':'Pairedloss event=BPPLjointsuccess andlambda4jointfailure. ExactCPone-sided95lossboundsalpha.05. Supportupper<=.1pp; falsifylower>.1pp; elseinconclusive. Reportgains/netseparately.'},
      'overall':'Support onlyifbothendpoints support; falsifyifeitherfalsifies; otherwiseinconclusive. Separateendpoints, no simultaneous95coverageclaim. Fixedmodels are repeats, not independentinputs.',
      'secondary':'Per-model wrongcertifiedstops, timeouts, nonfinite/hiddenfailures, binaryminimumflip, original/modifiedcontinuousresiduals, nearhalf/freecounts, energy, iterations/timing descriptive. Positivebinaryranking is not solvingsuccess.',
      'budget':{'total_gpu_minute_envelope':30,'per_worker_wall_seconds':900,'watchdog':'Checkeverybatch, preservepartial onexpiry; no extension/rescue.'},
      'timing':'CUDA-synchronized sharedhidden andperarm answer seconds; diagnostics/transfers/archives separately includedworkerwall. One scientificpass, no runtimeCI. Sourcegate384trajectories separate.',
      'retained':'Allinputs/labels, outputs/logits, originalbinaryenergy/margins, original/modifiedcontinuousresiduals, fractionalcounts, sharedcouplings, model/source/protocolhashes, timings, failures.',
      'jobs':[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':(shard*3+seed)%2} for shard in range(16) for seed in range(3)]}
    S.save(PP,p);S.emit('lp_relaxation_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['parent_protocol_sha256']==S.sha(PARENT)
    for path,h in {**p['source_sha256'],**p['gates']}.items():assert S.sha(S.ROOT/path)==h,path
    for group in p['models'].values():
        for item in group.values():assert S.sha(S.ROOT/item['path'])==item['sha256']
    return p


def make_one(index):return S.maze(120,.2,np.random.SeedSequence([BASE,120,20,index]))


def prepare():
    protocol();started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz'
            if path.exists():raise FileExistsError(path)
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]))
            datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('lp_relaxation_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();from cram.maze_learned import gather
    p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={}
    for seed in range(3):
        models[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cuda');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location='cuda',weights_only=True));models[seed].append(model)
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens']
        saved={k:[] for k in ['predictions','answer_steps','stopped','finite','final_logits','minimum_flip_margin','energy','original_field_residual','modified_field_residual','near_half_free','fractional_free','hidden_steps','hidden_good','couplings']}
        answer_seconds=np.zeros(3);hidden_seconds=diagnostic_seconds=0.;job_start=time.perf_counter();seed=job['seed']
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');base=models[seed][0]
                torch.cuda.synchronize();start=time.perf_counter();psi,hs=base.settle(t,tol=1e-6,max_steps=4000,check=8)
                lg,la,lc=base.parts(t,base.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,base.zoffs,math.inf),lc[:,None]],1),1)
                good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
                a=base.couplings(t,psi);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-start
                out={};offset=(ji+lo//64)%3
                for v in list(range(3))[offset:]+list(range(3))[:offset]:
                    torch.cuda.synchronize();start=time.perf_counter();out[v]=L.solve(models[seed][v],t,a,torch.full_like(a[:,0],3.),8192,LAMBDAS[v]);torch.cuda.synchronize();answer_seconds[v]+=time.perf_counter()-start
                start=time.perf_counter()
                for k,key in enumerate(['predictions','answer_steps','stopped','finite','final_logits']):saved[key].append(np.stack([out[v][k].cpu().numpy() for v in range(3)]))
                diag={k:[] for k in ['minimum_flip_margin','energy','original_field_residual','modified_field_residual','near_half_free','fractional_free']}
                for v,model in enumerate(models[seed]):
                    binary,u=out[v][0],out[v][4];soft=torch.sigmoid(u);fr=t==1
                    diag['minimum_flip_margin'].append(model.flip_delta(t,binary,a).masked_fill(~fr,float('inf')).flatten(1).amin(1).cpu().numpy())
                    diag['energy'].append(model.energy(t,binary,a).cpu().numpy())
                    for key,strength in [('original_field_residual',0.),('modified_field_residual',LAMBDAS[v])]:diag[key].append((B.field(model,t,soft,a,strength)-u).abs().masked_fill(~fr,0).flatten(1).amax(1).cpu().numpy())
                    diag['near_half_free'].append(((soft>.45)&(soft<.55)&fr).sum((1,2)).cpu().numpy())
                    diag['fractional_free'].append(((soft>.05)&(soft<.95)&fr).sum((1,2)).cpu().numpy())
                for key,values in diag.items():saved[key].append(np.stack(values))
                saved['hidden_steps'].append(hs.cpu().numpy());saved['hidden_good'].append(good.cpu().numpy());saved['couplings'].append(a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-start
                state.update(current_job=job,completed_inputs=lo+64,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state)
                if time.perf_counter()-started>900:
                    np.savez_compressed(run/'partial.npz',**{k:np.concatenate(v,axis=0 if k in ['hidden_steps','hidden_good','couplings'] else 1) for k,v in saved.items()});raise TimeoutError('Frozen900sworkerbudget')
        arrays={k:np.concatenate(v,axis=0 if k in ['hidden_steps','hidden_good','couplings'] else 1) for k,v in saved.items()};arrays['predictions']=arrays['predictions'].astype(bool);np.savez_compressed(run/'outputs.npz',**arrays)
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':answer_seconds.tolist(),'diagnostic_seconds':diagnostic_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('lp_relaxation_job_complete',job=job['id'],gpu=gpu,answer_seconds=answer_seconds.tolist(),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('lp_relaxation_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','prepare','evaluate']);p.add_argument('--gpu',type=int,choices=[0,1]);args=p.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
