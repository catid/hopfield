"""H2ac mined-hard-negative LP transfer with the fixed binary decoder."""
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
import binary_coordinate as C
from prepare_hard_negative_transfer import DEST
from lp_transfer import PP as PARENT,protocol as parent_protocol

PP=S.HERE/'hard-negative-transfer-protocol.json'
BASE=2026111512
ARMS=('pl','lp_binary','hard_negative_binary')
AXIS0={'hidden_steps','hidden_good','couplings'}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=json.loads((S.HERE/'binary-coordinate-protocol.json').read_text());baseline=parent_protocol()
    p.update(id='norm-matched-hard-negative-LP-binary-transfer',hypothesis_id='H2ac',frozen_utc=S.utc(),arms=list(ARMS))
    p['hard_negative_feasibility_protocol_sha256']=S.sha(S.HERE/'hard-negative-feasibility-protocol.json')
    p['binary_decoder_protocol_sha256']=S.sha(S.HERE/'binary-coordinate-protocol.json')
    p['models']={str(seed):{'pl':baseline['models'][str(seed)]['pl'],'lp_binary':baseline['models'][str(seed)]['lp_scaled'],'hard_negative_binary':{'path':str((DEST/f'hard-negative-scaled-seed{seed}.pt').relative_to(S.ROOT)),'sha256':S.sha(DEST/f'hard-negative-scaled-seed{seed}.pt')}} for seed in range(3)}
    for name in ['cast-gate.json','numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json']:
        path=DEST/name;r=json.loads(path.read_text());assert r['status']=='passed';assert r['script_sha256']==S.sha(S.HERE/('prepare_hard_negative_transfer.py' if name=='cast-gate.json' else 'gate_hard_negative_transfer.py'));p['gates'][str(path.relative_to(S.ROOT))]=S.sha(path)
    for name in ['prepare_hard_negative_transfer.py','gate_hard_negative_transfer.py','hard_negative_transfer.py','verify_training_feasibility.py']:
        path=S.HERE/name;p['source_sha256'][str(path.relative_to(S.ROOT))]=S.sha(path)
    path=S.OUT/'hard_negative_feasibility/import-isolation.json';assert json.loads(path.read_text())['isolation_passed'];p['gates'][str(path.relative_to(S.ROOT))]=S.sha(path)
    p['candidate_selection']='Exactly H2ab independently verified witnesses, one positive uniform scale to each original answer coefficient maximum, one float32 cast; all 1699842 source inequalities verified positive. No selection, retry, extra mining round or target use.'
    p['controls']={'pl':'Frozen H2v BP AdamW continued PL native float32; objective/budget differs from LP, not equal-compute optimizer comparison.','lp_binary':'Frozen H2aa original LP coefficients, unchanged binary decoder.','hard_negative_binary':'Frozen H2ab mined-hard-negative coefficients, unchanged binary decoder.','hidden':'Same immutable mu/beta, one shared float32 readout per model-input.','labels':'No labels, shortest distance, graph search or repair enter inference.'}
    p['cohort'].update(base_seed=BASE,generator='study.maze SeedSequence([2026111512,120,20,index]), index0..4095',disjointness='Exact canonical token digests against H2r/H2s/H2t/H2u/H2v/H2y/H2z/H2aa; no pooling.')
    p['primary_gain'].update(hypothesis='New hard-negative LP improves joint-three-model valid shortest success by at least10 percentage points versus old LP binary.',target_pp=10.,decision='Paired gain/loss events over4096 independent inputs. Separately one-sided95 net bounds use CP marginal alpha.025. Support lower>=10pp; falsify upper<10pp; else inconclusive.')
    p['primary_noninferiority']['hypothesis']='New hard-negative LP joint valid shortest accuracy is noninferior within.1 percentage point versus BP PL native.'
    p['secondary']='Wrong certified stops, timeouts/nonfinite/hidden failures, original energy certificates, selected cells, binary flips/sweeps, native updates, times descriptive. No new fitting or graph rescue.'
    p['budget']={'total_gpu_minute_envelope':20,'per_worker_wall_seconds':600,'watchdog':'Every batch; preserve partial on expiry, no extension or rescue.'}
    p['retained']='All fresh inputs/labels, shared readouts, all predictions/flags, one native final-logit array, native update and binary sweep counts, both binary flip counts/cache margins, original-energy float64 diagnostics, native float32 margin, timings/hashes/failures.'
    S.save(PP,p);S.emit('hard_negative_transfer_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['parent_protocol_sha256']==S.sha(PARENT);assert p['hard_negative_feasibility_protocol_sha256']==S.sha(S.HERE/'hard-negative-feasibility-protocol.json');assert p['binary_decoder_protocol_sha256']==S.sha(S.HERE/'binary-coordinate-protocol.json')
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
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('hard_negative_transfer_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();from cram.maze_learned import gather
    p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={};doubles={}
    for seed in range(3):
        models[seed]=[];doubles[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cuda');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location='cuda',weights_only=True));models[seed].append(model);doubles[seed].append(copy.deepcopy(model).double())
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens']
        keys=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','hidden_steps','hidden_good','couplings'];saved={k:[] for k in keys};answer_seconds=np.zeros(3);hidden_seconds=diagnostic_seconds=0.;binary_setup_seconds=np.zeros(3);job_start=time.perf_counter();seed=job['seed']
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');base=models[seed][0]
                torch.cuda.synchronize();before=time.perf_counter();psi,hs=base.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=base.parts(t,base.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,base.zoffs,math.inf),lc[:,None]],1),1);good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1));a=base.couplings(t,psi);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                out={};offset=(ji+lo//64)%3;binary={}
                for v in list(range(3))[offset:]+list(range(3))[:offset]:
                    torch.cuda.synchronize();before=time.perf_counter()
                    if v==0:out[v]=L.solve(models[seed][v],t,a,torch.full_like(a[:,0],3.),8192,0.)
                    else:
                        setup=time.perf_counter();ad=a.double();cache=C.BinaryQuadratic(doubles[seed][v],t,ad);torch.cuda.synchronize();binary_setup_seconds[v]+=time.perf_counter()-setup;binary[v]=C.solve(cache);out[v]=tuple(binary[v][k] for k in ['prediction','sweeps','stopped','finite'])
                    torch.cuda.synchronize();answer_seconds[v]+=time.perf_counter()-before
                before=time.perf_counter()
                for k,key in enumerate(['predictions','iterations','stopped','finite']):saved[key].append(np.stack([out[v][k].cpu().numpy() for v in range(3)]))
                saved['native_final_logits'].append(np.stack([out[v][4].cpu().numpy() for v in range(1)]));ad=a.double();margins=[];energies=[];native_margins=[]
                for v in range(3):
                    model=doubles[seed][v];board=out[v][0].double();margins.append(model.flip_delta(t,board,ad).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(model.energy(t,board,ad).cpu().numpy())
                    if v==0:native_margins.append(models[seed][v].flip_delta(t,out[v][0],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy())
                saved['minimum_flip_margin64'].append(np.stack(margins));saved['energy64'].append(np.stack(energies));saved['native_minimum_flip_margin32'].append(np.stack(native_margins));saved['binary_flip_count'].append(np.stack([binary[v]['flip_count'].cpu().numpy() for v in [1,2]]));saved['binary_cache_margin'].append(np.stack([binary[v]['minimum_flip_margin'].cpu().numpy() for v in [1,2]]));saved['hidden_steps'].append(hs.cpu().numpy());saved['hidden_good'].append(good.cpu().numpy());saved['couplings'].append(a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                state.update(current_job=job,completed_inputs=lo+64,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state)
                if time.perf_counter()-started>600:
                    np.savez_compressed(run/'partial.npz',**{k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()});raise TimeoutError('Frozen600sworkerbudget')
        arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};arrays['predictions']=arrays['predictions'].astype(bool);np.savez_compressed(run/'outputs.npz',**arrays)
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':answer_seconds.tolist(),'binary_cache_setup_seconds_included':binary_setup_seconds.tolist(),'diagnostic_seconds':diagnostic_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('hard_negative_transfer_job_complete',job=job['id'],gpu=gpu,answer_seconds=answer_seconds.tolist(),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('hard_negative_transfer_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
