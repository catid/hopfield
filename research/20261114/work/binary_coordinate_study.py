"""H2aa fixed four-color binary descent versus native LP and BP PL."""
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
from gate_binary_coordinate import DEST
from lp_transfer import PP as PARENT,protocol as parent_protocol

PP=S.HERE/'binary-coordinate-protocol.json'
BASE=2026111511
ARMS=('pl','lp_native','lp_binary')
AXIS0={'hidden_steps','hidden_good','couplings','binary_flip_count','binary_cache_margin'}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=parent_protocol();gates={}
    for name in ['numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json']:
        path=DEST/name;r=json.loads(path.read_text());assert r['status']=='passed' and r['helper_sha256']==S.sha(C.__file__) and r['script_sha256']==S.sha(S.HERE/'gate_binary_coordinate.py');gates[str(path.relative_to(S.ROOT))]=S.sha(path)
    models={str(seed):{arm:parent['models'][str(seed)]['pl' if arm=='pl' else 'lp_scaled'] for arm in ARMS} for seed in range(3)}
    sources=dict(parent['source_sha256']);sources.update({str(path.relative_to(S.ROOT)):S.sha(path) for path in [S.HERE/'binary_coordinate_study.py',S.HERE/'binary_coordinate.py',S.HERE/'gate_binary_coordinate.py',S.HERE/'gate_lp_relaxation.py']})
    p={'id':'norm-matched-LP-four-color-binary-coordinate-descent','hypothesis_id':'H2aa','status':'frozen_before_fresh_generation_and_inference','frozen_utc':S.utc(),'parent_protocol_sha256':S.sha(PARENT),'arms':list(ARMS),'models':models,'source_sha256':sources,'gates':gates,
      'binary_decoder':{'initialization':'Every suppliedopen cell selected, walls0/endpoints1; no continuousprefix.','dtype':'float64; exactpromotion of frozenfloat32answerparameters andsharedfloat32readout.','colors':list(map(list,C.COLORS)),'flip_threshold':C.THRESHOLD,'sweep_cap':C.SWEEP_CAP,'update':'At eachphase simultaneouslyflip freecells ofthatparitycolor withoriginalbinaryenergydelta<-1e-8. Fourphases compriseonesweep.','stop':'Aftera fullsweep withzeroeligibleflips. Cap512withoutrescue; finalminimumdelta recorded evenifnotstopped.','mathematical_scope':'Distinctsamecolorcells have no pair/linkinteraction. Aphaseenergychange equals the sum ofindividualflipcosts. Exactsourcefixtures verifyadditivity andstrictdecrease. Finitecap successnotguaranteed.','implementation':'Precomputedbinarylinear/symmetricpaircoefficients; sourcebench2alternatingpasses verifyexactdiscreteparity withuncachedfloat64upstreamflip_delta.'},
      'controls':{'pl':'FrozenH2v BPAdamWcontinuedPL nativefloat32decoder; trainingobjective/budget differsfromLP, not equalcomputeoptimizercomparison.','lp_native':'SameLPcheckpoint/readout; nativefloat32continuousdecoder.','hidden':'Sameimmutablemu/beta, one sharedfloat32hiddenreadout permodel-input.','labels':'No graphlabel, shortestdistance, graphsearch, validationbasedselection orrepair entersinference.'},
      'cohort':{'n_unique':4096,'length':120,'loop_rate':.2,'base_seed':BASE,'generator':'study.maze SeedSequence([2026111511,120,20,index]),index0..4095','shards':16,'shard_size':256,'disjointness':'Exactcanonicaltoken digests against H2r/H2s/H2t/H2u/H2v/H2y/H2z; no pooling.'},
      'native_inference':{'dtype':'float32','hidden_tol':1e-6,'hidden_cap':4000,'hidden_check':8,'u0':3.,'eta':.5,'answer_cap':8192,'answer_check':16,'batch':64,'certificate':'Originalnativefloat32flip_delta>=-1e-6; preserve originalstopmask andbudget.'},
      'evaluation':'Success requires declareddecoderstop, finiteanswer, hiddenconvergence, commonoriginalbinaryenergyfloat64minimumflip>=-1e-6, and independentvalidshortestgraph check. The binaryderivative explicitlychangesdecoder/arithmetic; no postcapcredit, globaloptimality claim or samecomputeclaim.',
      'primary_gain':{'hypothesis':'Binarydescent improves joint-three-model validshortest success byatleast90percentagepoints versusLPnative.','target_pp':90.,'decision':'Pairedgain/loss events over4096independentinputs. Separatelyone-sided95netbounds useCPmarginalalpha.025. Supportlower>=90pp; falsifyupper<90pp; elseinconclusive.'},
      'primary_noninferiority':{'hypothesis':'Binarydescent jointvalidshortest accuracy noninferiorwithin.1percentagepoint versusBPPLnative.','margin_pp':.1,'decision':'Pairedloss=BPPLjointsuccess andbinaryjointfailure. ExactCPone-sided95lossboundsalpha.05. Supportupper<=.1pp; falsifylower>.1pp; elseinconclusive. Reportgains/net separately.'},
      'overall':'Supportifbothsupport; falsifyifeitherfalsifies; otherwiseinconclusive. Endpointsseparate, no simultaneous95claim. Modelrepeats are notindependentinputs.',
      'secondary':'Wrongcertifiedstops, timeouts/nonfinite/hiddenfailures, originalbinaryenergies/certificates, endpoint/allopenoutputs, selectedcells, individualflips, sweeps, nativeupdates, time descriptive.',
      'budget':{'total_gpu_minute_envelope':30,'per_worker_wall_seconds':900,'watchdog':'Everybatch; preservepartial onexpiry, no extension, cap/threshold/colororder changes orrescue.'},
      'timing':'Rotatedarmorder byjob/batch; CUDA-synchronized sharedhidden andperarmanswer. Binarycost includes float64readoutcast/cacheconstruction/solve. Nativecost includesoriginalsolver. Diagnostics/transfer/archive overhead separatelyinworkerwall; one scientificpass, no runtimeCI. Sourcewarmups/timingreplays separate.',
      'retained':'Allfreshinputs/labels, sharedreadouts, allpredictions/flags, twoactualnativefinal-logitarrays, iterationswithdeclaredunits(nativeupdates,binarysweeps), binaryflipcounts/cachemargin, commonfloat64energy/margins andnativefloat32margins, timings, hashes/failures.',
      'jobs':[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':(shard*3+seed)%2} for shard in range(16) for seed in range(3)]}
    S.save(PP,p);S.emit('binary_coordinate_protocol_frozen',sha256=S.sha(PP))


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
            indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8));np.savez_compressed(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]));datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('binary_coordinate_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started})


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();from cram.maze_learned import gather
    p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]};models={};doubles={}
    for seed in range(3):
        models[seed]=[];doubles[seed]=[]
        for arm in ARMS[:2]:
            model=S.model_for(seed,120,'original','cuda');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location='cuda',weights_only=True));models[seed].append(model);doubles[seed].append(copy.deepcopy(model).double())
    for ji,job in enumerate(p['jobs']):
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():raise FileExistsError(run)
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:tokens=x['tokens']
        keys=['predictions','iterations','stopped','finite','native_final_logits','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_flip_count','binary_cache_margin','hidden_steps','hidden_good','couplings'];saved={k:[] for k in keys};answer_seconds=np.zeros(3);hidden_seconds=diagnostic_seconds=binary_setup_seconds=0.;job_start=time.perf_counter();seed=job['seed']
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');base=models[seed][0]
                torch.cuda.synchronize();before=time.perf_counter();psi,hs=base.settle(t,tol=1e-6,max_steps=4000,check=8);lg,la,lc=base.parts(t,base.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,base.zoffs,math.inf),lc[:,None]],1),1);good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1));a=base.couplings(t,psi);torch.cuda.synchronize();hidden_seconds+=time.perf_counter()-before
                out={};offset=(ji+lo//64)%3;binary=None
                for v in list(range(3))[offset:]+list(range(3))[:offset]:
                    torch.cuda.synchronize();before=time.perf_counter()
                    if v<2:out[v]=L.solve(models[seed][v],t,a,torch.full_like(a[:,0],3.),8192,0.)
                    else:
                        setup=time.perf_counter();ad=a.double();cache=C.BinaryQuadratic(doubles[seed][1],t,ad);torch.cuda.synchronize();binary_setup_seconds+=time.perf_counter()-setup;binary=C.solve(cache);out[v]=(binary['prediction'],binary['sweeps'],binary['stopped'],binary['finite'])
                    torch.cuda.synchronize();answer_seconds[v]+=time.perf_counter()-before
                before=time.perf_counter()
                for k,key in enumerate(['predictions','iterations','stopped','finite']):saved[key].append(np.stack([out[v][k].cpu().numpy() for v in range(3)]))
                saved['native_final_logits'].append(np.stack([out[v][4].cpu().numpy() for v in range(2)]));ad=a.double();margins=[];energies=[];native_margins=[]
                for v in range(3):
                    model=doubles[seed][0 if v==0 else 1];board=out[v][0].double();margins.append(model.flip_delta(t,board,ad).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(model.energy(t,board,ad).cpu().numpy())
                    if v<2:native_margins.append(models[seed][v].flip_delta(t,out[v][0],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy())
                saved['minimum_flip_margin64'].append(np.stack(margins));saved['energy64'].append(np.stack(energies));saved['native_minimum_flip_margin32'].append(np.stack(native_margins));saved['binary_flip_count'].append(binary['flip_count'].cpu().numpy());saved['binary_cache_margin'].append(binary['minimum_flip_margin'].cpu().numpy());saved['hidden_steps'].append(hs.cpu().numpy());saved['hidden_good'].append(good.cpu().numpy());saved['couplings'].append(a.cpu().numpy());diagnostic_seconds+=time.perf_counter()-before
                state.update(current_job=job,completed_inputs=lo+64,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state)
                if time.perf_counter()-started>900:
                    np.savez_compressed(run/'partial.npz',**{k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()});raise TimeoutError('Frozen900sworkerbudget')
        arrays={k:np.concatenate(v,axis=0 if k in AXIS0 else 1) for k,v in saved.items()};arrays['predictions']=arrays['predictions'].astype(bool);np.savez_compressed(run/'outputs.npz',**arrays)
        r={'status':'complete','utc':S.utc(),'job':job,'n':256,'protocol_sha256':S.sha(PP),'dataset_sha256':ds['sha256'],'checkpoint_sha256_by_arm':[p['models'][str(seed)][arm]['sha256'] for arm in ARMS],'outputs_sha256':S.sha(run/'outputs.npz'),'hidden_seconds_shared':hidden_seconds,'answer_seconds_by_arm':answer_seconds.tolist(),'binary_cache_setup_seconds_included':binary_setup_seconds,'diagnostic_seconds':diagnostic_seconds,'job_seconds':time.perf_counter()-job_start,'inventory':inventory};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('binary_coordinate_job_complete',job=job['id'],gpu=gpu,answer_seconds=answer_seconds.tolist(),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);S.emit('binary_coordinate_worker_complete',gpu=gpu,seconds=state['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='evaluate':evaluate(args.gpu)
    else:globals()[args.command]()
