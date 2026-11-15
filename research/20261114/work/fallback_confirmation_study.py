"""H2bk fresh loop80 confirmation, fixed cohort and bounded GPU workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import fast_npz as Z
import confirmation96_batch as Q
from fallback_confirmation_job import run_job

PP=S.HERE/'fallback-confirmation-protocol.json';DEST=S.OUT/'fallback_confirmation';BASE=2026111528
ARMS=Q.ARMS;REPORTED=('pl','unified96','hybrid','dual_certificate')


def protocol():
    p=json.loads(PP.read_text())
    for field in ['source_sha256','source_evidence_sha256']:
        for path,h in p[field].items():assert S.sha(S.ROOT/path)==h,path
    for group in p['models'].values():
        for model in group.values():assert S.sha(S.ROOT/model['path'])==model['sha256']
    return p


def load_models(p,seed,device='cuda'):
    torch,_,_=S.imports();models=[];doubles=[];times={}
    for arm in ARMS[:2]:
        if device=='cuda':torch.cuda.synchronize()
        before=time.perf_counter();m=S.model_for(seed,180,'original',device);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location=device,weights_only=True));double=copy.deepcopy(m).double()
        if device=='cuda':torch.cuda.synchronize()
        times[arm]=time.perf_counter()-before;models.append(m);doubles.append(double)
    for k,v in models[0].state_dict().items():
        if k not in ['T','b','kappa']:assert torch.equal(v,models[1].state_dict()[k]),k
    return models,doubles,times


def make_one(index):return S.maze(180,.8,np.random.SeedSequence([BASE,180,80,index]))


def prepare():
    p=protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();datasets=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for shard in range(16):
            path=DEST/f'shard{shard:02d}.npz';assert not path.exists();indices=np.arange(shard*256,(shard+1)*256);items=list(pool.map(make_one,indices,chunksize=8))
            Z.save(path,tokens=np.stack([v[0] for v in items]),labels=np.stack([v[1] for v in items]),shortest_edges=np.array([v[2] for v in items]),indices=indices,opened_loops=np.array([v[3] for v in items]),eligible_walls=np.array([v[4] for v in items]))
            datasets.append({'shard':shard,'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path),'n':256});S.emit('fallback_confirmation_data',shard=shard,inputs=(shard+1)*256)
    S.save(DEST/'datasets.json',{'protocol_sha256':S.sha(PP),'datasets':datasets,'seconds':time.perf_counter()-started,'gate_sha256':S.sha(DEST/'numerical-gate.json')})


def extract_models():
    p=protocol();records=[]
    for seed in range(3):
        models,_,_=load_models(p,seed,'cpu')
        for arm,m in zip(ARMS[:2],models):
            path=DEST/f'model-seed{seed}-{arm}.npz';Z.save(path,theta=np.concatenate([m.T.numpy().ravel(),m.b.numpy(),m.kappa.numpy()]),mu_offsets=np.array(m.mu_offsets),aoffs=np.array(m.aoffs));records.append({'seed':seed,'arm':arm,'path':path.name,'sha256':S.sha(path),'checkpoint':p['models'][str(seed)][arm]})
    S.save(DEST/'model-extraction.json',{'status':'complete','protocol_sha256':S.sha(PP),'records':records,'script_sha256':S.sha(__file__)})


def evaluate(gpu):
    started=time.perf_counter();lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP)
    inp=json.loads((DEST/'independent-input-audit.json').read_text());assert inp['status']=='passed' and inp['protocol_sha256']==S.sha(PP)
    p={**p,'datasets':manifest['datasets']};state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[],'watchdog_checks':0};last=[0.]
    def watch():
        import resource
        elapsed=time.perf_counter()-started;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;state['watchdog_checks']+=1;state['peak_rss_kib']=rss
        if elapsed>600 or rss>8*1024**2:
            state.update(status='budget_exhausted',wall_seconds=elapsed);S.save(DEST/f'eval-worker{gpu}.json',state);raise RuntimeError('Frozen600s/8GiB worker cap; no extension')
        if elapsed-last[0]>=1:state.update(wall_seconds=elapsed,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state);last[0]=elapsed
    watch();models={};doubles={};loads={}
    for seed in range(3):models[seed],doubles[seed],loads[str(seed)]=load_models(p,seed)
    state['model_load_seconds_by_seed']=loads;state['additional_bp_load_seconds_charged_once']=sum(x['pl'] for x in loads.values());watch()
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
        state['current_job']=job;watch();run_job(p,job,tokens,models[job['seed']],doubles[job['seed']],DEST/'runs'/job['id'],S.sha(PP),watch)
        state['jobs_complete'].append(job['id']);watch();S.emit('fallback_confirmation_job_complete',job=job['id'],gpu=gpu,seconds=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'eval-worker{gpu}.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','prepare','extract_models','evaluate']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    try:
        if args.command=='freeze':
            from freeze_fallback_confirmation import freeze
            freeze()
        elif args.command=='evaluate':evaluate(args.gpu)
        else:globals()[args.command]()
    except BaseException as error:
        import traceback
        S.save(DEST/f'failure-{args.command}-{args.gpu}.json',{'status':'incomplete','error':repr(error),'traceback':traceback.format_exc(),'partial_evidence_retained':True});raise
