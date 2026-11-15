"""H3p: prospective power confirmation with unchanged H3o models."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3p';OUT=ROOT/'artifacts/research/20261114/loki/h3p';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent/'h3i'),str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')


def prepare():
    assert not(SRC/'protocol.json').exists();parent=read(SRC.parent/'h3o/protocol.json');dataset=read(OUT/'dataset.json');base=ROOT/'artifacts/research/20261114/loki/h3o'
    sources=[Path(__file__),SRC/'prepare.py',SRC/'reuse_identities.py',SRC.parent/'h3o/driver.py',SRC.parent/'h3o/enumerate.py',SRC.parent/'h3i/backend.py',SRC.parent/'h3l/enumerate.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py'))]
    data_files=[OUT/'training.npz',OUT/'data.npz',OUT/'dataset.json',OUT/'feasibility.json',OUT/'input-universe.npz',SRC.parent/'h3o/protocol.json',base/'data.npz',base/'source-support.npz',base/'source-fits.json',base/'energy-groups.json']
    for cfg in parent['arms']:
        data_files.extend(base/f"models/{cfg['arm']}"/name for name in ['model.pt','model.json','construction.npz'])
    p=dict(parent,id='H3p-upward-transfer-prospective-confirmation',frozen_utc=now(),
        antecedent='H3o accuracy noninferiority was inconclusive. This is one prospective fixed-size confirmation on disjoint inputs, retaining the unchanged hypotheses/settings and reporting separately; no repeated optional confirmation is authorized by this protocol.',
        construction='Copy exact H3o model.pt and construction.npz files, source counts and fit receipts; verify all solver parameters and input/settings identity. No retraining or refitting.',
        search=dict(parent['search'],unique_inputs=16384),dataset=dataset,
        budget=dict(logical_model_seed_jobs=6,maximum_unique_search_jobs=6,gpu_minutes_envelope=25,optimizer_fits=0),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},data_sha256={str(path.relative_to(ROOT)):sha(path) for path in data_files},
        limits=parent['limits']+['This one prospective confirmation is reported separately from H3o. Its sample size was chosen after H3o, but no confirmation outcomes were inspected before protocol freezing.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))


def construct():
    import shutil
    p=protocol();base=ROOT/'artifacts/research/20261114/loki/h3o';receipts=[]
    for name in ['source-support.npz','source-fits.json','energy-groups.json']:shutil.copy2(base/name,OUT/name)
    for cfg in p['arms']:
        arm=cfg['arm'];source=base/f'models/{arm}';dest=OUT/f'models/{arm}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['model.pt','construction.npz']:shutil.copy2(source/name,dest/name)
        rec=dict(read(source/'model.json'),utc=now(),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'),
            derivation='Copied exact frozen H3o model and construction files; no new fit.',original_model_receipt=str((source/'model.json').relative_to(ROOT)),original_model_receipt_sha256=sha(source/'model.json'))
        assert rec['model_sha256']==sha(dest/'model.pt') and rec['construction_sha256']==sha(dest/'construction.npz')
        save(dest/'model.json',rec);receipts.append(rec)
    save(OUT/'construction-summary.json',dict(status='complete',models=receipts));print(json.dumps(dict(status='copied_exact_H3o_models',models=len(receipts))))


def super_valid(boards,inputs):
    import numpy as np
    valid=S.independent_ok(boards,inputs,15);i,j=np.triu_indices(225,1);dr=np.abs(j//15-i//15);dc=np.abs(j%15-i%15)
    knight=((dr==1)&(dc==2))|((dr==2)&(dc==1))
    return valid&~(boards[...,i[knight]]&boards[...,j[knight]]).any(-1)


def validate_backend(device):
    import numpy as np
    import torch
    import backend
    from cram import cram_nqueens as Q
    # Synthetic Ising fixtures only; no N15 target solution enters validation.
    c=225;model=dict(S=torch.zeros(c,c,2,2),B=torch.zeros(c,2),q=15,delta=14.,tau=-105.,unit=protocol()['calibration']['unit'])
    rng=np.random.default_rng(83114);inputs=np.zeros((16,c),bool)
    for k,row in enumerate(inputs):row[rng.choice(c,2+k%3,replace=False)]=True
    kwargs=dict(chains=3,cap=64,seed=17,device=device,temperature=1.,stop='local',margin=.001)
    start=time.perf_counter();a=Q.solve(model,inputs,**kwargs);b=backend.solve(model,inputs,**kwargs)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    model['S'][:,:,1,1]=14*(1-torch.eye(c));given=np.zeros((1,c),bool);given[0,:3]=True;kwargs.update(chains=1,cap=1)
    original=Q.solve(model,given,**kwargs);gated=backend.solve(model,given,**kwargs)
    assert original[1].all() and not gated[1].any() and original[3].item()==0 and gated[3].item()==1
    given[0,:15]=True;original=Q.solve(model,given,**kwargs);gated=backend.solve(model,given,**kwargs)
    assert all(np.array_equal(x,y) for x,y in zip(original,gated)) and gated[1].all()
    rec=dict(status='passed',utc=now(),device=device,synthetic_inputs=16,chains=3,cap=64,all_five_backend_outputs_identical=True,
        synthetic_three_clue_false_certificate_rejected=True,synthetic_q_gate_acceptance_checked=True,seconds=time.perf_counter()-start,
        backend_sha256=sha(SRC.parent/'h3i/backend.py'),reference_sha256=sha(UP/'cram/cram_nqueens.py'),source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'))
    save(OUT/f'backend-validation-{device}.json',rec);print(json.dumps(rec))


def search(cfg,seed,data,progress):
    import numpy as np
    import torch
    import backend
    arm=cfg['arm'];path=OUT/f'models/{arm}/model.pt';model=torch.load(path,weights_only=False);inputs=data['test_inputs']
    progress(stage='search',arm=arm,seed=seed);S.emit('search_start',arm=arm,seed=seed)
    start=time.perf_counter();boards,stopped,energy,moves,flop=backend.solve(model,inputs,chains=20,cap=20000,seed=seed,device='cuda',temperature=1.,stop='local',margin=.001)
    seconds=time.perf_counter()-start;valid=super_valid(boards,inputs);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    original=energy.argmin(1);selected=[]
    for e,s in zip(energy,stopped):
        pool=np.flatnonzero(s)
        if not len(pool):pool=np.arange(20)
        selected.append(int(pool[e[pool].argmin()]))
    selected=np.array(selected);ids=np.arange(len(inputs));n20=valid[ids,original];cert=valid[ids,selected]
    dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'records.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,moves=moves,flop=flop,
        cardinality_integer_additions=224*(moves+1),valid=valid,n20=n20,n20_certificate=cert,n1=valid[:,0],found=found,denom=denom,
        solution_counts=data['test_solution_counts'],original_index=original,certificate_index=selected,queen_counts=boards.sum(-1))
    rec=dict(status='complete',utc=now(),arm=arm,seed=seed,unique_inputs=len(inputs),
        original_accuracy_pct=100*float(n20.mean()),certificate_accuracy_pct=100*float(cert.mean()),coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),mean_moves=float(moves.mean()),search_seconds=seconds,
        model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),backend_sha256=sha(SRC.parent/'h3i/backend.py'))
    save(dest/'result.json',rec);S.emit('search_complete',**{k:rec[k] for k in ['arm','seed','certificate_accuracy_pct','coverage_pct','wrong_stops','search_seconds']})


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    assert query('hostname')=='loki';device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;p=protocol()
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    validation=read(OUT/'backend-validation-cuda.json');assert validation['status']=='passed' and validation['backend_sha256']==sha(SRC.parent/'h3i/backend.py')
    for cfg in p['arms']:
        dest=OUT/f"models/{cfg['arm']}";rec=read(dest/'model.json');assert rec['model_sha256']==sha(dest/'model.pt') and rec['source_sha256']==sha(__file__)
    data=dict(np.load(OUT/'data.npz'));started=time.monotonic()
    base=dict(status='running',started_utc=now(),gpu=gpu,pid=os.getpid(),host='loki',device=device,protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,updated_utc=now(),progress=values))
    progress(stage='starting')
    try:
        for cfg in p['arms']:
            for seed in [0,1,2]:
                dest=OUT/f"search/{cfg['arm']}/s{seed}";dest.mkdir(parents=True,exist_ok=True)
                if (dest/'result.json').exists():continue
                claim=(dest/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if time.monotonic()-started>750:raise RuntimeError('25GPU-minute campaign envelope reached before newjob; partial evidence retained')
                    if not(dest/'result.json').exists():search(cfg,seed,data,progress)
                finally:claim.close()
        save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True)
    sp.add_parser('prepare');sp.add_parser('construct');v=sp.add_parser('validate');v.add_argument('--device',choices=['cpu','cuda'],required=True)
    w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare()
    elif a.mode=='construct':construct()
    elif a.mode=='validate':validate_backend(a.device)
    else:worker(a.gpu)
