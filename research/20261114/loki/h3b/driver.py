"""H3b: fixed-LR no-decay Adam and paired finite-budget controls."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'research/20261114/loki/h3b'
OUT=ROOT/'artifacts/research/20261114/loki/h3b'
OLD=ROOT/'artifacts/research/20261114/loki'
UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):S.save(p,d)
def read(p):return json.loads(Path(p).read_text())
def protocol():return read(SRC/'protocol.json')
def emit(event,**kw):S.emit(event,**kw)


def prepare():
    import numpy as np
    assert not(SRC/'protocol.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    original=read(SRC.parent/'protocol.json');selection=read(OLD/'selection.json');assert selection['selected_arm']=='adamw-lr0.3'
    inp=UP/'data/gram/nqueens-10/test/all__inputs.npy'
    x=np.load(inp);unique,first=np.unique(x,axis=0,return_index=True);assert len(unique)==6513
    np.savez_compressed(OUT/'test-inputs.npz',inputs=unique,source_rows=first)
    shutil.copy2(OLD/'cohorts-n10.npz',OUT/'cohorts-n10.npz')
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SRC/'capture_solve.py',SRC.parent/'study.py',*sorted((UP/'cram').glob('*.py')),*sorted((UP/'experiments').glob('*.py'))]}
    arms={
      'adam-nodecay':dict(optimizer='adamw',lr=.3,precond=False,weight_decay=0.,label='BP Adam, no decay',implementation='torch.optim.AdamW(weight_decay=0) is Adam; direct update parity verified'),
      'adamw-decay':dict(cached_model=str((OLD/'n10/fits/adamw-lr0.3/model.pt').relative_to(ROOT)),cached_fit=str((OLD/'n10/fits/adamw-lr0.3/fit.json').relative_to(ROOT)),label='BP AdamW, decay0.01'),
      'sgd-pre':dict(cached_model=str((OLD/'n10/fits/sgd-pre/model.pt').relative_to(ROOT)),cached_fit=str((OLD/'n10/fits/sgd-pre/fit.json').relative_to(ROOT)),label='SGD + geometry preconditioner')}
    for arm in arms.values():
        if 'cached_model' in arm:arm['model_sha256']=sha(ROOT/arm['cached_model']);arm['fit_sha256']=sha(ROOT/arm['cached_fit'])
    p=dict(id='H3b-weight-decay',frozen_utc=now(),hypothesis='At fixed selectedLR0.3 and4000updates, removing AdamW weight decay closes n10N20 accuracy gap to within0.1pp and coverage gap to within2pp of frozenSGDpre.',
      antecedent='H3 selectedAdamW(.01) failed margin on development confirmation; this follow-up was proposed after those results. Officialtest is frozen here before new outcomes.',
      arms=arms,fit=original['fit'],cohorts={'10':original['cohorts']['10']},training_rows=97554,
      primary=dict(cohort='all6513unique officialtestinputs in numpy.unique lexicographic order',n=6513,seeds=[0,1,2],chains=20,cap=20000,temperature=1.,stop='local',margin=.001),
      secondary=dict(cohort='first1024of samecanonical unique testinputs',n=1024,seeds=[0,1,2],chains=20,cap=80000,prefix_cap=20000,
        purpose='Finite-budget accuracy-compute tradeoff, all3arms without selection;20kprefix captured in SAME1024input trajectory to avoid differing-batch RNG confound',
        parity='Small CPU single-block original-prefix/final parity and rejection of multi-block capture, plus one production1024input AdamW seed0 original20k comparison before using captured prefix metrics',
        block_constraint='Secondary20480chains fit inside the original65536block. Capture refuses multi-block inputs because longer earlier blocks shift the shared RNG stream in later blocks.'),
      decision=original['decision'],selection='No LR, temperature, stop, recipe or model selection on test. Report all3arms. OriginalAdamW LR0.3 transferred from frozenH3 discovery.',
      data_sha256={str(inp.relative_to(ROOT)):sha(inp),str((OUT/'test-inputs.npz').relative_to(ROOT)):sha(OUT/'test-inputs.npz'),str((OUT/'cohorts-n10.npz').relative_to(ROOT)):sha(OUT/'cohorts-n10.npz')},
      source_sha256=sources,source_commit=original['source_commit'],parent_protocol_sha256=sha(SRC.parent/'protocol.json'),
      diagnostics='After inference only: full saved-board validity, wrong stops, budget failure, PL/temperature/calibration, returned-board local-certificate-gap distributions. No diagnostics influence inference.',
      timing='Move-count cost tradeoff is primary;80k solver adds snapshot copies. No local baseline hardware-speedup claim.',
      prelaunch_amendment='v1 archived under prelaunch-v1; bounded CPU parity exposed cross-block RNG coupling before any accuracy measurement. v2 adds explicit single-block capture gate; experimental inputs, arms and budgets unchanged.')
    save(SRC/'protocol.json',p);emit('protocol_frozen',sha256=sha(SRC/'protocol.json'))


def validate():
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    from capture_solve import solve_with_prefix
    torch.set_num_threads(1)
    model=torch.load(OLD/'n10/fits/adamw-lr0.3/model.pt',weights_only=False)
    inputs=np.load(OUT/'test-inputs.npz')['inputs'][:13]==2
    # Fixture has a tiny budget and one block, without scoring accuracy.
    reference=Q.solve(model,inputs,chains=2,cap=11,seed=83,device='cpu',stop='local',temperature=1.,block=128)
    prefix,final=solve_with_prefix(model,inputs,chains=2,cap=37,prefix_cap=11,seed=83,device='cpu',stop='local',temperature=1.,block=128)
    equal=[np.array_equal(a,b) for a,b in zip(reference,prefix)];assert all(equal)
    reference_final=Q.solve(model,inputs,chains=2,cap=37,seed=83,device='cpu',stop='local',temperature=1.,block=128)
    final_equal=[np.array_equal(a,b) for a,b in zip(reference_final,final)];assert all(final_equal)
    try:solve_with_prefix(model,inputs,chains=2,cap=37,prefix_cap=11,seed=83,device='cpu',stop='local',temperature=1.,block=8)
    except AssertionError as exc:assert 'one block' in str(exc)
    else:raise AssertionError('Multi-block capture was not rejected')
    a=torch.tensor([.4,-.8],dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
    x=torch.optim.Adam([a],lr=.3,weight_decay=0.,foreach=False);y=torch.optim.AdamW([b],lr=.3,weight_decay=0.,foreach=False)
    for g in ([.2,.7],[-.3,.4],[.6,-.8]):a.grad=torch.tensor(g,dtype=torch.float64);b.grad=a.grad.clone();x.step();y.step();assert torch.equal(a,b)
    save(OUT/'validation.json',dict(status='passed',utc=now(),protocol_sha256=sha(SRC/'protocol.json'),prefix_fields_equal=equal,final_fields_equal=final_equal,
      field_names=['boards','stopped','energy','moves','flop'],multi_block_capture_rejected=True,adamw_zero_decay_bitwise_equal_adam=True,accuracy_measured=False,
      sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SRC/'capture_solve.py',SRC.parent/'study.py']}))
    emit('validation_passed')


def load_model(arm):
    import torch
    cfg=protocol()['arms'][arm]
    if 'cached_model' in cfg:
        assert sha(ROOT/cfg['cached_model'])==cfg['model_sha256']
        return torch.load(ROOT/cfg['cached_model'],weights_only=False),cfg['cached_model']
    path=OUT/'n10/fits/adam-nodecay/model.pt'
    rec=read(path.parent/'fit.json');assert sha(path)==rec['model_sha256']
    return torch.load(path,weights_only=False),str(path.relative_to(ROOT))


def score_arrays(inputs, arrays, arm, stage, seed, suffix, seconds, model_path):
    import numpy as np
    from cram import nqueens_data as D
    boards,stopped,energy,moves,flop=arrays;ok=S.independent_ok(boards,inputs,10)
    all_solutions=np.asarray(D.completions(np.zeros(100,bool),10),bool)
    counts=np.array([all_solutions[:,row].all(1).sum() for row in inputs]);denom=np.minimum(counts,20)
    found=np.array([min(len({b.tobytes() for b in boards[i,ok[i]]}),denom[i]) for i in range(len(inputs))])
    n20=ok[np.arange(len(inputs)),energy.argmin(1)]
    dest=OUT/f'{stage}/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{suffix}.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,moves=moves,flop=flop,valid=ok,n20=n20,n1=ok[:,0],found=found,denom=denom,solution_counts=counts)
    result=dict(status='complete',utc=now(),arm=arm,stage=stage,seed=seed,suffix=suffix,unique_inputs=len(inputs),
      n20_accuracy_pct=float(n20.mean()*100),n1_accuracy_pct=float(ok[:,0].mean()*100),coverage_pct=float(found.sum()/denom.sum()*100),
      wrong_stops=int((stopped&~ok).sum()),budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),search_seconds=seconds,
      records_sha256=sha(dest/f'{suffix}.npz'),model_path=model_path,model_sha256=sha(ROOT/model_path),protocol_sha256=sha(SRC/'protocol.json'),
      source_sha256=sha(Path(__file__)),capture_source_sha256=sha(SRC/'capture_solve.py'))
    save(dest/f'{suffix}.json',result);return result


def job(arm,stage,seed,progress):
    import numpy as np
    from cram import cram_nqueens as Q
    from capture_solve import solve_with_prefix
    p=protocol();model,path=load_model(arm);inputs=np.load(OUT/'test-inputs.npz')['inputs'][:p[stage]['n']]==2
    progress(stage=stage,arm=arm,seed=seed,inputs=len(inputs));emit('search_start',arm=arm,stage=stage,seed=seed,inputs=len(inputs))
    settings=dict(chains=20,seed=seed,device='cuda',stop='local',temperature=1.,margin=.001)
    start=time.perf_counter()
    if stage=='primary':
        arrays=Q.solve(model,inputs,cap=20000,**settings);seconds=time.perf_counter()-start
        result=score_arrays(inputs,arrays,arm,stage,seed,'final',seconds,path)
    else:
        prefix,final=solve_with_prefix(model,inputs,cap=80000,prefix_cap=20000,**settings);seconds=time.perf_counter()-start
        if arm=='adamw-decay' and seed==0:
            ref=Q.solve(model,inputs,cap=20000,**settings)
            equal=[np.array_equal(a,b) for a,b in zip(ref,prefix)];assert all(equal)
            save(OUT/'production-prefix-parity.json',dict(status='passed',utc=now(),arm=arm,seed=seed,n=1024,chains=20,prefix_cap=20000,fields_equal=equal,
              protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(SRC/'capture_solve.py')))
        score_arrays(inputs,prefix,arm,stage,seed,'prefix20k',None,path)
        result=score_arrays(inputs,final,arm,stage,seed,'final',seconds,path)
    emit('search_complete',**{k:result[k] for k in ['arm','stage','seed','n20_accuracy_pct','coverage_pct','wrong_stops','budget_failures','mean_moves','search_seconds']})


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CRAM_DATA=str(UP/'data'))
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *a:subprocess.check_output(a,text=True,timeout=20).strip()
    assert query('hostname')=='loki'
    dev=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');uuid=dev.split(',')[1].strip()
    assert uuid not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    p=protocol();assert read(OUT/'validation.json')['status']=='passed'
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in p['data_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in p['cohorts']['10']['data_sha256'].items():assert sha(ROOT/path)==h,path
    receipt=OUT/f'worker-g{gpu}.json';base=dict(status='running',started_utc=now(),pid=os.getpid(),gpu=gpu,device=dev,host='loki',protocol_sha256=sha(SRC/'protocol.json'))
    def progress(**v):save(receipt,dict(base,updated_utc=now(),progress=v))
    progress(stage='starting')
    try:
        if gpu==0:
            S.OUT=OUT;S.SRC=SRC;S.protocol=protocol
            model,fit=S.fit_model(10,'adam-nodecay',progress)
        jobs=[(arm,stage,seed) for stage in ('primary','secondary') for seed in (0,1,2) for arm in ('adamw-decay','sgd-pre','adam-nodecay')]
        while True:
            pending=False;worked=False
            for arm,stage,seed in jobs:
                dest=OUT/f'{stage}/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'final.json').exists():continue
                pending=True
                if arm=='adam-nodecay' and not(OUT/'n10/fits/adam-nodecay/fit.json').exists():continue
                claim=(dest/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if not(dest/'final.json').exists():job(arm,stage,seed,progress);worked=True
                finally:claim.close()
            if not pending:break
            if not worked:time.sleep(1)
        save(receipt,dict(base,status='complete',completed_utc=now()));emit('worker_complete',gpu=gpu)
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True);sp.add_parser('prepare');sp.add_parser('validate');w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare()
    elif a.mode=='validate':validate()
    else:worker(a.gpu)
