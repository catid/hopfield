"""Nested complete-pattern holdout with bank-incompatible evaluation inputs."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3g';OUT=ROOT/'artifacts/research/20261114/loki/h3g';UP=ROOT/'upstream'
sys.path[:0]=[str(SRC.parent),str(UP),str(UP/'experiments')]
import study as S
spec=importlib.util.spec_from_file_location('n12_aggregated',SRC.parent/'h3e/driver.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):S.save(path,value)
def protocol():return read(SRC/'protocol.json')
def emit(event,**values):S.emit(event,**values)


def prepare():
    assert not(SRC/'protocol.json').exists()
    data=read(OUT/'dataset.json');assert data['status']=='reserved_and_validated_before_fits'
    models=[]
    for bank in [512,32,128]:
        arms=['adamw-decay','adam-nodecay','sgd-pre','exclusion'] if bank==512 else ['adam-nodecay','sgd-pre','exclusion']
        for arm in arms:
            models.append(dict(bank=bank,arm=arm,updates=0 if arm=='exclusion' else 250 if arm=='adam-nodecay' else 4000,
                lr=.3 if arm.startswith('adam') else 30. if arm=='sgd-pre' else None,
                weight_decay=.01 if arm=='adamw-decay' else 0.,precond=arm=='sgd-pre'))
    parent=read(SRC.parent/'h3e/protocol.json')
    sources=[Path(__file__),SRC/'prepare.py',SRC.parent/'study.py',SRC.parent/'h3e/driver.py',*sorted((UP/'cram').glob('*.py')),*sorted((UP/'experiments').glob('*.py'))]
    p=dict(id='H3g-solution-pattern-holdout',frozen_utc=now(),
        hypothesis='At the fixed512-pattern bank, training-count exclusion is noninferior to learned Adam250 on new partial inputs whose every valid completion lies outside the training bank: certificate-first accuracy within0.1pp and coverage within2pp.',
        antecedent='H3f count and retrieval controls exposed complete-pattern overlap. H3g changes the split before fitting new models and prevents training-bank retrieval by construction.',
        dataset=data,models=models,primary=dict(bank=512,arm='exclusion',control='adam-nodecay',control_updates=250,
            accuracy_margin_pp=.1,coverage_margin_pp=2.,selection='certificate-first, original lowest-energy fallback and first-index tie-break'),
        fit=dict(parent['fit'],updates='250Adam,4000SGD/AdamW; all retain4000cosine denominator',
            calibration='Only selected bank patterns, each with298training-row multiplicity. No outside pattern enters fitting, count construction, unit, delta or tau.',
            labels='All298 raw1/2/3clue masks per selected complete pattern; exact restricted-bank label expansion, not the earlier all14200pattern expansion.'),
        count='q inferred from training pattern cardinality; forbidden offset iff zero positive training co-occurrence; fixed J=q*forbidden-1, unary0. Allowed-pair reward is an explicit fixed energy-design prior, and offset sharing supplies geometry.',
        search=parent['search'],decision=parent['decision'],
        secondary='All32/128bank models and512AdamW/SGD controls, original-selection metrics, unseen completion counts, stopped/capped invalid occupancies, inferred support diagnostics. No model or hyperparameter selected on test.',
        bank_control='A compatible-training-bank baseline has zero possible answers for every test input at every nested bank size, verified by exact matching before fits and during audit. Report0% accuracy/coverage as structural retrieval failure, without manufacturing search outputs.',
        validation='Data reserved before any fit:8192unique inputs,0one-clue/16two-clue/8176three-clue; no compatibility with any512bank pattern, no overlap with any expanded fit input, no overlap with priorH3e test. All true completions outside largestbank. Reconstruct free-cell sufficient statistics from every labeled row; inherited aggregatedPL path has prior independent loss/gradient validation. All saved outputs independently checked against14200solutions and the largest-bank exclusion.',
        budget=dict(gpu_search_jobs=30,bank_sizes=[32,128,512],gpu_minutes_envelope=35,training_fits=7,count_constructions=3,
            policy='Two exclusiveGPUworkers, dynamicclaims. Check the17.5min per-worker envelope before launchinganotherjob; retainpartialresults andreportincomplete ifexceeded, withoutsilentrecipe/cap changes.'),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        data_sha256={str((OUT/name).relative_to(ROOT)):sha(OUT/name) for name in ['data.npz','dataset.json']},
        inherited_optimization_validation={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'artifacts/research/20261114/loki/h3e/cpu-validation.json',ROOT/'artifacts/research/20261114/loki/h3e/gpu-validation.json']},
        limits=['The test distribution is explicitly conditioned on incompatibility with the512bank and excludes priorH3e testinputs; it is not the original GRAM distribution.',
                'Complete patterns are unseen relative to every new fit bank, although earlier H3e/H3f studies examined the full solution universe.',
                'Nested banks come from one fixed random pattern permutation, not repeated independent training samples.',
                'No one-clue input survives this conditioning; only16two-clue inputs appear. Primary metrics pool the declared conditioned distribution.',
                'Count support plus fixed energy and geometry is a different prior from optimized coefficients. No claim of learned-weight recovery.',
                'Certificate-first changes the original paper output policy; original-selection outcomes remain visible.'])
    save(SRC/'protocol.json',p);emit('protocol_frozen',sha256=sha(SRC/'protocol.json'))


def ready(A,b,targets):
    import torch
    import nqueens_offset as O
    from cram import cram_nqueens as Q
    with torch.no_grad():
        W,B=O.pair_tables(A.detach(),b.detach(),'ising',12);energy=Q.full_S(W,Q.pair_index(144,A.device),144)
        y=targets.long();g=Q.fields(energy,B,y);diff=g[:,:,1]-g[:,:,0]
        return dict(A=A.detach().cpu(),b=b.detach().cpu(),W=W.cpu(),B=B.cpu(),S=energy.cpu(),
            tau=float(Q.board_energy(energy,B,y).max()),unit=float(diff.abs().mean()),
            delta=float(((1-2*targets)*diff).min()),arm='ising')


def build(cfg,data,progress):
    import numpy as np
    import torch
    import nqueens_offset as O
    from cram import cram_nqueens as Q
    bank=cfg['bank'];arm=cfg['arm'];indices=data[f'bank{bank}_indices'];labels=data['all_solutions'][indices]
    torch.cuda.synchronize();start=time.perf_counter()
    targets=torch.tensor(labels,device='cuda',dtype=torch.float32)
    weights=torch.tensor(data[f'bank{bank}_free_counts'],device='cuda',dtype=torch.float32);weights/=weights.sum()
    idx,off=O.offsets(12,'cuda');i,j=Q.pair_index(144,'cuda')
    if arm=='exclusion':
        cooccur=targets.T@targets;counts=torch.zeros(264,device='cuda').index_add_(0,idx,cooccur[i,j])
        q_values=np.unique(labels.sum(1));assert len(q_values)==1
        forbidden=counts==0;A=(int(q_values[0])*forbidden.float()-1);b=torch.zeros(1,device='cuda')
        objective=None
    else:
        A=torch.zeros(264,device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True)
        params=[A,b];buffers=[torch.zeros_like(x) for x in params]
        counts=torch.bincount(idx).float();pre=counts/counts.mean()
        optimizer=torch.optim.AdamW(params,lr=.3,weight_decay=cfg['weight_decay'],foreach=False) if arm.startswith('adam') else None
        for k in range(cfg['updates']):
            objective=E.loss(A,b,targets,weights,12);grads=torch.autograd.grad(objective,params)
            lr=cfg['lr']*.5*(1+math.cos(math.pi*k/4000))
            if optimizer is not None:
                optimizer.param_groups[0]['lr']=lr
                for par,g in zip(params,grads):par.grad=g
                optimizer.step()
            else:
                grads=(grads[0]/pre,grads[1])
                with torch.no_grad():
                    for par,g,buf in zip(params,grads,buffers):buf.mul_(.5).add_(g);par.sub_(lr*buf)
            if (k+1)%250==0:progress(stage='fit',bank=bank,arm=arm,update=k+1,loss=float(objective.detach()))
    model=ready(A,b,targets);torch.cuda.synchronize();seconds=time.perf_counter()-start
    assert all(torch.isfinite(model[key]).all() for key in ['A','b','S'])
    dest=OUT/f'models/b{bank}/{arm}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
    rec=dict(status='complete',utc=now(),bank=bank,arm=arm,updates=cfg['updates'],expanded_rows=bank*298,
        final_loss=float(objective.detach()) if objective is not None else None,unit=model['unit'],delta=model['delta'],tau=model['tau'],
        fit_or_construct_seconds=seconds,model_sha256=sha(dest/'model.pt'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__))
    if arm=='exclusion':
        assert model['delta']>0
        offsets=off.cpu().numpy();mask=forbidden.cpu().numpy();attack=(offsets[:,0]==0)|(offsets[:,1]==0)|(np.abs(offsets[:,0])==np.abs(offsets[:,1]))
        rec['posthoc_support']=dict(forbidden=int(mask.sum()),false_forbidden=int((mask&~attack).sum()),missed_forbidden=int((~mask&attack).sum()),
                                   rule_use='Only after construction for diagnosis; no geometry rules choose coefficients.')
        np.savez_compressed(dest/'support.npz',counts=counts.cpu().numpy(),offsets=offsets,forbidden=mask)
        rec['support_sha256']=sha(dest/'support.npz')
    save(dest/'model.json',rec);emit('model_complete',bank=bank,arm=arm,seconds=seconds)


def search(cfg,seed,data,progress):
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    bank=cfg['bank'];arm=cfg['arm'];path=OUT/f'models/b{bank}/{arm}/model.pt';model=torch.load(path,weights_only=False)
    inputs=data['test_inputs'];progress(stage='search',bank=bank,arm=arm,seed=seed,inputs=len(inputs));emit('search_start',bank=bank,arm=arm,seed=seed)
    start=time.perf_counter()
    boards,stopped,energy,moves,flop=Q.solve(model,inputs,chains=20,cap=20000,seed=seed,device='cuda',temperature=1.,stop='local',margin=.001)
    seconds=time.perf_counter()-start;valid=S.independent_ok(boards,inputs,12);denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in samples[ok]}),d) for samples,ok,d in zip(boards,valid,denom)])
    original,selected=E.select(energy,stopped);ids=np.arange(len(inputs));n20=valid[ids,original];cert=valid[ids,selected];queens=boards.sum(-1)
    dest=OUT/f'search/b{bank}/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'records.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,moves=moves,flop=flop,
        valid=valid,n20=n20,n20_certificate=cert,n1=valid[:,0],found=found,denom=denom,solution_counts=data['test_solution_counts'],
        original_index=original,certificate_index=selected,queen_counts=queens)
    rec=dict(status='complete',utc=now(),bank=bank,arm=arm,updates=cfg['updates'],seed=seed,unique_inputs=len(inputs),
        original_accuracy_pct=100*float(n20.mean()),certificate_accuracy_pct=100*float(cert.mean()),coverage_pct=100*float(found.sum()/denom.sum()),
        wrong_stops=int((stopped&~valid).sum()),capped_chains=int((~stopped).sum()),mean_moves=float(moves.mean()),search_seconds=seconds,
        model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__))
    save(dest/'result.json',rec);emit('search_complete',**{key:rec[key] for key in ['bank','arm','seed','original_accuracy_pct','certificate_accuracy_pct','coverage_pct','search_seconds']})


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    assert query('hostname')=='loki';device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.empty(1,device='cuda');torch.cuda.synchronize()
    p=protocol()
    for group in ['source_sha256','data_sha256','inherited_optimization_validation']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    data=dict(np.load(OUT/'data.npz'));started=time.monotonic()
    base=dict(status='running',started_utc=now(),gpu=gpu,host='loki',pid=os.getpid(),device=device,protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,updated_utc=now(),progress=values))
    progress(stage='starting')
    try:
        while True:
            pending=worked=False
            for cfg in p['models']:
                bank=cfg['bank'];arm=cfg['arm'];dest=OUT/f'models/b{bank}/{arm}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'model.json').exists():continue
                pending=True;claim=(dest/'model.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if not(dest/'model.json').exists():build(cfg,data,progress);worked=True
                finally:claim.close()
            for cfg in p['models']:
                for seed in [0,1,2]:
                    bank=cfg['bank'];arm=cfg['arm'];dest=OUT/f'search/b{bank}/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'result.json').exists():continue
                    pending=True
                    if not(OUT/f'models/b{bank}/{arm}/model.json').exists():continue
                    claim=(dest/'job.lock').open('a')
                    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:claim.close();continue
                    try:
                        if time.monotonic()-started>1050:raise RuntimeError('35GPU-minute campaign envelope reached before newjob; partial evidence retained')
                        if not(dest/'result.json').exists():search(cfg,seed,data,progress);worked=True
                    finally:claim.close()
            if not pending:break
            if not worked:time.sleep(1)
        save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True);sp.add_parser('prepare');w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    else:worker(args.gpu)
