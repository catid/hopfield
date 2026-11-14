"""n12 fixed-recipe transfer with exact PL sufficient statistics."""
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

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3e'
OUT = ROOT / 'artifacts/research/20261114/loki/h3e'
UP = ROOT / 'upstream'
sys.path[:0] = [str(SRC.parent), str(UP), str(UP / 'experiments')]
import study as S


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): S.save(path, value)
def protocol(): return read(SRC / 'protocol.json')
def emit(event, **values): S.emit(event, **values)


def prepare():
    assert not (SRC / 'protocol.json').exists()
    data = read(OUT / 'dataset.json'); assert data['status'] == 'prepared_and_independently_validated'
    sources = [Path(__file__), SRC / 'prepare.py', SRC.parent / 'study.py', SRC.parent / 'h3d/driver.py',
               *sorted((UP / 'cram').glob('*.py')), *sorted((UP / 'experiments').glob('*.py'))]
    p = dict(id='H3e-n12-transfer', frozen_utc=now(),
        hypothesis='On the new disjoint n12 input split, no-decay Adam at the fixed250-update prefix with certificate-first selection is noninferior to preconditioned SGD4000: accuracy within0.1pp and pooled distinct-completion coverage within2pp.',
        antecedent='n10 H3c/H3d motivate the fixed250-update and certificate-first combination. No n12 fit or search outcome was seen before this protocol.',
        dataset=data, data_path=str((OUT / 'data.npz').relative_to(ROOT)),
        arms={'adamw-decay': dict(lr=.3, optimizer='adamw', weight_decay=.01, precond=False, snapshots=[4000]),
              'adam-nodecay': dict(lr=.3, optimizer='adamw', weight_decay=0., precond=False, snapshots=[250,1000,4000]),
              'sgd-pre': dict(lr=30., optimizer='sgd', weight_decay=0., precond=True, snapshots=[4000])},
        fit=dict(method='exact full-row pseudolikelihood', energy='offset-shared Ising,264 couplings+1unary', updates=4000,
            cosine_denominator=4000, initialization='zero', adam_betas=[.9,.999], adam_eps=1e-8, sgd_momentum=.5,
            tf32=False, dtype='float32', L1=0., seed=0,
            objective='sum(solution,cell) free_counts*binary_NLL / sum(free_counts); exactly the expanded2,049,002rowPL objective in real arithmetic.',
            calibration='tau and delta from all training solution states; temperature unit weighted by their full expanded-row multiplicities.',
            implementation='Binary Ising fields and exact solution/free-cell sufficient statistics replace redundant two-state/full-row arithmetic. Floating-point reduction order differs; this is declared and independently checked, not claimed bitwise identical.'),
        primary=dict(arm='adam-nodecay', updates=250, control='sgd-pre', control_updates=4000,
            selection='lowest energy among stopped chains if any, else original lowest-energy return; first-index tie break',
            accuracy_margin_pp=.1, coverage_margin_pp=2.),
        search=dict(n=12, unique_inputs=8192, chains=20, cap=20000, seeds=[0,1,2], temperature=1., stop='local', margin=.001,
            validation='Search uses learned weights/statistics only. Exact completion set and puzzle rules are evaluation-only.'),
        decision=dict(rule='Support only if exact one-sided95% bound on input fraction with any paired accuracy regression <0.001 and paired-input bootstrap lower coverage difference>-2pp. Falsify if bootstrap upper accuracy difference<-.1pp or upper coverage difference<-2pp; otherwise inconclusive.',
            bootstrap='2000 unique-input resamples, retain all three search seeds per input, RNG20261114',
            secondary='Original output selection, all other frozen checkpoints, N1, stopped/capped counts, per-clue metrics, selector/certificate audits and compute. No setting or checkpoint selected from held-out results.'),
        validation=dict(cpu='Independent complete-solution enumeration plus direct rules; float64 literal full-row two-statePL and gradient versus aggregated binaryPL on a duplicated n12 fixture, atol1e-10/rtol1e-9.',
            gpu='Before research fits: same full expanded training data loss/gradient comparison against uncompressed upstream OffsetTask at fixed deterministic parameters; absolute loss/gradient error<=1e-5. Three measured forward+backward repetitions after one warmup for each path, no optimizer step or search. Record speed and peak memory.',
            saved='Exhaustive14200solution membership, givens, original and certificate-first selection identities, per-seed counts, FLOPs, model/data/source hashes.'),
        budget=dict(research_fits=3, new_search_jobs=15, gpu_minutes_envelope=20,
            policy='Workers run on both allocated GPUs with exclusive locks and dynamic pending-job claims. If the envelope is unexpectedly exhausted, retain partial evidence and report incomplete; no silent cap/setting reduction.'),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        data_sha256={str((OUT / name).relative_to(ROOT)):sha(OUT / name) for name in ['data.npz','dataset.json']},
        limits=['New generated n12 split rather than official GRAM benchmark; original n10 hyperparameters transferred without n12 tuning.',
                'Complete solution labels may repeat between disjoint partial inputs, as in the original GRAM protocol; input groups are strictly disjoint.',
                'Clue strata held out contain only6one-clue and305two-clue inputs; primary estimate pools the declared distribution, not equally weighted strata.',
                'One deterministic fitted trajectory per optimizer; repeated inference seeds are not training-seed replication.',
                'Aggregated objective matches the full-row mathematical objective; reordered float32 arithmetic can perturb learned trajectories.',
                'Certificate-first differs from the paper output-selection convention; original-selection results remain separately visible.'])
    save(SRC / 'protocol.json', p); emit('protocol_frozen', sha256=sha(SRC / 'protocol.json'))


def loss(A, b, targets, weights, n):
    import torch
    import torch.nn.functional as F
    import nqueens_offset as O
    from cram import cram_nqueens as Q
    idx, _ = O.offsets(n, A.device); i, j = Q.pair_index(n*n, A.device)
    J = torch.zeros(n*n, n*n, device=A.device, dtype=A.dtype).index_put((i,j), A[idx]).index_put((j,i), A[idx])
    fields = targets @ J + b
    return (F.binary_cross_entropy_with_logits(-fields, targets, reduction='none') * weights).sum()


def validate_cpu():
    import numpy as np
    import torch
    import nqueens_offset as O
    import training_methods as AT
    torch.set_num_threads(1)
    data = np.load(OUT / 'data.npz'); solutions = data['solutions'][:9]
    ix = np.random.default_rng(715).integers(0,9,63)
    labels = torch.tensor(solutions[ix], dtype=torch.long)
    givens = torch.zeros_like(labels, dtype=torch.bool)
    for r in range(len(ix)):
        occupied = torch.nonzero(labels[r]).flatten(); givens[r, occupied[:1+r%3]] = True
    task = O.OffsetTask(givens, labels, 12, 'ising')
    task.A = torch.linspace(-.23,.31,264,dtype=torch.float64,requires_grad=True)
    task.b = torch.tensor([-.17],dtype=torch.float64,requires_grad=True)
    # Upstream onehot casts to default float32; literal two-state field keeps float64.
    task.begin(); Y = torch.nn.functional.one_hot(labels,2).double()
    field = (Y.reshape(len(Y),-1) @ task.M).reshape(len(Y),144,2) + task.B
    nll = -torch.log_softmax(-field,-1).gather(-1,labels[...,None]).squeeze(-1)
    full = nll[~givens].mean(); fg = torch.autograd.grad(full,(task.A,task.b))
    counts = torch.zeros(9,144,dtype=torch.float64)
    counts.index_add_(0,torch.tensor(ix),(~givens).double())
    A = task.A.detach().clone().requires_grad_(); b = task.b.detach().clone().requires_grad_()
    aggregated = loss(A,b,torch.tensor(solutions,dtype=torch.float64),counts/counts.sum(),12)
    ag = torch.autograd.grad(aggregated,(A,b))
    torch.testing.assert_close(full,aggregated,atol=1e-10,rtol=1e-9)
    for a,c in zip(fg,ag): torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-9)
    save(OUT / 'cpu-validation.json',dict(status='passed',utc=now(),loss_absolute_error=float(abs(full-aggregated)),
        gradient_max_absolute_error=max(float((a-c).abs().max()) for a,c in zip(fg,ag)),
        protocol_sha256=sha(SRC / 'protocol.json'), source_sha256=sha(__file__), no_accuracy_measured=True))


def validate_gpu(data):
    import numpy as np
    import torch
    import nqueens_offset as O
    import training_methods as AT
    # Full expanded rows are used only for this bounded optimization validation.
    keys = data['expanded_fit_keys']; digits = np.stack([keys//145**2,keys//145%145,keys%145],1)
    givens = np.zeros((len(keys),144),bool)
    for k in range(3):
        present=digits[:,k]>0;givens[np.flatnonzero(present),digits[present,k]-1]=True
    labels = torch.tensor(data['solutions'][data['expanded_fit_solution_indices']],device='cuda',dtype=torch.long)
    task = O.OffsetTask(torch.tensor(givens,device='cuda'),labels,12,'ising')
    with torch.no_grad(): task.A.copy_(torch.linspace(-.23,.31,264,device='cuda'));task.b.fill_(-.17)
    targets = torch.tensor(data['solutions'],device='cuda',dtype=torch.float32)
    weights = torch.tensor(data['free_counts'],device='cuda',dtype=torch.float32);weights /= weights.sum()
    AA=task.A.detach().clone().requires_grad_();bb=task.b.detach().clone().requires_grad_()
    timings={}; gradients={}; losses={};memory={}
    for path in ['expanded_upstream','aggregated_binary']:
        values=[]
        for repeat in range(4):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            if path=='expanded_upstream':
                task.begin();value=AT.loss_fn('pl',None,task,task.ctx,torch.Generator(device='cuda'),{})
                grad=torch.autograd.grad(value,task.params())
            else:
                value=loss(AA,bb,targets,weights,12);grad=torch.autograd.grad(value,(AA,bb))
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if repeat:values.append(seconds)
            gradients[path]=[g.detach().cpu() for g in grad];losses[path]=float(value.detach())
            memory[path]=torch.cuda.max_memory_allocated()
        timings[path]=values
    difference=max(float((a-b).abs().max()) for a,b in zip(gradients['expanded_upstream'],gradients['aggregated_binary']))
    error=abs(losses['expanded_upstream']-losses['aggregated_binary'])
    assert difference<=1e-5 and error<=1e-5,(difference,error)
    save(OUT / 'gpu-validation.json',dict(status='passed',utc=now(),expanded_rows=len(keys),compressed_solutions=len(targets),
        loss_values=losses,loss_absolute_error=error,gradient_max_absolute_error=difference,
        timings_seconds=timings,median_forward_backward_speedup=float(np.median(timings['expanded_upstream'])/np.median(timings['aggregated_binary'])),
        peak_memory_bytes=memory,scope='Three GPU forward+backward calls after one warmup on the actual complete fit data, with no optimizer update or search; not an end-to-end fit speedup.',
        protocol_sha256=sha(SRC / 'protocol.json'),source_sha256=sha(__file__)))
    del task,labels,targets,weights,AA,bb,grad;torch.cuda.empty_cache()


def fit(arm,data,progress):
    import torch
    import nqueens_offset as O
    from cram import cram_nqueens as Q
    cfg=protocol()['arms'][arm]
    torch.cuda.synchronize();start=time.perf_counter()
    target=torch.tensor(data['solutions'],device='cuda',dtype=torch.float32)
    weights=torch.tensor(data['free_counts'],device='cuda',dtype=torch.float32);weights/=weights.sum()
    multiplicity=torch.tensor(data['rows_per_solution'],device='cuda',dtype=torch.float32)
    A=torch.zeros(264,device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True)
    params=[A,b];buffers=[torch.zeros_like(t) for t in params]
    counts=torch.bincount(O.offsets(12,'cuda')[0]).float();pre=counts/counts.mean()
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'],foreach=False) if cfg['optimizer']=='adamw' else None
    snapshot_seconds=0.
    for k in range(4000):
        objective=loss(A,b,target,weights,12);grads=torch.autograd.grad(objective,params)
        lr=cfg['lr']*.5*(1+math.cos(math.pi*k/4000))
        if opt is not None:
            opt.param_groups[0]['lr']=lr
            for par,g in zip(params,grads):par.grad=g
            opt.step()
        else:
            grads=(grads[0]/pre,grads[1])
            with torch.no_grad():
                for par,g,buf in zip(params,grads,buffers):buf.mul_(.5).add_(g);par.sub_(lr*buf)
        if (k+1)%250==0:progress(stage='fit',arm=arm,update=k+1,loss=float(objective.detach()))
        if k+1 in cfg['snapshots']:
            torch.cuda.synchronize();snapstart=time.perf_counter()
            with torch.no_grad():
                W,B=O.pair_tables(A.detach(),b.detach(),'ising',12);energy=Q.full_S(W,Q.pair_index(144,'cuda'),144)
                fields=Q.fields(energy,B,target.long());diff=fields[:,:,1]-fields[:,:,0]
                tau=float(Q.board_energy(energy,B,target.long()).max())
                unit=float((diff.abs().mean(1)*multiplicity).sum()/multiplicity.sum())
                delta=float(((1-2*target)*diff).min())
                model=dict(A=A.detach().cpu(),b=b.detach().cpu(),W=W.cpu(),B=B.cpu(),S=energy.cpu(),tau=tau,unit=unit,delta=delta,arm='ising')
            torch.cuda.synchronize();snapshot_seconds+=time.perf_counter()-snapstart
            dest=OUT/f'fits/{arm}/u{k+1}';dest.mkdir(parents=True,exist_ok=True);torch.save(model,dest/'model.pt')
            save(dest/'fit.json',dict(status='complete',arm=arm,updates=k+1,rows=int(data['rows_per_solution'].sum()),
                unique_inputs=100000,distinct_solutions=len(target),final_loss=float(objective.detach()),unit=unit,delta=delta,tau=tau,
                elapsed_fit_seconds_including_prior_snapshots=time.perf_counter()-start,snapshot_seconds=snapshot_seconds,
                timing_scope='One instrumented trajectory; includes earlier snapshot overhead. No cross-arm training speed claim from these times.',
                model_sha256=sha(dest/'model.pt'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    emit('fit_complete',arm=arm,seconds=time.perf_counter()-start)


def select(energy,stopped):
    import numpy as np
    original=energy.argmin(1);certified=np.where(stopped,energy,np.inf).argmin(1)
    return original,np.where(stopped.any(1),certified,original)


def search(arm,update,seed,data,progress):
    import numpy as np
    import torch
    from cram import cram_nqueens as Q
    path=OUT/f'fits/{arm}/u{update}/model.pt';model=torch.load(path,weights_only=False)
    inputs=data['test_inputs'];progress(stage='search',arm=arm,update=update,seed=seed,inputs=len(inputs))
    emit('search_start',arm=arm,update=update,seed=seed)
    start=time.perf_counter()
    boards,stopped,energy,moves,flop=Q.solve(model,inputs,chains=20,cap=20000,seed=seed,device='cuda',temperature=1.,stop='local',margin=.001)
    seconds=time.perf_counter()-start;valid=S.independent_ok(boards,inputs,12)
    denom=np.minimum(data['test_solution_counts'],20)
    found=np.array([min(len({b.tobytes() for b in samples[ok]}),cap) for samples,ok,cap in zip(boards,valid,denom)])
    orig,cert=select(energy,stopped);ids=np.arange(len(inputs));n20=valid[ids,orig];n20cert=valid[ids,cert]
    dest=OUT/f'search/{arm}/u{update}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'records.npz',inputs=inputs,boards=np.packbits(boards,-1),stopped=stopped,energy=energy,
        moves=moves,flop=flop,valid=valid,n20=n20,n20_certificate=n20cert,n1=valid[:,0],found=found,denom=denom,
        solution_counts=data['test_solution_counts'],original_index=orig,certificate_index=cert)
    rec=dict(status='complete',utc=now(),arm=arm,updates=update,seed=seed,unique_inputs=len(inputs),
        original_accuracy_pct=100*float(n20.mean()),certificate_accuracy_pct=100*float(n20cert.mean()),
        coverage_pct=100*float(found.sum()/denom.sum()),wrong_stops=int((stopped&~valid).sum()),
        capped_chains=int((~stopped).sum()),mean_moves=float(moves.mean()),search_seconds=seconds,
        model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),records_sha256=sha(dest/'records.npz'),
        protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__))
    save(dest/'result.json',rec);emit('search_complete',**{key:rec[key] for key in ['arm','updates','seed','original_accuracy_pct','certificate_accuracy_pct','coverage_pct','search_seconds']})


def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip()
    assert query('hostname')=='loki'
    device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader')
    assert device.split(',')[1].strip() not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.empty(1,device='cuda');torch.cuda.synchronize()
    p=protocol();assert read(OUT/'cpu-validation.json')['status']=='passed'
    for group in ['source_sha256','data_sha256']:
        for path,digest in p[group].items():assert sha(ROOT/path)==digest,path
    data=dict(np.load(OUT/'data.npz'));started=time.monotonic()
    base=dict(status='running',started_utc=now(),host='loki',gpu=gpu,device=device,pid=os.getpid(),protocol_sha256=sha(SRC/'protocol.json'))
    receipt=OUT/f'worker-g{gpu}.json'
    def progress(**values):save(receipt,dict(base,updated_utc=now(),progress=values))
    progress(stage='validation_gate')
    try:
        if gpu==0 and not(OUT/'gpu-validation.json').exists():validate_gpu(data)
        while not(OUT/'gpu-validation.json').exists():time.sleep(1)
        assert read(OUT/'gpu-validation.json')['status']=='passed'
        while True:
            pending=worked=False
            for arm in p['arms']:
                dest=OUT/f'fits/{arm}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'u4000/fit.json').exists():continue
                pending=True;claim=(dest/'fit.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if not(dest/'u4000/fit.json').exists():fit(arm,data,progress);worked=True
                finally:claim.close()
            jobs=[(a,u,s) for a in p['arms'] for u in p['arms'][a]['snapshots'] for s in [0,1,2]]
            for arm,u,seed in jobs:
                dest=OUT/f'search/{arm}/u{u}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'result.json').exists():continue
                pending=True
                if not(OUT/f'fits/{arm}/u{u}/fit.json').exists():continue
                claim=(dest/'job.lock').open('a')
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:claim.close();continue
                try:
                    if time.monotonic()-started>600:raise RuntimeError('Conservative20GPU-minute campaign envelope reached; partial evidence retained')
                    if not(dest/'result.json').exists():search(arm,u,seed,data,progress);worked=True
                finally:claim.close()
            if not pending:break
            if not worked:time.sleep(1)
        save(receipt,dict(base,status='complete',completed_utc=now()))
    except BaseException as exc:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(exc)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='mode',required=True)
    sp.add_parser('prepare');sp.add_parser('validate');w=sp.add_parser('worker');w.add_argument('--gpu',type=int,choices=[0,1],required=True)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    elif args.mode=='validate':validate_cpu()
    else:worker(args.gpu)
