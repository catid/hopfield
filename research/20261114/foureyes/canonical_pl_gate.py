"""Source-only algebra, convex-update and CPU/GPU checks before fresh fits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/canonical_pl_gate'
PROTOCOL=HERE/'canonical_pl_gate.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    previous=json.loads((HERE/'noisy_pl_long.json').read_text())
    files=[Path(__file__),HERE/'canonical_pl.py',ROOT/'upstream/experiments/sudoku.py']
    p=dict(id='canonical-PL-source-only-gate',frozen_utc=datetime.now(timezone.utc).isoformat(),
        banks=[1,2,3,4],n=600,raw_lambda=1/30000,updates=256,initial_step=540.,dtype='float64',
        input=previous['gate_source'],noise=previous['matched_gate']['corruptions'],
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
        inputs_sha256={name:sha(ROOT/name) for name in [previous['gate_source'],previous['matched_gate']['corruptions']]},
        conditions=['manualgradient equalsautograd','centralfinite-differences','independentraw-fullenergy loss equalscanonical',
            'per-row digitpermutation invariance','exactorigin-gradient countidentity',
            'CPU64/GPU64 supportsidentical andcoefficienterror<quarterthresholdseparation','everyaccepted objective/majorizergatepasses'],
        max_case_seconds=120,
        scope='Fourpreviousnoisysourcefixtures only; nofreshsourcegeneration,hyperparamsearch,targetdata orscientificinference.',
        proof_scope='Uniformall9!digit-averagedrawPL withunorderedWentrywiseL1 hasaglobalmin atW=JI,B0,unrestrictedrealJ,penalty9lambda|J|. No samefiniteoptimizerpath orconvergenceclaim.')
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n');OUT.mkdir(parents=True)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def gate(bank):
    gpu=bank-1;lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    import canonical_pl as C
    sys.path[:0]=[str(ROOT/'upstream'),str(ROOT/'upstream/experiments')]
    import sudoku as CF
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    p=json.loads(PROTOCOL.read_text());start=time.perf_counter()
    for name,digest in (p['source_sha256']|p['inputs_sha256']).items():assert sha(ROOT/name)==digest
    source=np.load(ROOT/p['input']);noise=np.load(ROOT/p['noise'])
    values=source['solutions'][bank,:600].astype(np.int64);free=~source['given'][bank,:600]
    np.put_along_axis(values,noise['cells'][bank,:600].astype(np.int64),noise['values'][bank,:600],1)
    sol=torch.as_tensor(values);fr=torch.as_tensor(free);ctx=C.context(torch,sol,fr,torch.float64)
    rng=torch.Generator().manual_seed(20261115800+bank)
    probe=torch.randn(3240,generator=rng,dtype=torch.float64)*.03;probe.requires_grad_(True)
    f,g=C.loss_gradient(probe,ctx)
    automatic=torch.autograd.grad(f,probe)[0]
    gradient_error=float((g-automatic).abs().max());assert gradient_error<1e-12
    finite_errors=[]
    for index in np.random.default_rng(20261115800+bank).choice(3240,16,replace=False):
        before=probe.detach().clone();after=before.clone();before[index]-=1e-5;after[index]+=1e-5
        central=(C.loss_gradient(after,ctx,False)-C.loss_gradient(before,ctx,False))/(2e-5)
        finite_errors.append(abs(float(central-g[index])))
    assert max(finite_errors)<1e-9
    w=probe.detach()[:,None,None]*torch.eye(9,dtype=torch.float64)
    full=CF.full_S(w,CF.pair_index('cpu'));fields=torch.einsum('nmac,bmc->bna',full,ctx['y'])
    raw_loss=-(torch.log_softmax(-fields,-1).gather(2,sol[...,None]).squeeze(-1)*fr).sum()/fr.sum()
    assert abs(float(raw_loss-f))<1e-12
    relabel=torch.rand(600,9,generator=rng).argsort(1).gather(1,sol)
    invariant=C.loss_gradient(probe.detach(),C.context(torch,relabel,fr,torch.float64),False)
    assert abs(float(invariant-f))<1e-12
    ii,jj=np.triu_indices(81,1);exposure=free[:,ii].astype(int)+free[:,jj].astype(int)
    A=exposure.sum(0);equal=(values[:,ii]==values[:,jj]);coincidence=(exposure*equal).sum(0)
    exact=(9*coincidence-A)/(9*int(free.sum()))
    _,origin=C.loss_gradient(torch.zeros(3240,dtype=torch.float64),ctx)
    assert np.max(np.abs(origin.numpy()-exact))<1e-14
    outputs={};fits={};metas={}
    for label,device in [('gpu64','cuda'),('cpu64','cpu')]:
        j,record=C.fit(torch,sol.to(device),fr.to(device),p['raw_lambda'],p['updates'],p['initial_step'])
        support,meta=C.select(torch,j);assert meta['eligible']
        outputs[label]=j.cpu();outputs[label+'_support']=support.cpu();fits[label]=record;metas[label]=meta
        assert time.perf_counter()-start<p['max_case_seconds']
    error=float((outputs['gpu64']-outputs['cpu64']).abs().max())
    separation=min(m['minimum_threshold_distance'] for m in metas.values())/4
    passed=torch.equal(outputs['gpu64_support'],outputs['cpu64_support']) and error<separation
    np.savez_compressed(OUT/f'bank{bank}-coefficients.npz',**{k:v.numpy() for k,v in outputs.items()})
    result=dict(status='passed' if passed else 'failed',bank=bank,protocol_sha256=sha(PROTOCOL),gradient_error=gradient_error,
        finite_difference_error=max(finite_errors),raw_loss_error=abs(float(raw_loss-f)),permutation_error=abs(float(invariant-f)),
        origin_count_error=float(np.max(np.abs(origin.numpy()-exact))),coefficient_error=error,allowed=separation,
        support_equal=torch.equal(outputs['gpu64_support'],outputs['cpu64_support']),fits=fits,selection=metas,
        seconds=time.perf_counter()-start,coefficient_sha256=sha(OUT/f'bank{bank}-coefficients.npz'))
    (OUT/f'bank{bank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['fits']}));assert passed


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--bank',type=int)
    args=parser.parse_args()
    if args.freeze:freeze()
    else:gate(args.bank)
