"""Source-only precision diagnosis; never reads fresh H4s inputs or test data."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/long_precision'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int,required=True);args=parser.parse_args()
    p=json.loads((HERE/'long_precision.json').read_text());case=p['cases'][args.case]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.case),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(ROOT/'upstream'),str(ROOT/'upstream/experiments')]
    import fcntl
    lock=(ROOT/f'artifacts/gpu{args.case}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={args.case}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    import numpy as np
    import torch
    import sudoku as CF
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for name,digest in p['frozen_inputs'].items():assert sha(ROOT/name)==digest
    n=case['n'];z=np.load(ROOT/p['input']);s=z['solutions'][0,:n].astype(np.int64);free=~z['given'][0,:n]
    if case['noise']:
        noise=np.load(ROOT/p['corruptions']);np.put_along_axis(s,noise['cells'][0,:n].astype(np.int64),noise['values'][0,:n],1)
    draws=torch.rand((256,n,9),generator=torch.Generator().manual_seed(2026111512 if n==32 else 2026111513))
    dest=OUT/f'case{args.case}';dest.mkdir(exist_ok=False,parents=True)
    receipts=[];snapshots={};start=time.monotonic()
    for label,device,dtype in [('gpu32','cuda',torch.float32),('gpu64','cuda',torch.float64),('cpu32','cpu',torch.float32),('cpu64','cpu',torch.float64)]:
        sol=torch.as_tensor(s,device=device);fr=torch.as_tensor(free,device=device);pair=CF.pair_index(device)
        w=torch.zeros(3240,9,9,dtype=dtype,device=device,requires_grad=True);b=torch.zeros(81,9,dtype=dtype,device=device,requires_grad=True)
        bufs=[torch.zeros_like(w),torch.zeros_like(b)];records=[];fit_start=time.monotonic()
        for k in range(256):
            labeled=draws[k].to(device).argsort(1).gather(1,sol)
            if dtype==torch.float32:
                loss=CF.pl_loss(CF.full_S(w,pair),b,labeled,fr)
            else:
                # Same algebra; original helper hardcodes one-hot to float32.
                y=torch.nn.functional.one_hot(labeled,9).to(dtype)
                fields=torch.einsum('nmac,bmc->bna',CF.full_S(w,pair),y)+b[None]
                losses=-torch.log_softmax(-fields,-1).gather(2,labeled[...,None]).squeeze(-1)
                loss=(losses*fr).sum()/fr.sum()
            grads=torch.autograd.grad(loss,(w,b));lr=540*.5*(1+math.cos(math.pi*k/256))
            with torch.no_grad():
                for parameter,gradient,buffer in zip((w,b),grads,bufs):
                    buffer.mul_(.5).add_(gradient);parameter.sub_(lr*buffer)
                    parameter.copy_(parameter.sign()*(parameter.abs()-lr*(.003/9)).clamp_min(0))
            if k+1 in p['checkpoints']:
                array=w.detach().cpu().numpy().astype(np.float64)
                diag=np.diagonal(array,axis1=1,axis2=2).mean(1)
                mask=~np.eye(9,dtype=bool);off=array[:,mask].mean(1)
                coeff=np.maximum(diag-off,0);snapshots[f'{label}_{k+1}']=coeff
                positive=np.sort(coeff[coeff>0]);ix=int(np.diff(positive).argmax());cut=(positive[ix]+positive[ix+1])/2
                snapshots[f'{label}_{k+1}_support']=coeff>cut
                records.append(dict(update=k+1,loss_before_step=float(loss.detach()),coefficient_max=float(coeff.max()),
                               threshold=float(cut),min_threshold_distance=float(np.abs(coeff-cut).min()),retained=int((coeff>cut).sum())))
                write(dest/'progress.json',dict(case=case,stage=label,update=k+1,wall_s=time.monotonic()-start,pid=os.getpid()))
        if device=='cuda':torch.cuda.synchronize()
        seconds=time.monotonic()-fit_start
        receipts.append(dict(label=label,seconds=seconds,checkpoints=records))
        torch.save(dict(W=w.detach().cpu(),B=b.detach().cpu()),dest/f'{label}.pt')
        if label=='gpu32':
            original=CF.train(CF.FullConfig(updates=256),sol,fr,draws=draws)
            assert torch.equal(original[0],w) and torch.equal(original[1],b),'Copied update loop disagrees with native function'
        assert time.monotonic()-start<p['max_case_seconds']
    comparisons=[]
    for update in p['checkpoints']:
        for left,right in [('cpu32','gpu32'),('gpu32','gpu64'),('cpu64','gpu64')]:
            a,b=snapshots[f'{left}_{update}'],snapshots[f'{right}_{update}']
            comparisons.append(dict(update=update,left=left,right=right,max_coefficient_error=float(np.max(np.abs(a-b))),
                                    support_disagreements=int(np.count_nonzero(snapshots[f'{left}_{update}_support']!=snapshots[f'{right}_{update}_support']))))
    np.savez_compressed(dest/'trajectories.npz',**snapshots)
    result=dict(status='complete',case=case,protocol_sha256=sha(HERE/'long_precision.json'),producer_sha256=sha(__file__),
                native_gpu32_final_bitwise_equal=True,receipts=receipts,comparisons=comparisons,wall_s=time.monotonic()-start)
    write(dest/'result.json',result);print(json.dumps(dict(case=case,final=[r for r in comparisons if r['update']==256],wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
