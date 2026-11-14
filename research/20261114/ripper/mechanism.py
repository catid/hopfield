"""H1c: fixed-scale gauge/repulsion interventions with dynamic GPU scheduling."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from run import ROOT, UP, HERE, sha, save, now, projection, captured_search_class, independent_validity

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'mechanism'
PROTOCOL=HERE/'protocol-mechanism.json'


def queue_update(job_id=None, status=None, worker=None):
    lock=(OUT/'queue.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        q=json.loads((OUT/'queue.json').read_text())
        if job_id is None:
            job=next((j for j in q if j['status']=='pending'),None)
            if job is not None:job.update(status='running',worker=worker,claimed_utc=now())
        else:
            job=next(j for j in q if j['job_id']==job_id);job.update(status=status,updated_utc=now())
        save(OUT/'queue.json',q)
        return job


def run_job(job,gpu,protocol,torch,np,CF,GS,SS):
    k,seed,arm=job['k'],job['seed'],job['arm'];dest=OUT/'jobs'/job['job_id']
    if dest.exists():raise RuntimeError(f'Refusing existing job directory {dest}')
    dest.mkdir(parents=True)
    entry=next(r for r in protocol['frozen_inputs'] if r['k']==k and r['seed']==seed)
    src=ROOT/entry['input_dir']
    for name,h in entry['sha256'].items():assert sha(src/name)==h
    sd=torch.load(src/'model.pt',weights_only=True,map_location='cuda')
    rawW,rawB=projection(sd['W'],sd['B'])
    eye=torch.eye(9,device='cuda',dtype=torch.bool)
    diag=rawW[:,eye].mean(-1);off=rawW[:,~eye].mean(-1);J=diag-off
    canonical=J[:,None,None]*eye
    constant=float(off.double().sum()+rawB[:,0].double().sum())
    # Board-independent error bound: sum worst pair residuals, plus unary residuals.
    residual=(rawW.double()-canonical.double()-off[:,None,None].double()).abs().flatten(1).max(1).values
    bound=float(residual.sum())
    assert bound<1e-4 and torch.equal(rawB,rawB[:,0,None].expand_as(rawB))
    tr=np.load(src/'train.npz');train=torch.as_tensor(tr['solutions'].astype(np.int64),device='cuda')
    Sraw=CF.full_S(rawW,CF.pair_index('cuda'));Scan=CF.full_S(canonical,CF.pair_index('cuda'))
    eraw=GS.board_energy(Sraw,rawB,train);ecan=GS.board_energy(Scan,torch.zeros_like(rawB),train)
    observed=float((eraw-ecan-constant).abs().max())
    assert observed<1e-4, observed
    rawtau=float(eraw.max());cantau=float(ecan.max())
    assert abs(rawtau-cantau-constant)<1e-4
    if arm=='projected':W,B=rawW,rawB
    elif arm=='canonical':W,B=canonical,torch.zeros_like(rawB)
    elif arm=='canonical_repulsive':W,B=J.clamp_min(0)[:,None,None]*eye,torch.zeros_like(rawB)
    else:raise ValueError(arm)
    W=W.contiguous();B=B.contiguous();unit=float(rawW[rawW>0].mean())
    m=CF.prepare(W,B,train)
    lo,hi=protocol['rows_half_open'];sol,free,idx=SS.load_puzzles('dev10k',lo,hi-lo)
    ss=captured_search_class(GS)(m['S'],B,sol,free,temp=.4*unit,theta=.45*unit,focused=True,C=4,
        tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=seed,
        anneal=(.4*unit,.31*unit,524288))
    ss.T_cap=262144;ss.cap=ss.T_cap*ss.fl_move*4
    proof={'analytic_constant':constant,'all_board_float32_coefficient_residual_bound':bound,
           'max_training_energy_shift_error':observed,'training_threshold_shift_error':abs(rawtau-cantau-constant),
           'raw_training_tau':rawtau,'canonical_training_tau':cantau,
           'scope':'Constant-energy equivalence in real arithmetic, bounded float32 coefficient residual for every board; focus trajectory need not be gauge invariant.'}
    rec={'status':'running','job_id':job['job_id'],'k':k,'seed':seed,'arm':arm,'gpu':gpu,'host':platform.node(),
         'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),
         'capture_helper_sha256':sha(HERE/'run.py'),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [UP/'experiments/sudoku.py',UP/'cram/cram_sudoku_general.py',UP/'cram/cram_sudoku_search.py']},
         'input_sha256':entry['sha256'],'development_sha256':sha(UP/'data/frm/dev10k/dev.npz'),
         'used_hbar':unit,'model_hbar':m['hbar'],'statistics':{z:m[z] for z in ['g','ebar','tau_train','hbar','D']},
         'gauge_proof':proof,'positive_J':int((J>0).sum()),'negative_J':int((J<0).sum()),
         'fl_move':ss.fl_move,'move_cap':ss.T_cap,'chains':4,'test_rows':len(sol),'development_rows_half_open':[lo,hi],
         'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec);torch.save({'W':W.cpu(),'B':B.cpu()},dest/'searched_model.pt')
    last=[time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0]>=2 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,'active':len(s.act),'elapsed_s':s.elapsed});last[0]=time.monotonic()
    start=time.monotonic();result=ss.run(progress,log_every=65536);torch.cuda.synchronize();wall=time.monotonic()-start
    boards=ss.boards().cpu().numpy().astype(np.int8);exact=(boards==sol).all(1);valid=independent_validity(boards,sol,free)
    assert np.array_equal(exact,result['solved']) and np.array_equal(exact,valid)
    stopped=ss.stopped.cpu().numpy();moves=torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,puzzle=idx,stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=ss.t,solved=int(exact.sum()),valid=int(valid.sum()),stopped=int(stopped.sum()),wrong_stops=int((stopped&~exact).sum()),budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),output_sha256={n:sha(dest/n) for n in ['boards.npz','result.json','searched_model.pt']})
    save(dest/'receipt.json',rec)
    print(json.dumps({'event':'complete',**{key:rec[key] for key in ['job_id','solved','wrong_stops','budget_failures','wall_s']}}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initialize',action='store_true');ap.add_argument('--gpu',type=int);a=ap.parse_args()
    assert platform.node()=='ripper'
    p=json.loads(PROTOCOL.read_text());OUT.mkdir(parents=True,exist_ok=True)
    if a.initialize:
        assert not (OUT/'queue.json').exists()
        rows=[{'job_id':f'k{k}-s{s}-{arm}','k':k,'seed':s,'arm':arm,'status':'pending'} for k in p['primary_k'] for s in p['training_seeds'] for arm in p['arms']]
        save(OUT/'queue.json',rows);print('Initialized',len(rows),'jobs');return
    assert a.gpu in range(4)
    lock=(ROOT/f'artifacts/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=lambda args:subprocess.check_output(args,text=True,timeout=20).strip()
    gpuinfo=q(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader']);uuid=gpuinfo.split(',')[0].strip()
    assert uuid not in q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(UP),str(UP/'experiments')]
    import torch
    import numpy as np
    import sudoku as CF
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    proof=json.loads((BASE/'capture-parity.json').read_text());assert proof['source_sha256']==sha(HERE/'run.py') and proof['state_dict_field_count']==25
    claimed=[]
    save(OUT/f'worker{a.gpu}.json',{'status':'running','pid':os.getpid(),'gpu':a.gpu,'gpu_before':gpuinfo,'started_utc':now()})
    while True:
        job=queue_update(worker={'pid':os.getpid(),'gpu':a.gpu})
        if job is None:break
        claimed.append(job['job_id'])
        try:run_job(job,a.gpu,p,torch,np,CF,GS,SS)
        except BaseException:
            queue_update(job['job_id'],'failed');raise
        queue_update(job['job_id'],'complete')
    save(OUT/f'worker{a.gpu}.json',{'status':'complete','pid':os.getpid(),'gpu':a.gpu,'jobs':claimed,'completed_utc':now()})


if __name__=='__main__':main()
