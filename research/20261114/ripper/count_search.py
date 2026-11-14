"""H1j: no-optimizer count graph versus learned normalized graphs."""
import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from run import ROOT, UP, HERE, sha, save, now, captured_search_class, independent_validity

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'count_search'
PROTOCOL=HERE/'protocol-count-search.json'


def queue(job_id=None,status=None,gpu=None):
    with (OUT/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        q=json.loads((OUT/'queue.json').read_text())
        if job_id is None:
            j=next((j for j in q if j['status']=='pending'),None)
            if j is not None:j.update(status='running',gpu=gpu,pid=os.getpid(),started_utc=now())
        else:
            j=next(j for j in q if j['job_id']==job_id);j.update(status=status,updated_utc=now())
        save(OUT/'queue.json',q);return j


def run(j,gpu,p,torch,np,CF,GS,SS,SP):
    from sparse_backend import sparse_search_class
    k,seed,arm=j['k'],j['seed'],j['arm'];dest=OUT/'jobs'/j['job_id'];dest.mkdir(parents=True,exist_ok=False)
    src=BASE/'inputs'/f'k{k}-s{seed}'
    entry=next(e for e in p['frozen_inputs'] if e['k']==k and e['seed']==seed)
    for n,h in entry['sha256'].items():assert sha(src/n)==h
    if arm=='count_sparse':
        assert k==10
        countfile=BASE/'count'/f'k10-s{seed}'/'counts.npz'
        support=np.load(countfile)['support'];J=torch.as_tensor(support.astype(np.float32),device='cuda')
        sd={'W':(J[:,None,None]*torch.eye(9,device='cuda')).contiguous(),'B':torch.zeros(81,9,device='cuda')}
        tr=np.load(src/'train.npz');train=torch.as_tensor(tr['solutions'].astype(np.int64),device='cuda')
        m=CF.prepare(sd['W'],sd['B'],train);assert m['tau_train']==0 and m['ebar']==0 and m['hbar']==1
        unit=1.0;model_sha=sha(countfile)
    else:
        assert arm=='normalized_sparse'
        prior=BASE/'scale/jobs'/f'k{k}-s{seed}-normalized_unit'
        rr=json.loads((prior/'receipt.json').read_text());sd=torch.load(prior/'searched_model.pt',weights_only=True,map_location='cuda')
        assert sha(prior/'searched_model.pt')==rr['output_sha256']['searched_model.pt']
        m=rr['statistics'];unit=rr['used_hbar'];model_sha=sha(prior/'searched_model.pt')
        m['S']=CF.full_S(sd['W'],CF.pair_index('cuda'))
    lo,hi=p['rows_half_open'];sol,free,idx=SS.load_puzzles('full-all',lo,hi-lo)
    klass=sparse_search_class(GS,SP) if arm.endswith('_sparse') else captured_search_class(GS)
    torch.cuda.synchronize();start=time.monotonic()
    ss=klass(m['S'],sd['B'],sol,free,temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=seed,anneal=(.4*unit,.31*unit,524288))
    ss.T_cap=262144;ss.cap=ss.T_cap*ss.fl_move*4
    torch.cuda.synchronize();setup=time.monotonic()-start
    rec={'status':'running','job_id':j['job_id'],'k':k,'seed':seed,'arm':arm,'gpu':gpu,'host':platform.node(),'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'backend_sha256':sha(HERE/'sparse_backend.py'),'capture_sha256':sha(HERE/'run.py'),'input_sha256':entry['sha256'],'model_sha256':model_sha,'used_hbar':unit,'statistics':{key:m[key] for key in ['ebar','tau_train','hbar','D']},'move_cap':ss.T_cap,'chains':4,'test_rows':len(sol),'test_rows_half_open':[lo,hi],'split':'full-all','fl_move':ss.fl_move,'setup_s':setup,'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec);last=[time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0]>=2 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,'active':len(s.act),'elapsed_s':s.elapsed});last[0]=time.monotonic()
    start=time.monotonic();result=ss.run(progress,log_every=65536);torch.cuda.synchronize();wall=time.monotonic()-start
    boards=ss.boards().cpu().numpy().astype(np.int8);exact=(boards==sol).all(1);valid=independent_validity(boards,sol,free)
    assert np.array_equal(exact,result['solved']) and np.array_equal(exact,valid)
    stopped=ss.stopped.cpu().numpy();moves=torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,puzzle=idx,stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    # Retain exact normalized trajectories for a future budget curve without rerunning the prefix.
    if arm=='count_sparse':
        torch.save(ss.state_dict(),dest/'final_state.pt')
        torch.save({'best_boards':ss.best_boards.cpu(),'cert_boards':ss.cert_boards.cpu()},dest/'capture_state.pt')
    rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=ss.t,solved=int(exact.sum()),valid=int(valid.sum()),stopped=int(stopped.sum()),wrong_stops=int((stopped&~exact).sum()),budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),output_sha256={n:sha(dest/n) for n in ['boards.npz','result.json']})
    save(dest/'receipt.json',rec);print(json.dumps({'job_id':j['job_id'],'solved':rec['solved'],'wrong_stops':rec['wrong_stops'],'wall_s':wall}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int);ap.add_argument('--initialize',action='store_true');a=ap.parse_args();assert platform.node()=='ripper'
    p=json.loads(PROTOCOL.read_text());OUT.mkdir(exist_ok=True)
    if a.initialize:
        assert not (OUT/'queue.json').exists()
        jobs=[{'job_id':f'k10-s{s}-count_sparse','k':10,'seed':s,'arm':'count_sparse','status':'pending'} for s in p['training_seeds']]
        jobs += [{'job_id':f'k{k}-s{s}-normalized_sparse','k':k,'seed':s,'arm':'normalized_sparse','status':'pending'} for k in p['k'] for s in p['training_seeds']]
        save(OUT/'queue.json',jobs);return
    lock=(ROOT/f'artifacts/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(UP),str(UP/'experiments')]
    import torch
    import numpy as np
    import sudoku as CF
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS,cram_sudoku_sparse as SP
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for k in (10,17,21):
        for seed in range(6):
            r=json.loads((BASE/'sparse'/f'preflight-k{k}-s{seed}-n256-c64.json').read_text());assert r['status']=='passed' and r['backend_sha256']==sha(HERE/'sparse_backend.py')
    while True:
        j=queue(gpu=a.gpu)
        if j is None:break
        try:run(j,a.gpu,p,torch,np,CF,GS,SS,SP)
        except BaseException:queue(j['job_id'],'failed');raise
        queue(j['job_id'],'complete')


if __name__=='__main__':main()
