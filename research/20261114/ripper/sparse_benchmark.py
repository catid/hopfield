"""Frozen H1d paired GPU timing; independent cells are dynamically queued."""
import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from run import ROOT, HERE, UP, save, now, sha, captured_search_class

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'sparse'


def claim(gpu):
    with (OUT/'benchmark.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        q=json.loads((OUT/'benchmark-queue.json').read_text())
        j=next((j for j in q if j['status']=='pending'),None)
        if j is not None:j.update(status='running',gpu=gpu,pid=os.getpid(),started_utc=now())
        save(OUT/'benchmark-queue.json',q)
        return j


def finish(j):
    with (OUT/'benchmark.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        q=json.loads((OUT/'benchmark-queue.json').read_text())
        old=next(x for x in q if x['id']==j['id']);old.update(status='complete',completed_utc=now())
        save(OUT/'benchmark-queue.json',q)


def run(j,torch,SS,GS,SP,CF):
    from sparse_backend import sparse_search_class
    src=BASE/'mechanism/jobs'/f'k{j["k"]}-s0-canonical_repulsive'
    r=json.loads((src/'receipt.json').read_text());unit=r['used_hbar'];m=r['statistics']
    sd=torch.load(src/'searched_model.pt',weights_only=True,map_location='cuda')
    S=CF.full_S(sd['W'],CF.pair_index('cuda'));B=sd['B']
    sol,free,_=SS.load_puzzles('dev10k',4608,j['n'])
    kw=dict(temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=0,anneal=(.4*unit,.31*unit,524288))
    types={'general':captured_search_class(GS),'sparse':sparse_search_class(GS,SP)}
    dest=OUT/'benchmark'/j['id'];dest.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','cell':j,'started_utc':now(),'backend_sha256':sha(HERE/'sparse_backend.py'),'wrapper_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol-sparse.json'),'model_sha256':sha(src/'searched_model.pt'),'hardware':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'runs':[],'pairs':[]}
    def execute(backend,repeat):
        torch.cuda.synchronize();begin=time.perf_counter();s=types[backend](S,B,sol,free,**kw)
        s.T_cap=262144;s.cap=s.T_cap*s.fl_move*4
        torch.cuda.synchronize();ready=time.perf_counter();events=[];calls=[0]
        move=s._moves
        def timed_move(n):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();move(n);b.record();events.append((a,b));calls[0]+=len(s.y)*n
        s._moves=timed_move
        # Production checks/compaction/refresh, without logging or progress I/O.
        while not s.done():s.step_check()
        torch.cuda.synchronize();end=time.perf_counter()
        rec={'backend':backend,'repeat':repeat,'setup_s':ready-begin,'inference_s':end-ready,'complete_s':end-begin,'moves_gpu_s':sum(a.elapsed_time(b) for a,b in events)/1000,'issued_chain_moves':calls[0],'solved':int(s.result()['solved'].count(True)),'terminal_move':s.t,'D':s.D}
        receipt['runs'].append(rec);save(dest/'receipt.json',receipt)
        return s.state_dict(),s.boards().cpu(),s.best_boards.cpu(),s.cert_boards.cpu()
    for rep in range(-1,6):
        order=['general','sparse'] if rep%2==0 else ['sparse','general'];states={}
        for backend in order:states[backend]=execute(backend,rep)
        x,y=states['general'],states['sparse'];bad=[]
        for field,u in x[0].items():
            v=y[0][field]
            if torch.is_tensor(u):ok=torch.equal(u,v)
            else:ok=u==v
            if not ok:bad.append(field)
        for i in range(1,4):
            if not torch.equal(x[i],y[i]):bad.append(f'boards{i}')
        receipt['pairs'].append({'repeat':rep,'exact_state_and_boards':not bad,'mismatches':bad})
        save(dest/'receipt.json',receipt);assert not bad,bad
    receipt.update(status='complete',completed_utc=now());save(dest/'receipt.json',receipt)
    print(j['id'],'complete',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int);ap.add_argument('--initialize',action='store_true');a=ap.parse_args()
    assert platform.node()=='ripper'
    if a.initialize:
        assert not (OUT/'benchmark-queue.json').exists()
        save(OUT/'benchmark-queue.json',[{'id':f'k{k}-n{n}','k':k,'n':n,'status':'pending'} for n in (2048,512,128) for k in (10,17,21)]);return
    lock=(ROOT/f'artifacts/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True);assert info not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(UP),str(UP/'experiments')]
    import torch
    import sudoku as CF
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS,cram_sudoku_sparse as SP
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    for k in (10,17,21):
        for seed in range(6):
            p=json.loads((OUT/f'preflight-k{k}-s{seed}-n256-c64.json').read_text())
            assert p['status']=='passed' and p['backend_sha256']==sha(HERE/'sparse_backend.py')
    while True:
        j=claim(a.gpu)
        if j is None:break
        run(j,torch,SS,GS,SP,CF);finish(j)


if __name__=='__main__':main()
