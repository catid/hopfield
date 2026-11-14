"""H1g: fixed-graph temperature/focus scale normalization."""
import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from run import ROOT,UP,HERE,save,sha,now,captured_search_class,independent_validity

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'scale'
PROTOCOL=HERE/'protocol-scale.json'


def claim(gpu,job_id=None,status=None):
    with (OUT/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);q=json.loads((OUT/'queue.json').read_text())
        if job_id is None:
            j=next((j for j in q if j['status']=='pending'),None)
            if j is not None:j.update(status='running',gpu=gpu,pid=os.getpid(),started_utc=now())
        else:j=next(j for j in q if j['job_id']==job_id);j.update(status=status,completed_utc=now())
        save(OUT/'queue.json',q);return j


def run(j,gpu,p,torch,np,CF,GS,SS,SP):
    from sparse_backend import sparse_search_class
    k,seed,arm=j['k'],j['seed'],j['arm'];dest=OUT/'jobs'/j['job_id'];dest.mkdir(parents=True,exist_ok=False)
    src=BASE/'gap/jobs'/f'k{k}-s{seed}-gap_uniform'
    rr=json.loads((src/'receipt.json').read_text());assert sha(src/'searched_model.pt')==rr['output_sha256']['searched_model.pt']
    sd=torch.load(src/'searched_model.pt',weights_only=True,map_location='cuda')
    J=sd['W'][:,0,0];level=float(J[J>0][0]);assert bool((J[J>0]==level).all())
    assert arm in ('inherited_unit','normalized_unit')
    transform={'inherited_unit':rr['used_hbar'],'normalized_unit':level,'unit_ratio':rr['used_hbar']/level,'positive_edges':int((J>0).sum())}
    W=sd['W'];B=sd['B']
    tr=np.load(BASE/'inputs'/f'k{k}-s{seed}'/'train.npz');train=torch.as_tensor(tr['solutions'].astype(np.int64),device='cuda')
    m=CF.prepare(W,B,train);unit=rr['used_hbar'] if arm=='inherited_unit' else level;lo,hi=p['rows_half_open'];sol,free,idx=SS.load_puzzles('dev10k',lo,hi-lo)
    kw=dict(temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=seed,anneal=(.4*unit,.31*unit,524288))
    klass=sparse_search_class(GS,SP)
    # Both fixed-weight scale settings pass bounded numerical backend parity.
    x=captured_search_class(GS)(m['S'],B,sol[:16],free[:16],**kw);y=klass(m['S'],B,sol[:16],free[:16],**kw)
    for check in range(5):
        xs,ys=x.state_dict(),y.state_dict()
        assert xs.keys()==ys.keys()
        for key,u in xs.items():assert torch.equal(u,ys[key]) if torch.is_tensor(u) else u==ys[key],key
        assert torch.equal(x.best_boards,y.best_boards) and torch.equal(x.cert_boards,y.cert_boards)
        if check<4:x.step_check();y.step_check()
    del x,y
    ss=klass(m['S'],B,sol,free,**kw);ss.T_cap=262144;ss.cap=ss.T_cap*ss.fl_move*4
    rec={'status':'running','job_id':j['job_id'],'k':k,'seed':seed,'arm':arm,'gpu':gpu,'host':platform.node(),'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'backend_sha256':sha(HERE/'sparse_backend.py'),'source_model_sha256':sha(src/'searched_model.pt'),'training_sha256':sha(BASE/'inputs'/f'k{k}-s{seed}'/'train.npz'),'used_hbar':unit,'statistics':{key:m[key] for key in ['tau_train','ebar','hbar','D']},'transform':transform,'preflight_checks':5,'preflight_state_fields':25,'move_cap':262144,'chains':4,'rows_half_open':[lo,hi],'test_rows':len(sol),'fl_move':ss.fl_move,'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec);torch.save({'W':W.cpu(),'B':B.cpu()},dest/'searched_model.pt');last=[time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0]>=1 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,'active':len(s.act),'elapsed_s':s.elapsed});last[0]=time.monotonic()
    start=time.monotonic();result=ss.run(progress,log_every=65536);torch.cuda.synchronize();wall=time.monotonic()-start
    boards=ss.boards().cpu().numpy().astype(np.int8);exact=(boards==sol).all(1);valid=independent_validity(boards,sol,free)
    assert np.array_equal(exact,result['solved']) and np.array_equal(exact,valid)
    stopped=ss.stopped.cpu().numpy();moves=torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,puzzle=idx,stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=ss.t,solved=int(exact.sum()),valid=int(valid.sum()),wrong_stops=int((stopped&~exact).sum()),budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),output_sha256={n:sha(dest/n) for n in ['boards.npz','result.json','searched_model.pt']})
    save(dest/'receipt.json',rec);print(json.dumps({'job_id':j['job_id'],'solved':rec['solved'],'wrong_stops':rec['wrong_stops'],'wall_s':wall}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int);ap.add_argument('--initialize',action='store_true');a=ap.parse_args();assert platform.node()=='ripper'
    p=json.loads(PROTOCOL.read_text());OUT.mkdir(exist_ok=True)
    if a.initialize:
        assert not (OUT/'queue.json').exists()
        save(OUT/'queue.json',[{'job_id':f'k{k}-s{s}-{arm}','k':k,'seed':s,'arm':arm,'status':'pending'} for k in p['k'] for s in p['training_seeds'] for arm in p['arms']]);return
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
    while True:
        j=claim(a.gpu)
        if j is None:break
        try:run(j,a.gpu,p,torch,np,CF,GS,SS,SP)
        except BaseException:claim(a.gpu,j['job_id'],'failed');raise
        claim(a.gpu,j['job_id'],'complete')


if __name__=='__main__':main()
