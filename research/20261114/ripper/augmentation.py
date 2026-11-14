"""H1k: frozen augmentation ablation with unchanged fit recipe."""
import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from run import ROOT,UP,HERE,save,sha,now,captured_search_class,independent_validity,projection

BASE=ROOT/'artifacts/research/20261114/ripper'
OUT=BASE/'augmentation'
PROTOCOL=HERE/'protocol-augmentation.json'


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
    entry=next(e for e in p['training_inputs'] if e['seed']==seed and e['arm']==arm)
    trainpath=ROOT/entry['path'];assert sha(trainpath)==entry['sha256'];tr=np.load(trainpath)
    train=torch.as_tensor(tr['solutions'].astype(np.int64),device='cuda');frtrain=torch.as_tensor(tr['free'],device='cuda')
    fitconfig=json.loads((BASE/'inputs'/f'k10-s{seed}'/'fit.json').read_text())['config']
    cfg=CF.FullConfig(**fitconfig)
    if arm=='full_spatial':
        src=BASE/'scale/jobs'/f'k10-s{seed}-normalized_unit'
        old=torch.load(src/'searched_model.pt',weights_only=True,map_location='cuda')
        W,B=old['W'],old['B'];level=float(W[W>0][0]);transform={'control_reused':True,'level':level}
        fitted=torch.load(BASE/'inputs'/f'k10-s{seed}'/'model.pt',weights_only=True,map_location='cpu');fit_s=0.0
    else:
        Wfit,Bfit,fit_s=CF.train(cfg,train,frtrain);fitted={'W':Wfit.cpu(),'B':Bfit.cpu()}
        rawW,rawB=projection(Wfit,Bfit);eye=torch.eye(9,dtype=torch.bool,device='cuda')
        J=(rawW[:,eye].mean(-1)-rawW[:,~eye].mean(-1)).clamp_min(0)
        positive=J[J>0].sort().values;assert len(positive)>=2,'No eligible coefficient gap'
        gaps=positive[1:]-positive[:-1];ix=int(gaps.argmax());assert float(gaps[ix])>0,'No positive coefficient gap'
        threshold=float((positive[ix].double()+positive[ix+1].double())/2);keep=J>threshold
        level=float(J[keep].mean());J=torch.where(keep,torch.full_like(J,level),torch.zeros_like(J))
        W=(J[:,None,None]*eye).contiguous();B=torch.zeros_like(Bfit)
        transform={'gap':float(gaps[ix]),'threshold':threshold,'retained':int(keep.sum()),'level':level}
    m=CF.prepare(W,B,train);unit=level;lo,hi=p['rows_half_open'];sol,free,idx=SS.load_puzzles('full-all',lo,hi-lo)
    kw=dict(temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=seed,anneal=(.4*unit,.31*unit,524288))
    klass=sparse_search_class(GS,SP)
    # General search remains the reference for all newly learned support patterns.
    x=captured_search_class(GS)(m['S'],B,sol[:16],free[:16],**kw);y=klass(m['S'],B,sol[:16],free[:16],**kw)
    for check in range(5):
        xs,ys=x.state_dict(),y.state_dict()
        for key,u in xs.items():assert torch.equal(u,ys[key]) if torch.is_tensor(u) else u==ys[key],key
        assert torch.equal(x.best_boards,y.best_boards) and torch.equal(x.cert_boards,y.cert_boards)
        if check<4:x.step_check();y.step_check()
    del x,y
    ss=klass(m['S'],B,sol,free,**kw);ss.T_cap=262144;ss.cap=ss.T_cap*ss.fl_move*4
    rec={'status':'running','job_id':j['job_id'],'k':k,'seed':seed,'arm':arm,'gpu':gpu,'host':platform.node(),'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'backend_sha256':sha(HERE/'sparse_backend.py'),'fit_config':fitconfig,'fit_s':fit_s,'training_sha256':sha(trainpath),'used_hbar':unit,'statistics':{key:m[key] for key in ['tau_train','ebar','hbar','D']},'transform':transform,'preflight_checks':5,'preflight_state_fields':25,'move_cap':262144,'chains':4,'rows_half_open':[lo,hi],'test_rows':len(sol),'fl_move':ss.fl_move,'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec);torch.save(fitted,dest/'fitted_model.pt');torch.save({'W':W.cpu(),'B':B.cpu()},dest/'searched_model.pt');last=[time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0]>=1 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,'active':len(s.act),'elapsed_s':s.elapsed});last[0]=time.monotonic()
    start=time.monotonic();result=ss.run(progress,log_every=65536);torch.cuda.synchronize();wall=time.monotonic()-start
    boards=ss.boards().cpu().numpy().astype(np.int8);exact=(boards==sol).all(1);valid=independent_validity(boards,sol,free)
    assert np.array_equal(exact,result['solved']) and np.array_equal(exact,valid)
    stopped=ss.stopped.cpu().numpy();moves=torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,puzzle=idx,stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=ss.t,solved=int(exact.sum()),valid=int(valid.sum()),wrong_stops=int((stopped&~exact).sum()),budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),output_sha256={n:sha(dest/n) for n in ['boards.npz','result.json','searched_model.pt','fitted_model.pt']})
    save(dest/'receipt.json',rec);print(json.dumps({'job_id':j['job_id'],'solved':rec['solved'],'wrong_stops':rec['wrong_stops'],'wall_s':wall}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int);ap.add_argument('--initialize',action='store_true');ap.add_argument('--preflight',action='store_true');a=ap.parse_args();assert platform.node()=='ripper'
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
    if a.preflight:
        src=BASE/'inputs/k10-s0';z=np.load(src/'train.npz');sol=torch.as_tensor(z['solutions'].astype(np.int64),device='cuda');free=torch.as_tensor(z['free'],device='cuda')
        cfg=CF.FullConfig(**json.loads((src/'fit.json').read_text())['config']);W,B,seconds=CF.train(cfg,sol,free);old=torch.load(src/'model.pt',weights_only=True,map_location='cuda')
        checks={'W':torch.equal(W,old['W']),'B':torch.equal(B,old['B'])}
        save(OUT/'fit-preflight.json',{'utc':now(),'checks':checks,'fit_source_sha256':sha(UP/'experiments/sudoku.py'),'model_sha256':sha(src/'model.pt'),'training_sha256':sha(src/'train.npz'),'seconds':seconds})
        assert all(checks.values()),checks
        print('Full-spatial seed0 refit matches frozen parameters exactly',flush=True);return
    proof=json.loads((OUT/'fit-preflight.json').read_text());assert all(proof['checks'].values()) and proof['fit_source_sha256']==sha(UP/'experiments/sudoku.py')
    while True:
        j=claim(a.gpu)
        if j is None:break
        try:run(j,a.gpu,p,torch,np,CF,GS,SS,SP)
        except BaseException:claim(a.gpu,j['job_id'],'failed');raise
        claim(a.gpu,j['job_id'],'complete')


if __name__=='__main__':main()
