"""Exact full-state parity gate before any sparse-backend speed claim."""
import argparse
import fcntl
import json
import os
import platform
import sys
from run import ROOT, HERE, UP, save, now, sha, captured_search_class


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--k',type=int,default=17);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--n',type=int,default=16);ap.add_argument('--checks',type=int,default=4)
    a=ap.parse_args();assert platform.node()=='ripper'
    lock=(ROOT/f'artifacts/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(UP),str(UP/'experiments')]
    import torch
    from cram import cram_sudoku_general as GS, cram_sudoku_search as SS, cram_sudoku_sparse as SP
    from sparse_backend import sparse_search_class
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    base=ROOT/'artifacts/research/20261114/ripper'
    src=base/'mechanism/jobs'/f'k{a.k}-s{a.seed}-canonical_repulsive'
    sd=torch.load(src/'searched_model.pt',weights_only=True,map_location='cuda')
    r=json.loads((src/'receipt.json').read_text());unit=r['used_hbar'];m=r['statistics']
    import sudoku as CF
    S=CF.full_S(sd['W'],CF.pair_index('cuda'));B=sd['B']
    sol,free,_=SS.load_puzzles('dev10k',4608,a.n)
    kw=dict(temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=a.seed,anneal=(.4*unit,.31*unit,524288))
    out=base/'sparse'/f'preflight-k{a.k}-s{a.seed}-n{a.n}-c{a.checks}.json'
    assert not out.exists(),out
    rec={'status':'running','started_utc':now(),'host':platform.node(),'gpu':a.gpu,'arguments':vars(a),'backend_sha256':sha(HERE/'sparse_backend.py'),'wrapper_sha256':sha(__file__),'model_sha256':sha(src/'searched_model.pt'),'checks':[]}
    try:
        SP.SparseSearch(S[:,:,0,0]/2,torch.zeros(10,9,device='cuda'),sol,free,**kw)
        rec['stock_constructor']='accepted'
    except AssertionError as e:
        rec['stock_constructor']=str(e)
    x=captured_search_class(GS)(S,B,sol,free,**kw)
    y=sparse_search_class(GS,SP)(S,B,sol,free,**kw)
    rec['D']=x.D
    def check(label):
        xs,ys=x.state_dict(),y.state_dict();bad=[]
        assert xs.keys()==ys.keys()
        for field in xs:
            u,v=xs[field],ys[field]
            ok=torch.equal(u,v) if torch.is_tensor(u) else u==v
            if not ok:
                d={'field':field}
                if torch.is_tensor(u):
                    d.update(shape_x=list(u.shape),shape_y=list(v.shape))
                    if u.shape==v.shape:
                        d.update(mismatches=int((u!=v).sum()),max_abs=float((u.double()-v.double()).abs().max()))
                bad.append(d)
        for field in ['best_boards','cert_boards']:
            if not torch.equal(getattr(x,field),getattr(y,field)):bad.append({'field':field})
        rec['checks'].append({'label':label,'state_fields':len(xs),'mismatches':bad})
        save(out,rec)
        assert not bad,bad
    try:
        check('initial')
        for i in range(a.checks):
            x.step_check();y.step_check();check(f'check{i+1}')
            if not len(x.act):break
        rec.update(status='passed',completed_utc=now())
    except BaseException as e:
        rec.update(status='failed',error=repr(e),completed_utc=now());save(out,rec);raise
    save(out,rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
