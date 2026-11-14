"""Supplementary parity at individual transitions and before float refresh.

This strengthens bounded evidence without changing any production model or search.
"""
import argparse
import fcntl
import json
import os
import sys
import platform
from run import ROOT,HERE,UP,save,sha,now,captured_search_class


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args();assert platform.node()=='ripper'
    lock=(ROOT/f'artifacts/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path[:0]=[str(UP),str(UP/'experiments')]
    import torch
    import sudoku as CF
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS,cram_sudoku_sparse as SP
    from sparse_backend import sparse_search_class
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cells=[('mechanism',10,0,'canonical_repulsive'),('mechanism',17,0,'canonical_repulsive'),('scale',10,0,'normalized_unit'),('gap',21,5,'gap_pruned')]
    phase,k,s,arm=cells[a.gpu];base=ROOT/'artifacts/research/20261114/ripper';src=base/f'{phase}/jobs/k{k}-s{s}-{arm}'
    r=json.loads((src/'receipt.json').read_text());sd=torch.load(src/'searched_model.pt',weights_only=True,map_location='cuda');S=CF.full_S(sd['W'],CF.pair_index('cuda'));B=sd['B'];unit=r['used_hbar'];m=r['statistics'];sol,free,_=SS.load_puzzles('dev10k',4608,32)
    out=base/'sparse'/f'transitions-gpu{a.gpu}.json';assert not out.exists()
    rec={'status':'running','started_utc':now(),'source':str(src.relative_to(ROOT)),'backend_sha256':sha(HERE/'sparse_backend.py'),'wrapper_sha256':sha(__file__),'model_sha256':sha(src/'searched_model.pt'),'individual_move_comparisons':0,'production_pre_refresh_comparisons':0,'state_fields':25,'value_mismatches':[],'byte_mismatches':[]}
    kw=dict(temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,seed=s,anneal=(.4*unit,.31*unit,524288))
    def compare(xs,ys,label):
        assert xs.keys()==ys.keys() and len(xs)==25
        for key,u in xs.items():
            v=ys[key]
            eq=torch.equal(u,v) if torch.is_tensor(u) else u==v
            if not eq:rec['value_mismatches'].append({'label':label,'field':key});save(out,rec);raise AssertionError((label,key))
            if torch.is_tensor(u) and not torch.equal(u.contiguous().view(torch.uint8),v.contiguous().view(torch.uint8)):
                rec['byte_mismatches'].append({'label':label,'field':key})
    try:
        # check=1 changes random seed each transition, exercising single-move field arithmetic.
        x=captured_search_class(GS)(S,B,sol,free,check=1,**kw);y=sparse_search_class(GS,SP)(S,B,sol,free,check=1,**kw)
        compare(x.state_dict(),y.state_dict(),'initial')
        for move in range(1024):
            x._moves(1);y._moves(1);x.t+=1;y.t+=1
            compare(x.state_dict(),y.state_dict(),f'move{move+1}');rec['individual_move_comparisons']+=1
        # Production check=4096, capture fields immediately before each refresh can erase drift.
        x=captured_search_class(GS)(S,B,sol,free,check=4096,**kw);y=sparse_search_class(GS,SP)(S,B,sol,free,check=4096,**kw)
        xr,yr=x._periodic_refresh,y._periodic_refresh;pending=[]
        def refresh_x():pending.append(x.state_dict());xr()
        def refresh_y():compare(pending.pop(),y.state_dict(),f'pre-refresh{y.t}');rec['production_pre_refresh_comparisons']+=1;yr()
        x._periodic_refresh=refresh_x;y._periodic_refresh=refresh_y
        for check in range(64):
            x.step_check();y.step_check();compare(x.state_dict(),y.state_dict(),f'post-check{check}')
            assert torch.equal(x.best_boards,y.best_boards) and torch.equal(x.cert_boards,y.cert_boards)
            if not len(x.act):break
        assert not rec['byte_mismatches'],rec['byte_mismatches']
        rec.update(status='passed',completed_utc=now())
    except BaseException as e:rec.update(status='failed',error=repr(e),completed_utc=now());save(out,rec);raise
    save(out,rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
