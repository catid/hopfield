"""Exactly one resumed check for cross-host continuation validation."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import platform
import shutil
import sys
from run import ROOT,HERE,UP,save,sha,now


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
    base=ROOT/'artifacts/research/20261114/ripper'
    for seed in list(range(6))[a.gpu::4]:
        src=base/f'finaltest/jobs/k10-s{seed}-normalized_sparse';d=base/f'finaltest/resume-proof/k10-s{seed}';d.mkdir(parents=True,exist_ok=False)
        r=json.loads((src/'receipt.json').read_text());model=base/f'scale/jobs/k10-s{seed}-normalized_unit/searched_model.pt'
        sd=torch.load(model,weights_only=True,map_location='cuda');S=CF.full_S(sd['W'],CF.pair_index('cuda'));unit=r['used_hbar'];m=r['statistics'];sol,free,_=SS.load_puzzles('full-all',30000,10000)
        s=sparse_search_class(GS,SP)(S,sd['B'],sol,free,temp=.4*unit,theta=.45*unit,focused=True,C=4,tau=m['tau_train'],tol=.001,ebar=m['ebar'],cap=4e11,check=4096,seed=seed,anneal=(.4*unit,.31*unit,524288))
        before=torch.load(src/'final_state.pt',weights_only=True,map_location='cpu');capture=torch.load(src/'capture_state.pt',weights_only=True,map_location='cpu')
        s.load_state_dict(before)
        s.best_boards=capture['best_boards'].to(s.dev);s.cert_boards=capture['cert_boards'].to(s.dev)
        s.T_cap=16777216;s.cap=s.T_cap*s.fl_move*4
        assert s.t==262144;s.step_check();torch.cuda.synchronize();assert s.t==266240
        state=s.state_dict();capture={'best_boards':s.best_boards.cpu(),'cert_boards':s.cert_boards.cpu()}
        fields={}
        for name,value in {**state,**capture}.items():
            if torch.is_tensor(value):
                fields[name]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
            else:fields[name]={'value':value}
        for name,value in [('state',state),('capture',capture)]:
            torch.save(value,d/f'{name}.pt')
            with (d/f'{name}.pt').open('rb') as fi,gzip.open(d/f'{name}.pt.gz','wb',compresslevel=1) as fo:shutil.copyfileobj(fi,fo)
        rec={'status':'passed_one_check','utc':now(),'seed':seed,'host':platform.node(),'gpu':a.gpu,'before_move':262144,'after_move':s.t,'T_cap':s.T_cap,'cap':s.cap,'check':s.check,'C':s.C,'source_state_sha256':sha(src/'final_state.pt'),'source_capture_sha256':sha(src/'capture_state.pt'),'source_receipt_sha256':sha(src/'receipt.json'),'model_sha256':sha(model),'backend_sha256':sha(HERE/'sparse_backend.py'),'wrapper_sha256':sha(__file__),'state_fields':fields,'output_sha256':{n:sha(d/n) for n in ['state.pt','capture.pt','state.pt.gz','capture.pt.gz']},'scope':'Exactly one production step_check from frozen prefix; elapsed is unchanged because step_check does not update walltime. No prefix replay.'}
        save(d/'receipt.json',rec);print(seed,'complete',flush=True)


if __name__=='__main__':main()
