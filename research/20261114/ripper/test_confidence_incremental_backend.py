"""Nonuniform-weight parity and exact stopping/field checks on old/synthetic inputs."""
import fcntl
import json
import os
import subprocess
import sys
import time
from run import ROOT,HERE,UP,sha,save,now,captured_search_class


def main():
    start=time.monotonic();lock=(ROOT/'artifacts/gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0,str(UP));import torch;import numpy as np
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS,cram_sudoku_sparse as SP
    from sparse_backend import sparse_search_class
    from confidence_integer_audit import check_state
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    pair=np.triu_indices(81,1);i,j=pair;rng=np.random.default_rng(20403100)
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    standard=SS.load_puzzles('full-all',0,16);sol,free,_=standard
    cases=[]
    for kind in range(3):
        if kind==0:w=np.where(peer| (rng.random(3240)<.03),rng.choice([1,16],3240),0).astype(np.uint8)
        elif kind==1:w=np.where(rng.random(3240)<.3,rng.choice([1,16],3240),0).astype(np.uint8)
        else:w=np.where(peer,1,0).astype(np.uint8)
        cases.append((f'nonuniform-{kind}',w,sol,free))
    for conflict in (False,True):
        w=np.zeros(3240,np.uint8);w[0]=1;target=np.zeros((1,81),np.int8);target[0,1]=0 if conflict else 1
        fr=np.zeros((1,81),bool);fr[:,80]=True;cases.append((f'minimum-energy-{conflict}',w,target,fr))
    general=sparse_search_class(GS,SP);parity=hidden=states=chains=0;reports=[]
    for name,w,target,fr in cases:
        J=torch.as_tensor(w.astype(np.float32)/16,device='cuda');W=J[:,None,None]*torch.eye(9,device='cuda')
        S=torch.zeros((81,81,9,9),device='cuda');ii=torch.triu_indices(81,81,1,device='cuda');S[ii[0],ii[1]]=W;S[ii[1],ii[0]]=W.transpose(1,2);B=torch.zeros((81,9),device='cuda')
        kw=dict(temp=.4,theta=.45,focused=True,C=4,tau=0.,tol=.001,ebar=0.,cap=4e11,check=4096,seed=0,anneal=(.4,.31,524288))
        x=captured_search_class(GS)(S,B,target,fr,**kw);y=general(S,B,target,fr,**kw)
        alt=target.copy();alt[fr]=rng.integers(0,9,int(fr.sum()));z=general(S,B,alt,fr,**kw)
        for check in range(6):
            a,b,c=x.state_dict(),y.state_dict(),z.state_dict();assert a.keys()==b.keys()==c.keys()
            for key,value in a.items():
                assert torch.equal(value,b[key]) if torch.is_tensor(value) else value==b[key],key;parity+=1
                if key not in ('tgt','ok','bestok'):
                    assert torch.equal(b[key],c[key]) if torch.is_tensor(b[key]) else b[key]==c[key],key;hidden+=1
            chains+=check_state(y,w);states+=1
            assert torch.equal(x.best_boards,y.best_boards) and torch.equal(x.cert_boards,y.cert_boards)
            assert torch.equal(y.best_boards,z.best_boards) and torch.equal(y.cert_boards,z.cert_boards)
            if check<4:x.step_check();y.step_check();z.step_check()
            elif check==4:
                for state in (x,y,z):state._moves(127);state.t+=127
        if name=='minimum-energy-True':assert not y.stopped.any() and torch.all(y.P==1/16)
        if name=='minimum-energy-False':assert y.stopped.all()
        reports.append({'case':name,'weight_values':sorted(set((w.astype(float)/16).tolist())),'stopped':int(y.stopped.sum()),'puzzles':len(target)})
        del x,y,z
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    report={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'integer_audit_sha256':sha(HERE/'confidence_integer_audit.py'),
        'backend_sha256':sha(HERE/'sparse_backend.py'),'capture_sha256':sha(HERE/'run.py'),'old_inputs_only':[0,16],
        'general_sparse_state_comparisons':parity,'hidden_target_state_comparisons':hidden,'integer_state_checks':states,'integer_chain_states':chains,
        'incremental_moves_without_refresh':127,'minimum_positive_energy_does_not_stop':True,'zero_energy_does_stop':True,'reports':reports,'gpu':0,'allocated_gpu_seconds':elapsed,
        'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name()}
    save(ROOT/'artifacts/research/20261114/ripper/confidence_incremental_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':main()
