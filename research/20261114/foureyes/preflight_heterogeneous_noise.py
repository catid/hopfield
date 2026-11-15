"""Synthetic checks only; no H4ad random bank/profile draws or outcomes."""
from pathlib import Path
import json
import os
import time
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
import torch
from heterogeneous_noise_study import ROOT,HERE,sha,save,corrupt
from audit_heterogeneous_noise import numpy_raw_loss_gradient
from audit_pooled_source_study import numpy_objective
import canonical_pl as C


def main():
    start=time.monotonic();cpu=time.process_time();torch.set_num_threads(1)
    path=HERE/'heterogeneous-noise-preflight.json';assert not path.exists()
    rng=np.random.default_rng(473);labels=rng.integers(0,9,(17,81),dtype=np.uint8)
    free=rng.random((17,81))>.35;hot=np.arange(18,dtype=np.int64)
    uniforms=1-rng.random((17,81));offsets=rng.integers(1,9,(17,81),dtype=np.int64)
    counts=[]
    for condition in ['uniform','hot18_weight8']:
        noisy,cells=corrupt(np,labels,free,uniforms,offsets,hot,condition)
        for row in range(17):
            reference=sorted(range(81),key=lambda c:(float(-np.log(uniforms[row,c])/(8 if condition!='uniform' and c<18 else 1)) if free[row,c] else float('inf'),c))[:9]
            assert cells[row].tolist()==reference
            expected=labels[row].copy()
            for cell in reference:expected[cell]=(int(labels[row,cell])+int(offsets[row,cell]))%9
            assert np.array_equal(noisy[row],expected)
        assert np.array_equal(noisy[~free],labels[~free]) and ((noisy!=labels).sum(1)==9).all()
        counts.append(int((noisy!=labels).sum()))
    # Every priority ties at zero, including under unequal weights.
    ties=np.ones((17,81),np.float64)
    for condition in ['uniform','hot18_weight8']:
        _,cells=corrupt(np,labels,free,ties,offsets,hot,condition)
        assert all(row.tolist()==np.flatnonzero(mask)[:9].tolist() for row,mask in zip(cells,free))
    # Equal positive clocks must favor the higher-rate cells.
    all_free=np.ones_like(free);_,cells=corrupt(np,labels,all_free,np.full_like(uniforms,.5),offsets,hot,'hot18_weight8')
    assert np.array_equal(cells,np.tile(np.arange(9),(17,1)))
    one=sum(d==1 for d in range(1,9));both=sum(a==b for a in range(1,9) for b in [d for d in range(9) if d!=1])
    assert one==1 and both==7
    w=rng.normal(0,.03,(3240,9,9));b=rng.normal(0,.03,(81,9))
    loss,dw,db=numpy_raw_loss_gradient(w,b,labels,free)
    pair=torch.triu_indices(81,81,1);wt=torch.tensor(w,requires_grad=True);bt=torch.tensor(b,requires_grad=True)
    full=torch.zeros((81,81,9,9),dtype=torch.float64)
    full=full.index_put((pair[0],pair[1]),wt).index_put((pair[1],pair[0]),wt.transpose(1,2))
    sol=torch.tensor(labels.astype(np.int64));onehot=torch.nn.functional.one_hot(sol,9).to(torch.float64)
    field=torch.einsum('ijab,njb->nia',full,onehot)+bt
    observed=-torch.log_softmax(-field,-1).gather(2,sol[:,:,None]).squeeze(2)
    actual=observed[torch.tensor(free)].mean();actual.backward()
    le=abs(loss-float(actual.detach()));we=float(abs(dw-wt.grad.numpy()).max());be=float(abs(db-bt.grad.numpy()).max())
    assert le<1e-12 and max(we,be)<1e-12
    finite=[]
    for index in rng.choice(w.size,8,replace=False):
        x=np.unravel_index(index,w.shape);left=w.copy();right=w.copy();left[x]-=1e-5;right[x]+=1e-5
        difference=(numpy_raw_loss_gradient(right,b,labels,free)[0]-numpy_raw_loss_gradient(left,b,labels,free)[0])/2e-5
        finite.append(abs(difference-dw[x]))
    for cell,digit in [(0,0),(12,3),(80,8)]:
        left=b.copy();right=b.copy();left[cell,digit]-=1e-5;right[cell,digit]+=1e-5
        difference=(numpy_raw_loss_gradient(w,right,labels,free)[0]-numpy_raw_loss_gradient(w,left,labels,free)[0])/2e-5
        finite.append(abs(difference-db[cell,digit]))
    assert max(finite)<1e-9
    j=rng.normal(0,.03,3240);expected,gradient=numpy_objective(j,labels,free)
    actual,derivative=C.loss_gradient(torch.tensor(j),C.context(torch,sol,torch.tensor(free),torch.float64))
    ce=abs(expected-float(actual));ge=float(abs(gradient-derivative.numpy()).max());assert ce<1e-12 and ge<1e-12
    sources=[HERE/n for n in ['heterogeneous_noise_study.py','audit_heterogeneous_noise.py','preflight_heterogeneous_noise.py','canonical_pl.py','audit_pooled_source_study.py']]
    record=dict(status='passed',synthetic_rows=17,corrupted_cell_checks=counts,stable_ties=True,true_pair_noise_probabilities=['1/8','7/64'],
                raw_loss_error=le,raw_w_gradient_error=we,raw_b_gradient_error=be,max_finite_difference_error=max(finite),
                canonical_loss_error=ce,canonical_gradient_error=ge,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
                process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,scope='Synthetic-only prefreeze gate; no study source/profile RNGs or graph statistics used.')
    save(path,record);print(json.dumps(record))


if __name__=='__main__':main()
