"""Eight new incomplete banks with unchanged H3ad BP AdamW fitting functions."""
import argparse
import fcntl
import math
import os
import socket
import subprocess
import time
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load

def prepare():
    import numpy as np
    p=frozen();helper=load(SRC.parent/'h3ad/source.py','h3an_matched_source_functions');rows=[]
    for family,cfg in enumerate(p['families']):
        columns=np.load(ROOT/cfg['source_path']);assert columns.shape==(cfg['source_population'],12);population=np.zeros((len(columns),144),bool);population[np.arange(len(columns))[:,None],12*np.arange(12)[None,:]+columns]=True
        for bank in range(4):
            index=4*family+bank;seed=p['training']['bank_seeds'][index];indices=np.sort(np.random.default_rng(seed).choice(len(columns),128,replace=False));complement=np.setdiff1d(np.arange(len(columns)),indices);d=helper.design(population[indices]);d['population_indices']=indices;verification=helper.independent_design(d);folder=OUT/f'b{index}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'source-design.npz',**d);np.save(folder/'acquisition-population-indices.npy',complement);np.save(folder/'acquisition-population.npy',population[complement]);rows.append(dict(index=index,family=family,bank=bank,seed=seed,training_indices=indices.tolist(),complement_size=len(complement),source_design_sha256=sha(folder/'source-design.npz'),acquisition_population_sha256=sha(folder/'acquisition-population.npy'),**verification))
    save(OUT/'source-banks.json',dict(status='eight_frozen_draws_complete',utc=now(),banks=rows,within_family_overlaps={str(f):[[len(set(rows[4*f+a]['training_indices'])&set(rows[4*f+b]['training_indices']))for b in range(4)]for a in range(4)]for f in range(2)},protocol_sha256=sha(SRC/'protocol.json')))

def worker(gpu):
    started=time.perf_counter();os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    p=frozen();assert socket.gethostname()=='loki';lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);query=lambda *args:subprocess.check_output(args,text=True,timeout=20).strip();device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');assert device.split(',')[1].strip()not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader');torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;helper=load(SRC.parent/'h3ad/source.py','h3an_matched_fit_functions');original=load(SRC.parent/'h3u/driver.py','h3an_original_PL');receipt=OUT/f'fit-worker-g{gpu}.json';base=dict(gpu=gpu,pid=os.getpid(),device=device,started_utc=now());completed=[]
    def progress(**values):save(receipt,dict(base,status='running',utc=now(),elapsed_seconds=time.perf_counter()-started,progress=values,completed=completed))
    try:
        for index in range(gpu,8,2):
            folder=OUT/f'b{index}';d=dict(np.load(folder/'source-design.npz'));progress(bank=index,stage='loss_and_gradient_gate');validation=helper.validate_loss(d,'cuda');save(folder/'loss-validation.json',validation);torch.manual_seed(0);y=torch.tensor(d['labels'],device='cuda',dtype=torch.float32);weights=torch.where(y.bool(),231.,298.);weights/=weights.sum();groups=torch.tensor(d['pair_groups'],device='cuda');A=torch.zeros(int(d['groups'].max())+1,device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True);opt=torch.optim.AdamW([A,b],lr=.3,weight_decay=.01,betas=(.9,.999),eps=1e-8,foreach=False);torch.cuda.synchronize();start=time.perf_counter();curve=[];progress(bank=index,stage='fit',update=0)
            for k in range(4000):
                if k%100==0 and time.perf_counter()-start>30:raise TimeoutError('Frozen30second individual fit ceiling')
                opt.zero_grad(set_to_none=True);loss=original.source_loss(A,b,y,weights,groups);loss.backward();opt.param_groups[0]['lr']=.3*.5*(1+math.cos(math.pi*k/4000));opt.step()
                if(k+1)%250==0:curve.append(dict(update=k+1,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start));progress(bank=index,stage='fit',**curve[-1])
            torch.cuda.synchronize();seconds=time.perf_counter()-start;AA=A.detach().cpu();bb=b.detach().cpu();np.savez_compressed(folder/'parameters.npz',A=AA.numpy(),b=bb.numpy());model,info=helper.construct(AA,bb,d);np.savez_compressed(folder/'model.npz',**model);record=dict(index=index,status='complete'if info['source_eligible']else'source_fit_ineligible',utc=now(),gpu=gpu,fit_seconds=seconds,updates=4000,curve=curve,**info,parameters_sha256=sha(folder/'parameters.npz'),model_sha256=sha(folder/'model.npz'),source_design_sha256=sha(folder/'source-design.npz'),protocol_sha256=sha(SRC/'protocol.json'));save(folder/'model.json',record);completed.append(dict(index=index,status=record['status'],fit_seconds=seconds));print('FIT',index,seconds,record['status'],flush=True)
        save(receipt,dict(base,status='complete',utc=now(),elapsed_seconds=time.perf_counter()-started,completed=completed))
    except BaseException as exc:save(receipt,dict(base,status='failed',utc=now(),elapsed_seconds=time.perf_counter()-started,completed=completed,error=repr(exc)));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args();prepare()if args.mode=='prepare'else worker(args.gpu)
