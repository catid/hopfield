"""Independently drawn finite source banks; unchanged grouped BP AdamW fit."""
import argparse
import fcntl
import itertools
import math
import os
import signal
import socket
import subprocess
import time
from common import ROOT,SRC,OUT,BASE,sha,read,save,now,frozen,budget,load

def design(y):
    import numpy as np
    offsets=np.array([(r,c)for r in range(12)for c in range(-11,12)if r>0 or c>0]);lookup={tuple(o):k for k,o in enumerate(offsets)}
    i,j=np.triu_indices(144,1);index=np.array([lookup[b//12-a//12,b%12-a%12]for a,b in zip(i,j)])
    co=y.astype(np.int64).T@y.astype(np.int64);counts=np.bincount(index,weights=co[i,j],minlength=264).astype(np.int64);m=y.sum(0).astype(np.int64);exposure=np.bincount(index,weights=m[i]*m[j],minlength=264)/len(y);lines=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if(a==b==0)or math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
            covered=a*offsets[:,0]+b*offsets[:,1]==0
            if((counts==0)&covered).sum()>=3 and not((counts>0)&covered).any():lines.append((a,b))
    groups=np.full(264,-1,np.int64)
    for k,(a,b)in enumerate(lines):
        take=a*offsets[:,0]+b*offsets[:,1]==0;assert(groups[take]==-1).all();groups[take]=k
    off=np.flatnonzero(groups<0);groups[off]=np.arange(len(off))+len(lines)
    return dict(labels=y,offsets=offsets,counts=counts,exposure=exposure,lines=np.array(lines),groups=groups,pair_i=i,pair_j=j,pair_groups=groups[index])

def independent_design(d):
    import numpy as np
    y=d['labels'];offsets=[tuple(x)for x in d['offsets']];counts={o:0 for o in offsets}
    for board in y:
        for i,j in itertools.combinations(np.flatnonzero(board),2):counts[j//12-i//12,j%12-i%12]+=1
    lines=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if(a,b)==(0,0)or math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
            values=[counts[o]for o in offsets if a*o[0]+b*o[1]==0]
            if len(values)>=3 and not any(values):lines.append((a,b))
    assert np.array_equal(d['counts'],[counts[o]for o in offsets])and np.array_equal(d['lines'],lines)
    groups=[];nxt=len(lines);m=y.sum(0).astype(np.int64);exposure=[]
    for dr,dc in offsets:
        members=[k for k,(a,b)in enumerate(lines)if a*dr+b*dc==0];assert len(members)<=1
        if members:groups.append(members[0])
        else:groups.append(nxt);nxt+=1
        total=sum(int(m[12*r+c])*int(m[12*(r+dr)+c+dc])for r in range(12)for c in range(12)if 0<=r+dr<12 and 0<=c+dc<12);exposure.append(total/len(y))
    assert np.array_equal(groups,d['groups'])and np.array_equal(exposure,d['exposure'])
    return dict(source_patterns=len(y),lines=lines,couplings=nxt,zero_offsets=int((d['counts']==0).sum()),all_literal_counts_exposures_groups_verified=True)

def prepare():
    import numpy as np
    frozen();assert not(OUT/'source-banks.json').exists();gate=read(OUT/'enumeration-gate.json');assert gate['status']=='complete_verified';rows=[]
    for bank in range(2):
        folder=OUT/f'b{bank}';folder.mkdir(parents=True,exist_ok=True);columns=np.load(folder/'source-columns.npy');y=np.zeros((len(columns),144),bool);y[np.arange(len(y))[:,None],12*np.arange(12)[None,:]+columns]=True;d=design(y);d['population_indices']=np.arange(len(y));verification=independent_design(d);np.savez_compressed(folder/'source-design.npz',**d);rows.append(dict(bank=bank,source_design_sha256=sha(folder/'source-design.npz'),**verification))
    save(OUT/'source-banks.json',dict(status='complete_sources_after_protocol_freeze',utc=now(),banks=rows,enumeration_gate_sha256=sha(OUT/'enumeration-gate.json'),protocol_sha256=sha(SRC/'protocol.json')));print('COMPLETE_SOURCES',rows,flush=True)

def validate_loss(d,device):
    import numpy as np
    import torch
    D=load(SRC.parent/'h3u/driver.py','h3ad_original_fit');lossfn=D.source_loss;g=torch.tensor(d['pair_groups']);n=int(d['groups'].max())+1
    A=torch.linspace(-.23,.31,n,dtype=torch.float64,requires_grad=True);b=torch.tensor([-.17],dtype=torch.float64,requires_grad=True);y=torch.tensor(d['labels'][:3],dtype=torch.float64)
    subsets=[s for k in[1,2,3]for s in itertools.combinations(range(12),k)];ix=[];free=[]
    for k,board in enumerate(d['labels'][:3]):
        cells=np.flatnonzero(board)
        for subset in subsets:
            row=np.ones(144,bool);row[cells[list(subset)]]=False;free.append(row);ix.append(k)
    i,j=torch.triu_indices(144,144,1);J=A.new_zeros(144,144).index_put((i,j),A[g]).index_put((j,i),A[g]);expanded=y[ix]
    literal=torch.nn.functional.binary_cross_entropy_with_logits(-(expanded@J+b),expanded,reduction='none')[torch.tensor(np.array(free))].mean();lg=torch.autograd.grad(literal,(A,b))
    weights=torch.where(y.bool(),231.,298.).double();weights/=weights.sum();value=lossfn(A,b,y,weights,g);grad=torch.autograd.grad(value,(A,b));torch.testing.assert_close(value,literal,atol=1e-10,rtol=1e-9)
    for u,v in zip(lg,grad):torch.testing.assert_close(u,v,atol=1e-10,rtol=1e-9)
    y=torch.tensor(d['labels'],dtype=torch.float64);weights=torch.where(y.bool(),231.,298.).double();weights/=weights.sum();cpu=lossfn(A,b,y,weights,g);cg=torch.autograd.grad(cpu,(A,b));AA=A.detach().float().to(device).requires_grad_();bb=b.detach().float().to(device).requires_grad_();gpu=lossfn(AA,bb,y.float().to(device),weights.float().to(device),g.to(device));gg=torch.autograd.grad(gpu,(AA,bb));error=max(float((u-v.cpu().double()).abs().max())for u,v in zip(cg,gg));loss_error=abs(float(cpu.detach())-float(gpu.detach()));assert error<1e-5 and loss_error<1e-5
    return dict(status='passed',literal_expanded_rows=len(ix),cpu_literal_loss_error=float(abs(value.detach()-literal.detach())),cpu_gpu_loss_error=loss_error,cpu_gpu_gradient_error=error,original_loss_function_sha256=sha(SRC.parent/'h3u/driver.py'))

def construct(A,b,d):
    import numpy as np
    D=load(SRC.parent/'h3u/driver.py','h3ad_matrix');J,h=D.matrix(A,b,d,12);x=d['labels'].astype(np.float64);field=x@J.double().numpy()+h.double().numpy();native_delta=float(((1-2*x)*field).min())
    coef=A.numpy()[d['groups']];mask=(d['counts']==0)&(d['exposure']>=math.log(264/.01));mask|=(d['offsets']@d['lines'].T==0).any(1);unit=float(coef[mask].mean())
    eligible=bool(np.isfinite(A.numpy()).all()and np.isfinite(b.numpy()).all()and np.isfinite(native_delta)and native_delta>0 and np.isfinite(unit)and unit>0 and np.isfinite(coef[mask]).all()and(coef[mask]>0).all())
    J=np.zeros((169,169),np.float32);F=np.zeros((169,169),np.int32);lookup={tuple(o):(bool(v),float(c))for o,v,c in zip(d['offsets'],mask,coef)}
    for i in range(169):
        for j in range(i+1,169):
            dr,dc=j//13-i//13,j%13-i%13;line=next((k for k,(a,z)in enumerate(d['lines'])if a*dr+z*dc==0),None)
            inside,w=(True,float(A[line]))if line is not None else lookup.get((dr,dc),(False,0.))
            if inside:F[i,j]=F[j,i]=1;J[i,j]=J[j,i]=w
    # Independently reconstruct transferred F and weights from raw counts/exposure and offsets.
    for i,j in zip(*np.triu_indices(169,1)):
        dr,dc=j//13-i//13,j%13-i%13;on=np.flatnonzero(np.array([a*dr+z*dc==0 for a,z in d['lines']]));finite=np.flatnonzero((d['offsets']==[dr,dc]).all(1));inside=False;w=0.
        if len(on):inside=True;w=float(A[int(on[0])])
        elif len(finite):
            k=int(finite[0]);inside=bool(d['counts'][k]==0 and d['exposure'][k]>=math.log(264/.01));w=float(A[int(d['groups'][k])])if inside else 0.
        assert F[i,j]==inside and J[i,j]==w
    eligible=eligible and bool(np.isfinite(J).all()and(J[F.astype(bool)]>0).all())
    return dict(J=J,h=np.zeros(169,np.float32),F=F,unit=np.array(unit),q=np.array(13)),dict(source_eligible=eligible,native_source_delta=native_delta,unit=unit,source_retained_offsets=int(mask.sum()),minimum_retained_weight=float(coef[mask].min()),target_forbidden_pairs=int(F.sum()//2),transferred_edges_independently_verified=True)

def worker(gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    frozen();assert socket.gethostname()=='loki';lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    query=lambda *a:subprocess.check_output(a,text=True,timeout=20).strip();device=query('nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader');assert device.split(',')[1].strip()not in query('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    deadline=budget();base=dict(started_utc=now(),phase='fit',gpu=gpu,device=device,pid=os.getpid(),protocol_sha256=sha(SRC/'protocol.json'));receipt=OUT/f'fit-worker-g{gpu}.json'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;D=load(SRC.parent/'h3u/driver.py','h3ad_fit_fn')
    def progress(**v):save(receipt,dict(base,status='running',utc=now(),progress=v))
    try:
        for bank in range(gpu,2,2):
            folder=OUT/f'b{bank}';d=dict(np.load(folder/'source-design.npz'));assert sha(folder/'source-design.npz')==read(OUT/'source-banks.json')['banks'][bank]['source_design_sha256'];progress(bank=bank,stage='loss_gate');validation=validate_loss(d,'cuda');save(folder/'loss-validation.json',validation)
            torch.manual_seed(0);y=torch.tensor(d['labels'],device='cuda',dtype=torch.float32);weights=torch.where(y.bool(),231.,298.);weights/=weights.sum();g=torch.tensor(d['pair_groups'],device='cuda');A=torch.zeros(int(d['groups'].max())+1,device='cuda',requires_grad=True);b=torch.zeros(1,device='cuda',requires_grad=True)
            opt=torch.optim.AdamW([A,b],lr=.3,weight_decay=.01,betas=(.9,.999),eps=1e-8,foreach=False);torch.cuda.synchronize();start=time.perf_counter();curve=[];progress(bank=bank,stage='fit',update=0)
            for k in range(4000):
                if k%100==0 and time.perf_counter()-start>120:raise TimeoutError('Fixed120secondfit ceiling')
                opt.zero_grad(set_to_none=True);loss=D.source_loss(A,b,y,weights,g);loss.backward();opt.param_groups[0]['lr']=.3*.5*(1+math.cos(math.pi*k/4000));opt.step()
                if(k+1)%250==0:curve.append(dict(update=k+1,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start));progress(bank=bank,stage='fit',**curve[-1])
            torch.cuda.synchronize();seconds=time.perf_counter()-start;AA=A.detach().cpu();bb=b.detach().cpu();np.savez_compressed(folder/'parameters.npz',A=AA.numpy(),b=bb.numpy());model,info=construct(AA,bb,d);np.savez_compressed(folder/'model.npz',**model)
            save(folder/'model.json',dict(status='complete'if info['source_eligible']else'source_fit_ineligible',utc=now(),bank=bank,gpu=gpu,fit_seconds=seconds,updates=4000,curve=curve,**info,parameters_sha256=sha(folder/'parameters.npz'),model_sha256=sha(folder/'model.npz'),source_design_sha256=sha(folder/'source-design.npz'),protocol_sha256=sha(SRC/'protocol.json')));print('FIT',bank,seconds,info,flush=True)
        signal.setitimer(signal.ITIMER_REAL,0);save(receipt,dict(base,status='complete',completed_utc=now(),deadline_unix=deadline))
    except BaseException as e:save(receipt,dict(base,status='failed',completed_utc=now(),error=repr(e)));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker']);a.add_argument('--gpu',type=int,choices=[0,1]);args=a.parse_args();prepare()if args.mode=='prepare'else worker(args.gpu)
