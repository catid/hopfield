"""Independent finite-difference, objective, cache and throughput gates; no fits."""
import argparse
import copy
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import contrastive_objective as C
from prepare_contrastive import DEST


def independent_energy(torch, model, t, y, a):
    """Direct half-pair/link sum, independently of convolution/support helpers."""
    p=torch.where(t==1,y,(t>=2).to(y.dtype));h,w=t.shape[-2:]
    result=-(torch.stack([1-p,p],-1)*model.b).sum(-1).masked_fill(t!=1,0).sum((1,2))
    def sl(dr,dc):
        return (slice(max(0,-dr),min(h,h-dr)),slice(max(0,-dc),min(w,w-dc))), (slice(max(0,dr),min(h,h+dr)),slice(max(0,dc),min(w,w+dc)))
    for k,(dr,dc) in enumerate(model.mu_offsets):
        z,v=sl(dr,dc);left=p[:,z[0],z[1]];right=p[:,v[0],v[1]];table=model.T[k]
        result+=((1-left)*(1-right)*table[0,0]+(1-left)*right*table[0,1]+left*(1-right)*table[1,0]+left*right*table[1,1]).sum((1,2))
    for k,(dr,dc) in enumerate(model.aoffs):
        z,v=sl(dr,dc);result-=(model.kappa[k]*a[:,k,z[0],z[1]]*p[:,z[0],z[1]]*p[:,v[0],v[1]]).sum((1,2))
    # Binary entropy is independent of the selected board and cancels.
    return result


def numerical():
    torch,_,_=S.imports();records=[]
    with np.load(DEST/'training-arrays.npz') as x:tokens=x['tokens'][:4];labels=x['labels'][:4]
    for seed in range(3):
        model=S.model_for(seed,30,'original','cpu',torch.float64);params=C.make_trainable(model)
        with np.load(DEST/f'train-readout-seed{seed}.npz') as x:a=torch.tensor(x['couplings'][:4],dtype=torch.float64)
        t=torch.tensor(tokens);y=torch.tensor(labels,dtype=torch.float64);zero=torch.zeros_like(y)
        direct=independent_energy(torch,model,t,y,a)-independent_energy(torch,model,t,zero,a)
        native=model.energy(t,y,a)-model.energy(t,zero,a);energy_error=float((direct-native).abs().max());assert energy_error<1e-8
        # Independent PL: brute energy flips at every free cell in one small fixture.
        fixture=torch.tensor([[[2,1,0,1],[1,1,1,1],[0,1,1,3],[1,0,1,1]]]);rng=np.random.default_rng(843+seed)
        fa=torch.tensor(rng.random((1,5,4,4)),dtype=torch.float64);fy=torch.tensor(rng.integers(0,2,(1,4,4)),dtype=torch.float64)
        base=independent_energy(torch,model,fixture,fy,fa);terms=[]
        for r,c in np.argwhere(fixture.numpy()[0]==1):
            flip=fy.clone();flip[0,r,c]=1-flip[0,r,c];terms.append(torch.nn.functional.softplus(-(independent_energy(torch,model,fixture,flip,fa)-base)))
        direct_pl=torch.stack(terms).mean();native_pl=model.pl_loss(fixture,fy,fa)
        dg=torch.autograd.grad(direct_pl,params,retain_graph=True);ng=torch.autograd.grad(native_pl,params)
        pl_value_error=float((direct_pl-native_pl).abs());pl_gradient_error=max(float((x-z).abs().max()) for x,z in zip(dg,ng));assert pl_value_error<1e-10 and pl_gradient_error<1e-10
        weights=[]
        for weight in [0.,C.WEIGHT]:
            objective=C.losses(model,t,y,a,weight)[0];grads=torch.autograd.grad(objective,params);errors=[]
            for p,g in zip(params,grads):
                for i in range(p.numel()):
                    original=float(p.detach().reshape(-1)[i]);eps=1e-4
                    with torch.no_grad():
                        p.reshape(-1)[i]=original+eps;hi=float(C.losses(model,t,y,a,weight)[0]);p.reshape(-1)[i]=original-eps;lo=float(C.losses(model,t,y,a,weight)[0]);p.reshape(-1)[i]=original
                    errors.append(abs((hi-lo)/(2*eps)-float(g.reshape(-1)[i])))
            assert max(errors)<1e-7;weights.append({'weight':weight,'parameters_checked':sum(p.numel() for p in params),'maximum_finite_difference_error':max(errors)})
        # Positive control: a small contrast-only step lowers the contrast and gap.
        _,_,before,bgap=C.losses(model,t,y,a,C.WEIGHT);grads=torch.autograd.grad(before,params)
        with torch.no_grad():
            for p,g in zip(params,grads):p.add_(g,alpha=-1e-4)
        _,_,after,agap=C.losses(model,t,y,a,C.WEIGHT)
        assert float(after)<float(before) and float(agap.mean())<float(bgap.mean())
        records.append({'seed':seed,'direct_energy_delta_error':energy_error,'independent_pl_value_error':pl_value_error,'independent_pl_gradient_error':pl_gradient_error,'finite_difference':weights,'positive_control':{'contrast_before':float(before.detach()),'contrast_after':float(after.detach()),'gap_before':float(bgap.mean().detach()),'gap_after':float(agap.mean().detach())}})
    S.save(DEST/'numerical-gate.json',{'status':'passed','utc':S.utc(),'records':records,'script_sha256':S.sha(__file__),'objective_sha256':S.sha(C.__file__),'scope':'Disposable models only, no scientific fits or target-cohort inputs.'});S.emit('contrastive_numerical_gate',status='passed',records=records)


def performance(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();records=[]
    with np.load(DEST/'training-arrays.npz') as x:t=torch.tensor(x['tokens'][:64],device='cuda');y=torch.tensor(x['labels'][:64],device='cuda',dtype=torch.float32)
    for seed in range(3):
        if seed%2!=gpu:continue
        model=S.model_for(seed,30,'original','cuda');C.make_trainable(model)
        with np.load(DEST/f'train-readout-seed{seed}.npz') as x:a=torch.tensor(x['couplings'][:64],device='cuda')
        with torch.no_grad():psi,steps=model.settle(t,tol=1e-6,max_steps=4000,check=8);recomputed=model.couplings(t,psi)
        assert torch.equal(a,recomputed)
        timings=[]
        for weight in [0.,C.WEIGHT]:
            for _ in range(3):model.zero_grad(set_to_none=True);C.losses(model,t,y,a,weight)[0].backward()
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(32):model.zero_grad(set_to_none=True);C.losses(model,t,y,a,weight)[0].backward()
            torch.cuda.synchronize();timings.append({'weight':weight,'backward_passes':32,'seconds':time.perf_counter()-start})
        records.append({'seed':seed,'cache_exact':True,'maximum_hidden_steps':int(steps.max()),'timings':timings})
    S.save(DEST/f'performance-gate-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'inventory':inventory,'records':records,'script_sha256':S.sha(__file__),'objective_sha256':S.sha(C.__file__),'scope':'Repeated forward/backward passes only. No optimizer steps and no fit selection. Exact64input cached-readout recompute.'});S.emit('contrastive_performance_gate',gpu=gpu,records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['numerical','performance']);p.add_argument('--gpu',type=int,choices=[0,1]);args=p.parse_args()
    if args.command=='numerical':numerical()
    else:performance(args.gpu)
