"""Fixed-cardinality Metropolis swaps with exact integer support-conflict certificates.

Every proposal removes one occupied nongiven cell and fills one empty cell.
The two candidate sets are disjoint; their independent maxima in one IID random
score vector give independent uniform choices. Proposal probabilities are equal
in both directions because q and the number of clues remain fixed.
"""
import numpy as np
import torch

@torch.no_grad()
def solve(J,h,F,givens,*,q,unit,chains=20,cap=20000,seed=0,device='cpu',block=1<<16,end_ratio=.01,trace=False,initial=None):
    dev=torch.device(device);J=torch.as_tensor(J,dtype=torch.float32,device=dev);h=torch.as_tensor(h,dtype=torch.float32,device=dev)
    F=torch.as_tensor(F,dtype=torch.int32,device=dev);assert torch.equal(J,J.T) and not J.diag().any();assert torch.equal(F,F.T) and not F.diag().any()
    c=len(J);given=torch.as_tensor(givens,device=dev,dtype=torch.bool).repeat_interleave(chains,0);assert ((given.sum(1)<q)&(given.sum(1)>=0)).all();assert 0<q<c and unit>0
    gen=torch.Generator(device=dev).manual_seed(seed);total=len(given);out_x=torch.zeros_like(given);out_stop=torch.zeros(total,device=dev,dtype=torch.bool);out_energy=torch.zeros(total,device=dev);out_moves=torch.zeros(total,device=dev,dtype=torch.int64);out_conflict=torch.zeros(total,device=dev,dtype=torch.int32);out_accept=torch.zeros(total,device=dev,dtype=torch.int64);traces=[]
    for lo in range(0,total,block):
        ids=torch.arange(lo,min(total,lo+block),device=dev);giv=given[ids];free=~giv;size=len(ids)
        if initial is None:
            scores=torch.rand((size,c),device=dev,generator=gen);scores[giv]=-1
            order=scores.argsort(dim=1,descending=True);rank=torch.empty_like(order);rank.scatter_(1,order,torch.arange(c,device=dev).expand(size,c));x=giv|(rank<(q-giv.sum(1))[:,None])
        else:
            x=torch.as_tensor(initial,device=dev,dtype=torch.bool).reshape(total,c)[ids].clone()
        assert (x.sum(1)==q).all()and(x>=giv).all()
        field=x.float()@J+h;counts=(x.float()@F.float()).to(torch.int32);conflict=(x.to(torch.int32)*counts).sum(1,dtype=torch.int32)//2;energy=.5*(x.float()*(field+h)).sum(1)
        best_x=x.clone();best_e=energy.clone();best_conflict=conflict.clone();moves=torch.zeros(size,device=dev,dtype=torch.int64);accepted=torch.zeros_like(moves)
        done=conflict==0;out_x[ids[done]]=x[done];out_stop[ids[done]]=True;out_energy[ids[done]]=energy[done];out_conflict[ids[done]]=conflict[done]
        keep=~done;ids,x,giv,free,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted=(t[keep] for t in [ids,x,giv,free,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted])
        for step in range(cap):
            if not len(ids):break
            a=torch.arange(len(ids),device=dev);scores=torch.rand(x.shape,device=dev,generator=gen);occupied=x&free;empty=~x
            i=torch.where(occupied,scores,-torch.inf).argmax(1);j=torch.where(empty,scores,-torch.inf).argmax(1)
            change=field[a,j]-field[a,i]-J[i,j];count_change=counts[a,j]-counts[a,i]-F[i,j]
            T=unit*(end_ratio**(step/max(cap-1,1)));u=torch.rand(len(ids),device=dev,generator=gen);take=(change<=0)|(u<torch.exp(-change/T));accepted+=take
            old_x=x.clone() if trace else None
            x[a,i]=~take;x[a,j]=take;field+=take[:,None]*(J[j]-J[i]);counts+=take[:,None].to(torch.int32)*(F[j]-F[i]);conflict+=take.to(torch.int32)*count_change;energy+=take*change;moves+=1
            better=energy<best_e;best_x[better]=x[better];best_e[better]=energy[better];best_conflict[better]=conflict[better]
            if trace:traces.append(dict(ids=ids.cpu().numpy(),before=old_x.cpu().numpy(),after=x.cpu().numpy(),i=i.cpu().numpy(),j=j.cpu().numpy(),accepted=take.cpu().numpy(),delta=change.cpu().numpy(),conflicts=conflict.cpu().numpy(),energy=energy.cpu().numpy(),field=field.cpu().numpy(),counts=counts.cpu().numpy(),step=step,T=T))
            done=conflict==0
            if done.any() or step==cap-1:
                finish=done if step<cap-1 else torch.ones_like(done);out_x[ids[finish]]=torch.where(done[:,None],x,best_x)[finish];out_energy[ids[finish]]=torch.where(done,energy,best_e)[finish]
                out_conflict[ids[finish]]=torch.where(done,conflict,best_conflict)[finish];out_stop[ids[finish]]=done[finish];out_moves[ids[finish]]=moves[finish];out_accept[ids[finish]]=accepted[finish]
                keep=~finish;ids,x,giv,free,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted=(t[keep] for t in [ids,x,giv,free,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted])
    shape=(len(givens),chains);result=dict(boards=out_x.view(*shape,c).cpu().numpy(),stopped=out_stop.view(shape).cpu().numpy(),energy=out_energy.view(shape).cpu().numpy(),moves=out_moves.view(shape).cpu().numpy(),conflicts=out_conflict.view(shape).cpu().numpy(),accepted=out_accept.view(shape).cpu().numpy())
    if trace:result['trace']=traces
    return result
