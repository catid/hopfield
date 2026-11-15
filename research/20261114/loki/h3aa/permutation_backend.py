"""Permutation Metropolis swaps preserving inferred row/column constraints and givens."""
import numpy as np
import torch

@torch.no_grad()
def solve(J,h,F,givens,*,q,unit,chains=20,cap=20000,seed=0,device='cpu',block=1<<16,end_ratio=.01,trace=False,initial=None):
    dev=torch.device(device);J=torch.as_tensor(J,dtype=torch.float32,device=dev);h=torch.as_tensor(h,dtype=torch.float32,device=dev);F=torch.as_tensor(F,dtype=torch.int32,device=dev)
    n=q;c=n*n;assert J.shape==(c,c)and torch.equal(J,J.T)and not J.diag().any();assert torch.equal(F,F.T)and not F.diag().any()and unit>0
    given=torch.as_tensor(givens,device=dev,dtype=torch.bool).repeat_interleave(chains,0);G=given.view(-1,n,n);assert(G.sum(1)<=1).all()and(G.sum(2)<=1).all()and(given.sum(1)<=q-2).all()
    total=len(given);gen=torch.Generator(device=dev).manual_seed(seed);out_x=torch.zeros_like(given);out_stop=torch.zeros(total,device=dev,dtype=torch.bool);out_energy=torch.zeros(total,device=dev);out_moves=torch.zeros(total,device=dev,dtype=torch.int64);out_conflict=torch.zeros(total,device=dev,dtype=torch.int32);out_accept=torch.zeros_like(out_moves);traces=[]
    for lo in range(0,total,block):
        ids=torch.arange(lo,min(total,lo+block),device=dev);giv=given[ids];g= giv.view(-1,n,n);fixed=g.any(2);used=g.any(1);given_col=g.to(torch.int64).argmax(2);row_rank=(~fixed).to(torch.int64).cumsum(1)-1
        if initial is None:
            score=torch.rand((len(ids),n),generator=gen,device=dev);score[used]=-1;permutation=score.argsort(1,descending=True);cols=torch.where(fixed,given_col,permutation.gather(1,row_rank.clamp_min(0)))
            cells=torch.arange(n,device=dev)[None]*n+cols;x=torch.zeros((len(ids),c),device=dev,dtype=torch.bool).scatter_(1,cells,True)
        else:
            x=torch.as_tensor(initial,device=dev,dtype=torch.bool).reshape(total,c)[ids].clone();cols=x.view(-1,n,n).to(torch.int64).argmax(2)
        assert(x.view(-1,n,n).sum(1)==1).all()and(x.view(-1,n,n).sum(2)==1).all()and(x>=giv).all()
        field=x.float()@J+h;counts=(x.float()@F.float()).to(torch.int32);conflict=(x.to(torch.int32)*counts).sum(1,dtype=torch.int32)//2;energy=.5*(x.float()*(field+h)).sum(1)
        best_x=x.clone();best_e=energy.clone();best_conflict=conflict.clone();moves=torch.zeros(len(ids),device=dev,dtype=torch.int64);accepted=torch.zeros_like(moves)
        done=conflict==0;out_x[ids[done]]=x[done];out_stop[ids[done]]=True;out_energy[ids[done]]=energy[done];out_conflict[ids[done]]=conflict[done]
        keep=~done;ids,x,giv,fixed,cols,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted=(t[keep]for t in [ids,x,giv,fixed,cols,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted])
        for step in range(cap):
            if not len(ids):break
            a=torch.arange(len(ids),device=dev);score=torch.rand(cols.shape,generator=gen,device=dev);score[fixed]=-torch.inf;rows=score.topk(2,dim=1).indices;r,s=rows[:,0],rows[:,1];cr,cs=cols[a,r].clone(),cols[a,s].clone()
            i=r*n+cr;j=s*n+cs;k=r*n+cs;l=s*n+cr
            change=field[a,k]+field[a,l]-field[a,i]-field[a,j]+J[i,j]+J[k,l]-J[i,k]-J[i,l]-J[j,k]-J[j,l]
            change_count=counts[a,k]+counts[a,l]-counts[a,i]-counts[a,j]+F[i,j]+F[k,l]-F[i,k]-F[i,l]-F[j,k]-F[j,l]
            T=unit*(end_ratio**(step/max(cap-1,1)));u=torch.rand(len(ids),device=dev,generator=gen);take=(change<=0)|(u<torch.exp(-change/T));accepted+=take;old_x=x.clone()if trace else None
            cols[a,r]=torch.where(take,cs,cr);cols[a,s]=torch.where(take,cr,cs);x[a,i]=~take;x[a,j]=~take;x[a,k]=take;x[a,l]=take
            field+=take[:,None]*(J[k]+J[l]-J[i]-J[j]);counts+=take[:,None].to(torch.int32)*(F[k]+F[l]-F[i]-F[j]);conflict+=take.to(torch.int32)*change_count;energy+=take*change;moves+=1
            better=energy<best_e;best_x[better]=x[better];best_e[better]=energy[better];best_conflict[better]=conflict[better]
            if trace:traces.append(dict(ids=ids.cpu().numpy(),before=old_x.cpu().numpy(),after=x.cpu().numpy(),rows=rows.cpu().numpy(),cells=torch.stack([i,j,k,l],1).cpu().numpy(),accepted=take.cpu().numpy(),delta=change.cpu().numpy(),conflicts=conflict.cpu().numpy(),energy=energy.cpu().numpy(),field=field.cpu().numpy(),counts=counts.cpu().numpy(),step=step,T=T))
            done=conflict==0
            if done.any()or step==cap-1:
                finish=done if step<cap-1 else torch.ones_like(done);out_x[ids[finish]]=torch.where(done[:,None],x,best_x)[finish];out_energy[ids[finish]]=torch.where(done,energy,best_e)[finish];out_conflict[ids[finish]]=torch.where(done,conflict,best_conflict)[finish];out_stop[ids[finish]]=done[finish];out_moves[ids[finish]]=moves[finish];out_accept[ids[finish]]=accepted[finish]
                keep=~finish;ids,x,giv,fixed,cols,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted=(t[keep]for t in [ids,x,giv,fixed,cols,field,counts,conflict,energy,best_x,best_e,best_conflict,moves,accepted])
    shape=(len(givens),chains);out=dict(boards=out_x.view(*shape,c).cpu().numpy(),stopped=out_stop.view(shape).cpu().numpy(),energy=out_energy.view(shape).cpu().numpy(),moves=out_moves.view(shape).cpu().numpy(),conflicts=out_conflict.view(shape).cpu().numpy(),accepted=out_accept.view(shape).cpu().numpy())
    if trace:out['trace']=traces
    return out
