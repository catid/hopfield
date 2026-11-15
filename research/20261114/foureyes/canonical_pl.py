"""Convex digit-invariant Sudoku PL with unrestricted real pair coefficients.

This is a new optimizer/parametrization, not the paper's finite SGD trajectory.
Raw unordered-pair entrywise L1 coefficient lambda becomes9*lambda here.
"""
import math
import time


def context(torch, solutions, free, dtype):
    onehot = torch.nn.functional.one_hot(solutions,9).to(dtype)
    pair = torch.triu_indices(81,81,1,device=solutions.device)
    mask = free.to(dtype)
    return dict(torch=torch,y=onehot,sol=solutions,mask=mask,F=mask.sum(),ii=pair[0],jj=pair[1])


def loss_gradient(j, ctx, gradient=True):
    torch=ctx['torch'];ii,jj=ctx['ii'],ctx['jj']
    adjacency=torch.zeros((81,81),dtype=j.dtype,device=j.device)
    adjacency[ii,jj]=j;adjacency[jj,ii]=j
    fields=torch.einsum('ij,bjv->biv',adjacency,ctx['y'])
    logprob=torch.log_softmax(-fields,dim=-1)
    loss=-(logprob.gather(2,ctx['sol'][...,None]).squeeze(-1)*ctx['mask']).sum()/ctx['F']
    if not gradient:return loss
    residual=(ctx['y']-logprob.exp())*ctx['mask'][...,None]/ctx['F']
    directed=torch.einsum('biv,bjv->ij',residual,ctx['y'])
    return loss,directed[ii,jj]+directed[jj,ii]


def fit(torch, solutions, free, raw_lambda=1/30000, updates=256,
        initial_step=540., dtype=None):
    dtype=dtype or torch.float64
    ctx=context(torch,solutions,free,dtype)
    j=torch.zeros(3240,dtype=dtype,device=solutions.device)
    penalty=9*raw_lambda;step=initial_step;history=[];start=time.perf_counter()
    for update in range(updates):
        f,g=loss_gradient(j,ctx);old=f+penalty*j.abs().sum()
        accepted=False
        for backtrack in range(32):
            proposal=j-step*g
            candidate=proposal.sign()*(proposal.abs()-step*penalty).clamp_min(0)
            delta=candidate-j
            new_f=loss_gradient(candidate,ctx,gradient=False)
            majorizer=f+(g*delta).sum()+delta.square().sum()/(2*step)
            allowance=1e-12*(1+float(f.abs()))
            if float(new_f-majorizer)<=allowance:
                accepted=True;break
            step*=.5
        assert accepted, 'Backtracking exhausted'
        objective=new_f+penalty*candidate.abs().sum()
        assert float(objective-old)<=allowance
        j=candidate
        history.append(dict(update=update+1,loss=float(new_f),objective=float(objective),step=step,
            backtracks=backtrack,step_linf=float(delta.abs().max()),majorizer_gap=float(new_f-majorizer)))
    f,g=loss_gradient(j,ctx)
    residual=torch.where(j>0,g+penalty,torch.where(j<0,g-penalty,(g.abs()-penalty).clamp_min(0)))
    if j.is_cuda:torch.cuda.synchronize()
    return j,dict(history=history,seconds=time.perf_counter()-start,kkt_linf=float(residual.abs().max()),
        raw_lambda=raw_lambda,canonical_penalty=penalty,updates=updates,dtype=str(dtype),
        scope='Finite-budget convexproximalgradient; no exactoptimum orconvergenceclaim.')


def select(torch, j):
    coefficients=j.clamp_min(0)
    values=coefficients[coefficients>0].sort().values
    if len(values)<2:return None,dict(eligible=False,reason='fewer_than_two_positive_coefficients')
    gaps=values.diff();index=int(gaps.argmax())
    if not float(gaps[index])>0:return None,dict(eligible=False,reason='no_positive_gap')
    threshold=float((values[index]+values[index+1])/2)
    return coefficients>threshold,dict(eligible=True,threshold=threshold,retained=int((coefficients>threshold).sum()),
        minimum_threshold_distance=float((coefficients-threshold).abs().min()))
