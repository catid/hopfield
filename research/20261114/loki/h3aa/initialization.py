"""Dedicated CPU initialization stream, independent of proposal RNG consumption."""
import numpy as np
import torch

@torch.no_grad()
def make(givens,*,q,chains,seed,kind):
    given=torch.as_tensor(givens,dtype=torch.bool).repeat_interleave(chains,0);n=q;c=n*n;gen=torch.Generator(device='cpu').manual_seed(2026112600+seed)
    if kind=='arbitrary':
        scores=torch.rand(given.shape,generator=gen);scores[given]=-1;order=scores.argsort(dim=1,descending=True);rank=torch.empty_like(order)
        rank.scatter_(1,order,torch.arange(c).expand(len(given),c));x=given|(rank<(q-given.sum(1))[:,None])
    else:
        assert kind=='permutation';g=given.view(-1,n,n);fixed=g.any(2);used=g.any(1);given_col=g.to(torch.int64).argmax(2);row_rank=(~fixed).to(torch.int64).cumsum(1)-1
        scores=torch.rand((len(given),n),generator=gen);scores[used]=-1;perm=scores.argsort(1,descending=True);cols=torch.where(fixed,given_col,perm.gather(1,row_rank.clamp_min(0)))
        cells=torch.arange(n)[None]*n+cols;x=torch.zeros_like(given).scatter_(1,cells,True);assert(x.view(-1,n,n).sum(1)==1).all()and(x.view(-1,n,n).sum(2)==1).all()
    assert(x.sum(1)==q).all()and(x>=given).all()
    return x.view(len(givens),chains,c).numpy()
