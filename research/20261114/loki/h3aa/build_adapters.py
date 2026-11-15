"""Copy frozen kernels and add explicit initial states without changing transitions."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa'
def main():
    assert not(SRC/'protocol.json').exists()
    for old,name in [('h3v','arbitrary'),('h3w','permutation')]:
        text=(SRC.parent/old/'backend.py').read_text();text=text.replace("end_ratio=.01,trace=False):","end_ratio=.01,trace=False,initial=None):")
        if name=='arbitrary':
            before="""        ids=torch.arange(lo,min(total,lo+block),device=dev);giv=given[ids];free=~giv;size=len(ids);scores=torch.rand((size,c),device=dev,generator=gen);scores[giv]=-1
        # rank descending; every row gets exactly q-number_of_clues additions.
        order=scores.argsort(dim=1,descending=True);rank=torch.empty_like(order);rank.scatter_(1,order,torch.arange(c,device=dev).expand(size,c));x=giv|(rank<(q-giv.sum(1))[:,None]);assert (x.sum(1)==q).all()
"""
            after="""        ids=torch.arange(lo,min(total,lo+block),device=dev);giv=given[ids];free=~giv;size=len(ids)
        if initial is None:
            scores=torch.rand((size,c),device=dev,generator=gen);scores[giv]=-1
            order=scores.argsort(dim=1,descending=True);rank=torch.empty_like(order);rank.scatter_(1,order,torch.arange(c,device=dev).expand(size,c));x=giv|(rank<(q-giv.sum(1))[:,None])
        else:
            x=torch.as_tensor(initial,device=dev,dtype=torch.bool).reshape(total,c)[ids].clone()
        assert (x.sum(1)==q).all()and(x>=giv).all()
"""
        else:
            before="""        score=torch.rand((len(ids),n),generator=gen,device=dev);score[used]=-1;permutation=score.argsort(1,descending=True);cols=torch.where(fixed,given_col,permutation.gather(1,row_rank.clamp_min(0)))
        cells=torch.arange(n,device=dev)[None]*n+cols;x=torch.zeros((len(ids),c),device=dev,dtype=torch.bool).scatter_(1,cells,True)
"""
            after="""        if initial is None:
            score=torch.rand((len(ids),n),generator=gen,device=dev);score[used]=-1;permutation=score.argsort(1,descending=True);cols=torch.where(fixed,given_col,permutation.gather(1,row_rank.clamp_min(0)))
            cells=torch.arange(n,device=dev)[None]*n+cols;x=torch.zeros((len(ids),c),device=dev,dtype=torch.bool).scatter_(1,cells,True)
        else:
            x=torch.as_tensor(initial,device=dev,dtype=torch.bool).reshape(total,c)[ids].clone();cols=x.view(-1,n,n).to(torch.int64).argmax(2)
"""
        assert text.count(before)==1;text=text.replace(before,after)
        (SRC/f'{name}_backend.py').write_text(text)
if __name__=='__main__':main()
