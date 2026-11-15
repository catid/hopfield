"""Prospective matched-size numerical gate on four prior N600/c9 banks.

Banks 1..4 were reserved before any long-schedule fit on them. This supplements,
and does not erase, the failed N32 fixture and the post hoc bank0 diagnosis.
"""
import argparse
import json
import time
from noisy_pl_long import ROOT, HERE, OUT, PROTOCOL, sha, write, dependencies, transform


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',type=int,required=True);a=parser.parse_args()
    assert 1<=a.bank<=4
    p=json.loads(PROTOCOL.read_text());np,torch,CF=dependencies(a.bank-1)
    for path,digest in p['source_sha256'].items():assert sha(ROOT/path)==digest
    z=np.load(ROOT/p['gate_source']);noise=np.load(ROOT/p['matched_gate']['corruptions'])
    assert sha(ROOT/p['matched_gate']['corruptions'])==p['matched_gate']['corruptions_sha256']
    s=z['solutions'][a.bank,:600].astype(np.int64);free=~z['given'][a.bank,:600]
    np.put_along_axis(s,noise['cells'][a.bank,:600].astype(np.int64),noise['values'][a.bank,:600],1)
    draws=torch.rand((256,600,9),generator=torch.Generator().manual_seed(2026111514+a.bank))
    start=time.monotonic();results={};records=[]
    for label,device,dtype in [('gpu32','cuda',torch.float32),('gpu64','cuda',torch.float64),('cpu32','cpu',torch.float32)]:
        sol=torch.as_tensor(s,device=device);fr=torch.as_tensor(free,device=device)
        original_loss=CF.pl_loss
        if dtype==torch.float64:
            def loss(S,B,sol,free):
                y=torch.nn.functional.one_hot(sol,9).to(S.dtype)
                fields=torch.einsum('nmac,bmc->bna',S,y)+B[None]
                losses=-torch.log_softmax(-fields,-1).gather(2,sol[...,None]).squeeze(-1)
                return (losses*free).sum()/free.sum()
            CF.pl_loss=loss
        torch.set_default_dtype(dtype)
        try:W,B,seconds=CF.train(CF.FullConfig(**p['long_pl']),sol,fr,draws=draws)
        finally:CF.pl_loss=original_loss;torch.set_default_dtype(torch.float32)
        support,coefficients,meta=transform(W,torch)
        assert meta['eligible'];results[label]=(support.cpu(),coefficients.cpu(),meta)
        records.append(dict(label=label,seconds=seconds,minimum_threshold_distance=meta['minimum_threshold_distance'],retained=meta['retained']))
    comparisons=[]
    for other in ['gpu64','cpu32']:
        s0,j0,m0=results['gpu32'];s1,j1,m1=results[other]
        error=float((j0-j1).abs().max());allowed=min(m0['minimum_threshold_distance'],m1['minimum_threshold_distance'])/4
        passed=torch.equal(s0,s1) and error<allowed
        comparisons.append(dict(other=other,max_coefficient_error=error,allowed=allowed,support_equal=torch.equal(s0,s1),passed=passed))
    passed=all(r['passed'] for r in comparisons)
    write(OUT/f'matched-gate-b{a.bank}.json',dict(status='passed' if passed else 'failed',bank=a.bank,protocol_sha256=sha(PROTOCOL),
          producer_sha256=sha(__file__),records=records,comparisons=comparisons,wall_s=time.monotonic()-start))
    print(json.dumps(dict(bank=a.bank,passed=passed,comparisons=comparisons)),flush=True)
    assert passed


if __name__=='__main__':main()
