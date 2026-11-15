"""H4s source-only long-schedule parity and throughput, before fresh fitting."""
import json
import time
from noisy_pl_long import ROOT, OUT, PROTOCOL, sha, write, dependencies, transform


def main():
    p=json.loads(PROTOCOL.read_text());np,torch,CF=dependencies(0)
    for name,digest in p['frozen_inputs'].items():assert sha(ROOT/name)==digest
    prior=json.loads((ROOT/'results/research/foureyes/noisy_pl/gate.json').read_text());assert prior['status']=='passed'
    z=np.load(ROOT/p['gate_source']);sol=torch.as_tensor(z['solutions'][0,:32].astype(np.int64));free=torch.as_tensor(~z['given'][0,:32])
    draws=torch.rand((256,32,9),generator=torch.Generator().manual_seed(2026111512));start=time.monotonic()
    wc,bc,tc=CF.train(CF.FullConfig(**p['long_pl']),sol,free,draws=draws)
    wg,bg,tg=CF.train(CF.FullConfig(**p['long_pl']),sol.cuda(),free.cuda(),draws=draws)
    sc,jc,mc=transform(wc,torch);sg,jg,mg=transform(wg,torch)
    error=float((jc-jg.cpu()).abs().max())
    assert mc['eligible'] and mg['eligible'] and torch.equal(sc,sg.cpu())
    assert error<min(mc['minimum_threshold_distance'],mg['minimum_threshold_distance'])/4
    sol=torch.as_tensor(z['solutions'][0,:600].astype(np.int64),device='cuda');free=torch.as_tensor(~z['given'][0,:600],device='cuda')
    draws=torch.rand((256,600,9),generator=torch.Generator().manual_seed(2026111513))
    _,_,throughput=CF.train(CF.FullConfig(**p['long_pl']),sol,free,draws=draws)
    assert throughput<p['max_fit_bank_s']/4
    write(OUT/'gate.json',dict(status='passed',protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),
          inherited_loss_gradient_gate_sha256=sha(ROOT/'results/research/foureyes/noisy_pl/gate.json'),
          cpu_gpu_support_equal=True,max_canonical_coefficient_error=error,
          min_threshold_distance=min(mc['minimum_threshold_distance'],mg['minimum_threshold_distance']),
          cpu32row_s=tc,gpu32row_s=tg,gpu600row_s=throughput,wall_s=time.monotonic()-start))
    print((OUT/'gate.json').read_text(),flush=True)


if __name__=='__main__':main()
