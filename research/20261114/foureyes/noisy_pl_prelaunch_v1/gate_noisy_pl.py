"""Source-only implementation gate; no fresh H4r model or test input is inspected."""
import json
import time
from noisy_pl import ROOT, HERE, OUT, PROTOCOL, sha, write, dependencies, fit_adam, transform


def main():
    start = time.monotonic(); p = json.loads(PROTOCOL.read_text())
    np, torch, CF = dependencies(0)
    z = np.load(ROOT / p['gate_source'])
    sol = torch.as_tensor(z['solutions'][0, :32].astype(np.int64))
    free = torch.as_tensor(~z['given'][0, :32])
    torch.manual_seed(2026111509)
    W = (torch.randn(3240, 9, 9) * .02).requires_grad_()
    B = (torch.randn(81, 9) * .02).requires_grad_()
    ii, jj = CF.pair_index('cpu'); pair = torch.arange(3240)[None, :]
    left = W[pair, :, sol[:, jj]]
    right = W[pair, sol[:, ii], :]
    G = B[None].expand(len(sol), -1, -1).clone().index_add(1, ii, left).index_add(1, jj, right)
    direct = (-torch.log_softmax(-G, -1).gather(2, sol[..., None]).squeeze(-1) * free).sum() / free.sum()
    native = CF.pl_loss(CF.full_S(W, (ii, jj)), B, sol, free)
    gd = torch.autograd.grad(direct, (W, B)); gn = torch.autograd.grad(native, (W, B))
    loss_error = abs(float(direct.detach()-native.detach()))
    gradient_error = max(float((a-b).abs().max()) for a,b in zip(gd,gn))
    assert loss_error < 1e-6 and gradient_error < 1e-7
    # Selected central differences on the independent pair-sum definition.
    def independent(w, b):
        g = b[None].expand(len(sol), -1, -1).clone()
        g = g.index_add(1, ii, w[pair, :, sol[:, jj]]).index_add(1, jj, w[pair, sol[:, ii], :])
        return (-torch.log_softmax(-g, -1).gather(2, sol[...,None]).squeeze(-1)*free).sum()/free.sum()
    wd, bd = W.detach().double().requires_grad_(), B.detach().double().requires_grad_()
    analytic = torch.autograd.grad(independent(wd, bd), (wd, bd)); finite_errors=[]
    for a, gradient in zip((wd,bd), analytic):
        for index in [0, 7, 31, a.numel()//2, a.numel()-1]:
            original=float(a.flatten()[index].detach()); h=1e-5
            with torch.no_grad():a.flatten()[index]=original+h
            plus=float(independent(wd,bd).detach())
            with torch.no_grad():a.flatten()[index]=original-h
            minus=float(independent(wd,bd).detach())
            with torch.no_grad():a.flatten()[index]=original
            finite_errors.append(abs((plus-minus)/(2*h)-float(gradient.flatten()[index])))
    assert max(finite_errors)<1e-9
    draws = torch.rand((32, 32, 9), generator=torch.Generator().manual_seed(2026111510))
    parity=[]
    for arm in ['adamw','paper_pl']:
        def fit(device):
            s,f=sol.to(device),free.to(device)
            return fit_adam(torch,CF,s,f,draws,p['adamw'])[:3] if arm=='adamw' else CF.train(CF.FullConfig(**p['paper_pl']),s,f,draws=draws)
        wc,bc,tc=fit('cpu');wg,bg,tg=fit('cuda')
        error=max(float((wc-wg.cpu()).abs().max()),float((bc-bg.cpu()).abs().max()))
        assert error<2e-5
        sc,_,mc=transform(wc,torch);sg,_,mg=transform(wg,torch)
        assert mc['eligible']==mg['eligible'] and (sc is None or torch.equal(sc,sg.cpu()))
        parity.append(dict(arm=arm,max_parameter_error=error,cpu_s=tc,gpu_s=tg,support_equal=True))
    # One original-size source-only fit measures whether the fixed fit envelope is feasible.
    s=torch.as_tensor(z['solutions'][0,:600].astype(np.int64),device='cuda')
    f=torch.as_tensor(~z['given'][0,:600],device='cuda')
    draws=torch.rand((32,600,9),generator=torch.Generator().manual_seed(2026111511))
    timing={}
    for arm in ['adamw','paper_pl']:
        fit=fit_adam(torch,CF,s,f,draws,p['adamw']) if arm=='adamw' else CF.train(CF.FullConfig(**p['paper_pl']),s,f,draws=draws)
        timing[arm]=fit[2]
    assert sum(timing.values())<p['max_fit_bank_s']/4
    write(OUT/'gate.json',dict(status='passed',protocol_sha256=sha(PROTOCOL),gate_source_sha256=sha(ROOT/p['gate_source']),
          script_sha256=sha(__file__),loss_error=loss_error,gradient_error=gradient_error,max_finite_difference_error=max(finite_errors),
          cpu_gpu_32row_fit_parity=parity,source_only_600row_fit_s=timing,wall_s=time.monotonic()-start))
    print((OUT/'gate.json').read_text(),flush=True)


if __name__=='__main__':main()
