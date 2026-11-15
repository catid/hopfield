"""Float64 original-binary-energy coordinate descent with four parity colors."""
import study as S

OFFSETS=tuple((dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if dr or dc)
COLORS=((0,0),(0,1),(1,0),(1,1))
THRESHOLD=-1e-8
SWEEP_CAP=512


def slices(h,w,dr,dc):
    return (slice(max(0,-dr),min(h,h-dr)),slice(max(0,-dc),min(w,w-dc))), (slice(max(0,dr),min(h,h+dr)),slice(max(0,dc),min(w,w+dc)))


class BinaryQuadratic:
    """Cache exact algebraic binary coefficients; all arithmetic is float64."""
    def __init__(self,model,tokens,couplings):
        torch,_,_=S.imports();assert model.T.dtype==couplings.dtype==torch.float64
        self.t=tokens;self.free=tokens==1;self.h=torch.zeros_like(tokens,dtype=torch.float64)
        self.h+=self.free*(model.b[0]-model.b[1]);self.weights={d:torch.zeros_like(self.h) for d in OFFSETS}
        height,width=tokens.shape[-2:]
        for k,(dr,dc) in enumerate(model.mu_offsets):
            l,r=slices(height,width,dr,dc);table=model.T[k]
            self.h[:,l[0],l[1]]+=table[1,0]-table[0,0]
            self.h[:,r[0],r[1]]+=table[0,1]-table[0,0]
            interaction=table[1,1]-table[1,0]-table[0,1]+table[0,0]
            self.weights[(dr,dc)][:,l[0],l[1]]+=interaction
            self.weights[(-dr,-dc)][:,r[0],r[1]]+=interaction
        for k,(dr,dc) in enumerate(model.aoffs):
            if dr==dc==0:
                self.h-=model.kappa[k]*couplings[:,k]
                continue
            l,r=slices(height,width,dr,dc);edge=model.kappa[k]*couplings[:,k,l[0],l[1]]
            self.weights[(dr,dc)][:,l[0],l[1]]-=edge
            self.weights[(-dr,-dc)][:,r[0],r[1]]-=edge
        rr=torch.arange(height,device=tokens.device)[:,None];cc=torch.arange(width,device=tokens.device)[None,:]
        self.colors=[((rr%2==r)&(cc%2==c))[None]&self.free for r,c in COLORS]

    def delta(self,p):
        current=self.h.clone();height,width=p.shape[-2:]
        for (dr,dc),weight in self.weights.items():
            l,r=slices(height,width,dr,dc);current[:,l[0],l[1]]+=weight[:,l[0],l[1]]*p[:,r[0],r[1]]
        return (1-2*p)*current

    def initial(self):return (self.t>0).to(self.h.dtype)


def solve(cache,cap=SWEEP_CAP,initial=None,trace=False):
    torch,_,_=S.imports();p=cache.initial() if initial is None else initial.clone()
    assert p.dtype==torch.float64
    assert torch.equal(p[~cache.free],cache.initial()[~cache.free])
    n=len(p);done=torch.zeros(n,dtype=torch.bool,device=p.device);used=torch.full((n,),cap,dtype=torch.int64,device=p.device);flips=torch.zeros_like(used);history=[]
    with torch.no_grad():
        for sweep in range(1,cap+1):
            changed=torch.zeros_like(done)
            for color,mask in enumerate(cache.colors):
                margins=cache.delta(p);change=(margins<THRESHOLD)&mask&~done[:,None,None]
                counts=change.sum((1,2));changed|=counts>0;flips+=counts
                if trace:history.append({'sweep':sweep,'color':color,'before':p.clone(),'flip':change.clone(),'margins':margins.clone()})
                p=torch.where(change,1-p,p)
            newly_done=~changed&~done;used=torch.where(newly_done,torch.full_like(used,sweep),used);done|=newly_done
            if done.all():break
        margin=cache.delta(p).masked_fill(~cache.free,float('inf')).flatten(1).amin(1)
        finite=torch.isfinite(p).flatten(1).all(1)&torch.isfinite(margin)
        assert torch.all(margin[done]>=THRESHOLD)
    return {'prediction':p,'sweeps':used,'stopped':done,'finite':finite,'flip_count':flips,'minimum_flip_margin':margin,'history':history}
