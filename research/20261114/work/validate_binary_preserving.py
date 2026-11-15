"""Exhaustive small binary boards and continuous derivative gates; no trial."""
import hashlib
import json
import time
import numpy as np
import study as S
import binary_preserving as B


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):digest.update(name.encode());digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    begin=time.perf_counter();torch,_,_=S.imports();records=[];rng=np.random.default_rng(2026111501)
    shapes=[]
    for center_wall in [False,True]:
        tok=np.ones((3,3),np.int64);tok[0,0]=2;tok[2,2]=3
        if center_wall:tok[1,1]=0
        shapes.append(tok)
    for seed in range(3):
        model=S.model_for(seed,90,"original","cpu",torch.float64);before=state_hash(model)
        for graph,tokens in enumerate(shapes):
            t=torch.tensor(tokens)[None];free=torch.flatnonzero if hasattr(torch,"flatnonzero") else None
            ids=np.flatnonzero(tokens.ravel()==1);nfree=len(ids);n=2**nfree
            with torch.no_grad():psi,_=model.settle(t,tol=1e-10,max_steps=4000,check=8);a=model.couplings(t,psi)
            assert torch.isfinite(a).all();tt=t.expand(n,-1,-1);aa=a.expand(n,-1,-1,-1)
            boards=torch.zeros((n,9),dtype=torch.float64)
            boards[:,ids]=torch.tensor(((np.arange(n)[:,None]>>np.arange(nfree)[None])&1).astype(np.float64));boards=boards.reshape(n,3,3)
            base=model.energy(tt,boards,aa);delta=model.flip_delta(tt,boards,aa).reshape(n,-1)[:,ids]
            flipped=np.arange(n)[:,None]^(1<<np.arange(nfree)[None]);direct=base[torch.tensor(flipped)]-base[:,None]
            native_flip_error=float((direct-delta).abs().max());assert native_flip_error<1e-9
            sample=torch.tensor(rng.uniform(.1,.9,(4,3,3)),dtype=torch.float64);st=t.expand(4,-1,-1);sa=a.expand(4,-1,-1,-1);fr=model.free(st)
            self_values=(model.kappa[0]*a[:,0])[model.free(t)]
            for strength in [-2.,-1.,0.,1.,2.]:
                augmented=B.energy(model,tt,boards,aa,strength)
                assert torch.equal(augmented,base)
                new_direct=augmented[torch.tensor(flipped)]-augmented[:,None]
                assert torch.equal(new_direct,direct) and torch.equal(B.flip_delta(model,tt,boards,aa),model.flip_delta(tt,boards,aa))
                x=sample.clone().requires_grad_(True)
                def nonentropy(p):
                    nc=model.clamped(st,p);free=model.free(st).to(p.dtype)
                    entropy=(free*(nc*(nc+1e-12).log()+(1-nc)*(1-nc+1e-12).log())).flatten(1).sum(1)
                    return B.energy(model,st,p,sa,strength)-entropy
                gradient=torch.autograd.grad(nonentropy(x).sum(),x)[0]
                target=B.field(model,st,x,sa,strength);autograd_error=float((gradient+target)[fr].abs().max());assert autograd_error<1e-10
                native_field=model.field(st,x,sa);assert torch.equal(target[~fr],native_field[~fr])
                errors=[];epsilon=1e-6
                for idx in ids:
                    plus=sample.clone().reshape(4,-1);minus=sample.clone().reshape(4,-1);plus[:,idx]+=epsilon;minus[:,idx]-=epsilon
                    finite=(nonentropy(plus.reshape(4,3,3))-nonentropy(minus.reshape(4,3,3)))/(2*epsilon)
                    errors.append(float((finite+target.detach().reshape(4,-1)[:,idx]).abs().max()))
                fd_error=max(errors);assert fd_error<1e-7
                naive=base-strength*(model.free(tt).to(boards.dtype)*boards.square()).flatten(1).sum(1)
                naive_relative_error=float(((naive-naive[0])-(base-base[0])).abs().max())
                if strength!=0:assert naive_relative_error>0
                records.append({"seed":seed,"graph":graph,"free_cells":nfree,"binary_states":n,"directed_binary_flips":n*nfree,"strength":strength,"binary_energy_bitwise_unchanged":True,"true_flip_cost_bitwise_unchanged":True,"native_certificate_bitwise_unchanged":True,"native_flip_cost_max_error":native_flip_error,"continuous_autograd_max_error":autograd_error,"continuous_central_difference_max_error":fd_error,"finite_difference_epsilon":epsilon,"clamped_fields_unchanged":True,"naive_uncompensated_diagonal_max_relative_energy_change":naive_relative_error,"self_coefficient_min":float(self_values.min()),"self_coefficient_max":float(self_values.max())})
        assert state_hash(model)==before
    result={"status":"passed","utc":S.utc(),"scope":"Algebra and implementation validation only; no scientific intervention, test-cohort inference or model mutation.","derivation":"E_lambda(p)=E(p)+lambda sum_free p_i(1-p_i). On binary boards R=0, hence all binary energies and true flip costs unchanged. Non-entropy negative gradient changes by lambda(2p_i-1) on free cells; original flip_delta certificate retained. Positive lambda adds concavity -2lambda in the continuous Hessian.","parameterization_caveat":"Adding self coupling alone changes binary energies. Uniform diagonal compensation requires +lambda*p unary and -lambda*p^2 self term. Changing scalar kappa[0] produces spatial a_self-weighted changes that global b cannot generally compensate. Adding a center MU pair term also invalidates the native off-diagonal-only flip formula unless its self correction is supplied. The explicit free-cell wrapper avoids these errors.","rows":records,"binary_energy_checks":sum(r["binary_states"] for r in records),"directed_flip_checks":sum(r["directed_binary_flips"] for r in records),"model_states_unchanged":True,"helper_sha256":S.sha(B.__file__),"script_sha256":S.sha(__file__),"source_sha256":{p:S.sha(S.ROOT/p) for p in ["upstream/cram/cram_maze_hard_graph.py","upstream/cram/maze_h3.py"]},"seconds":time.perf_counter()-begin}
    dest=S.OUT/"binary_preserving";S.save(dest/"validation.json",result);S.emit("binary_preserving_gate_passed",energy_checks=result["binary_energy_checks"],flip_checks=result["directed_flip_checks"],max_fd_error=max(r["continuous_central_difference_max_error"] for r in records),seconds=result["seconds"])


if __name__=="__main__":main()
