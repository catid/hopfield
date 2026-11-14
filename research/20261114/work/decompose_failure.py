"""Post-outcome energy decomposition using retained states, without new inference."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import study as S


def main():
    dest=S.OUT/"eta_boundary/failure-forensics";rows=[]
    for seed in range(3):
        model=S.model_for(seed,90,"original","cpu",torch.float64);base=np.load(dest/f"s{seed}-eta0.5-states.npz")
        tok=torch.tensor(base["tokens"].astype(np.int64))[None];a=torch.tensor(base["couplings"],dtype=torch.float64)[None]
        psi=torch.tensor(base["hidden_psi"],dtype=torch.float64)[None];lg,la,lc=model.parts(tok,model.mu)
        retained={"log_edge_conductance":lg[0].numpy(),"log_latent_diagonal":la[0].numpy(),"log_latent_source":lc[0].numpy()}
        states={"oracle_path":base["oracle_label"]}
        for eta in (.2,.5,.8,1.):states[f"eta{eta}"]=np.load(dest/f"s{seed}-eta{eta}-states.npz")["prediction"]
        for name,pred in states.items():
            nu=torch.tensor(pred,dtype=torch.float64)[None];nc=model.clamped(tok,nu);free=model.free(tok).to(torch.float64)
            answer_states=torch.stack([1-nc,nc],1);out,_,self_gate=model.support(a,nc)
            terms={"entropy":free*(nc*(nc+1e-12).log()+(1-nc)*(1-nc+1e-12).log()),
                   "latent_gated_links":-nc*out-self_gate*nc*nc,
                   "maze_unique_pairs":.5*(F.conv2d(answer_states,model.mu_kernel(),padding=1)*answer_states).sum(1),
                   "maze_unique_bias":-free*(answer_states*model.b[None,:,None,None]).sum(1)}
            total=sum(float(v.sum()) for v in terms.values());reference=float(model.energy(tok,nu,a)[0]);assert abs(total-reference)<1e-8
            row={"seed":seed,"state":name,"energy_float64_from_retained_float32_coefficients":total,**{k:float(v.sum()) for k,v in terms.items()}}
            rows.append(row)
            for key,value in terms.items():retained[f"{name}_{key}"]=value[0].numpy()
        np.savez_compressed(dest/f"s{seed}-energy-source-terms.npz",**retained)
    result={"status":"decomposed","utc":S.utc(),"post_outcome_diagnostic":True,"new_inference":False,"rows":rows,"script_sha256":S.sha(__file__),
            "precision":"Float64 evaluation of the saved float32 couplings, predictions and model weights; components sum to the repository energy within1e-8. Small differences from recorded float32 total energies are rounding.",
            "terms":"Answer energy separated into entropy, latent-gated links, Maze-Unique pair tables and free-cell bias. Full per-cell terms and latent log edge/diagonal/source arrays retained separately.",
            "full_terms_sha256":{f"s{s}-energy-source-terms.npz":S.sha(dest/f"s{s}-energy-source-terms.npz") for s in range(3)}}
    S.save(dest/"energy-decomposition.json",result);S.emit("failure_energy_decomposed",rows=rows)


if __name__=="__main__":main()
