"""Post-outcome diagnostics for all native or screened failures; no rescue."""
import fcntl
import json
import os
import subprocess
import numpy as np
import study as S


def classify(tok,pred):
    nodes=set(map(tuple,np.argwhere(pred)));degrees=[sum((r+dr,c+dc) in nodes for dr,dc in S.SIDES) for r,c in nodes]
    remaining=nodes.copy();components=0
    while remaining:
        stack=[remaining.pop()];components+=1
        while stack:
            r,c=stack.pop()
            for dr,dc in S.SIDES:
                q=(r+dr,c+dc)
                if q in remaining:remaining.remove(q);stack.append(q)
    return {"selected_nodes":len(nodes),"components":components,"branch_nodes":sum(x>2 for x in degrees),"isolated_nodes":sum(x==0 for x in degrees),"endpoint_only":len(nodes)==2 and bool(np.all(pred[tok>=2]))}


def main():
    gpu=0;lock=(S.ROOT/"artifacts/gpu0.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=uuid","--format=csv,noheader"],text=True).strip();owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();dest=S.OUT/"restart";indices=[1268,1373,2729,2857,3546,3820];rows=[]
    for seed in range(3):
        model=S.model_for(seed,90,"original","cuda")
        for index in indices:
            shard,i=divmod(index,256);lo=i//32*32;data=np.load(dest/f"shard{shard:02d}.npz");output=np.load(dest/f"runs/shard{shard:02d}-s{seed}/outputs.npz")
            t=torch.tensor(data["tokens"][lo:lo+32].astype(np.int64),device="cuda")
            with torch.no_grad():
                psi,_=model.settle(t,tol=1e-6,max_steps=4000,check=8);a=model.couplings(t,psi);tt=t[i-lo:i-lo+1];aa=a[i-lo:i-lo+1]
                boards=np.concatenate([output["predictions"][:,i],data["labels"][i][None]])
                bt=torch.tensor(boards,device="cuda",dtype=aa.dtype);energy=model.energy(tt.expand(6,-1,-1),bt,aa.expand(6,-1,-1,-1));margin=model.flip_delta(tt.expand(6,-1,-1),bt,aa.expand(6,-1,-1,-1))
                saved=output["energy"][:,i];delta=float(np.max(np.abs(energy[:5].cpu().numpy()-saved)))
                assert delta<.05,delta
                for variant in range(6):
                    free=tt[0]==1;margins=margin[variant][free];pred=boards[variant];tok=data["tokens"][i]
                    rows.append({"input_index":index,"seed":seed,"variant":"native" if variant==0 else "canonical_oracle_diagnostic" if variant==5 else f"restart{variant-1}","graph_distance_evaluation_only":int(data["shortest_edges"][i]),"saved_energy":float(saved[variant]) if variant<5 else None,"recomputed_energy":float(energy[variant]),"energy_minus_canonical":float(energy[variant]-energy[5]),"minimum_free_flip_margin":float(margins.min()),"negative_free_flip_margins":int((margins<0).sum()),"feasible":bool(output["feasible"][variant,i]) if variant<5 else True,"selected_by_energy":variant==int(output["selected_energy"][i]),"selected_by_screen":variant==int(output["selected_screen"][i]),"screen_returns_no_feasible":int(output["selected_screen"][i])<0,"certified":bool(output["stopped"][variant,i]) if variant<5 else None,"answer_steps":int(output["answer_steps"][variant,i]) if variant<5 else None,**classify(tok,pred)})
            S.emit("restart_failure_checked",input_index=index,seed=seed,energy_recompute_error=delta)
    result={"status":"diagnosed","utc":S.utc(),"post_outcome_descriptive":True,"indices":indices,"native_joint_failures":[2729,2857,3546,3820],"screen_joint_failures":indices,"no_native_failure_recovered":True,"additional_screen_failures":[1268,1373],"native_failures_subset_of_screen_failures":True,"rows":rows,"script_sha256":S.sha(__file__),"protocol_sha256":S.sha(S.HERE/"restart-protocol.json"),"scope":"All six failed inputs under native or screened selector; no answer reruns or changed initialization. Hidden readout recomputed in original32-input batches. Labels used only for diagnostic canonical-path energy, never selection."}
    S.save(dest/"failure-diagnostics.json",result)


if __name__=="__main__":main()
