"""Post-outcome forensic replay only; preserves the original eta trial unchanged."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import numpy as np
import study as S


def main(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();from cram.maze_learned import gather
    source=S.OUT/"eta_boundary";dest=source/"failure-forensics";dest.mkdir(exist_ok=True)
    data=np.load(source/"loops20.npz");index=38;lo=32;local=index-lo
    t=torch.as_tensor(data["tokens"][lo:lo+32].astype(np.int64),device="cuda")
    for seed in range(3):
        if (3+seed)%2!=gpu:continue
        model=S.model_for(seed,90,"original","cuda")
        with torch.no_grad():
            psi,hidden_steps=model.settle(t,tol=1e-6,max_steps=4000,check=8);a=model.couplings(t,psi)
            lg,la,lc=model.parts(t,model.mu);new=la-torch.logsumexp(torch.cat([lg-gather(psi,model.zoffs,math.inf),lc[:,None]],1),1)
            hidden_good=(((new==psi)|((new-psi).abs()<=1e-6*(1+new.abs()))).flatten(1).all(1)&(torch.isfinite(psi)|(t==0)).flatten(1).all(1))
            hidden_abs=torch.where(t>0,(new-psi).abs(),0).flatten(1).amax(1)
            for eta in (.2,.5,.8,1.):
                u=torch.full(t.shape,3.,device="cuda",dtype=a.dtype);stopped=torch.zeros(len(t),dtype=torch.bool,device="cuda")
                used=torch.full((len(t),),8192,dtype=torch.long,device="cuda");history=[];old1=None;old2=None;cycle=False
                for k in range(1,8193):
                    nxt=u+eta*(model.field(t,torch.sigmoid(u),a)-u);u=torch.where(stopped[:,None,None],u,nxt)
                    if old2 is not None:
                        free=t[local]==1;one=(u[local][free]-old1[free]).abs().max();two=(u[local][free]-old2[free]).abs().max()
                        cycle|=bool((two<1e-5)&(one>1e-3)&~stopped[local])
                    old2=old1;old1=u[local].clone()
                    if k%16==0:
                        h=(u>0).to(a.dtype);delta=model.flip_delta(t,h,a);ok=(delta>=-1e-6).flatten(1).all(1)
                        used=torch.where(ok&~stopped,torch.full_like(used,k),used);stopped|=ok
                        continuous=torch.sigmoid(u);field=model.field(t,continuous,a)
                        history.append({"update":k,"continuous_energy":float(model.energy(t,continuous,a)[local]),
                                        "discrete_energy":float(model.energy(t,h,a)[local]),"minimum_flip_delta":float(delta[local].min()),
                                        "negative_flip_count":int((delta[local]<0).sum()),"residual_free_max":float((field[local][free]-u[local][free]).abs().max()),
                                        "finite":bool(torch.isfinite(u[local]).all()),"stopped":bool(stopped[local]),"selected_cells":int(h[local].sum())})
                        if stopped.all():break
                prediction=(u>0).to(a.dtype);saved=np.load(source/f"runs/loops20-s{seed}/eta{eta}.npz")
                assert np.array_equal(prediction.cpu().bool().numpy(),saved["predictions"][lo:lo+32])
                assert np.array_equal(used.cpu().numpy(),saved["answer_steps"][lo:lo+32])
                assert np.array_equal(stopped.cpu().numpy(),saved["stopped"][lo:lo+32])
                continuous=torch.sigmoid(u);field=model.field(t,continuous,a);delta=model.flip_delta(t,prediction,a)
                target=torch.as_tensor(data["labels"][lo:lo+32],device="cuda",dtype=a.dtype)
                path=dest/f"s{seed}-eta{eta}-states.npz"
                if path.exists():raise FileExistsError(path)
                arrays={"tokens":data["tokens"][index],"oracle_label":data["labels"][index],"hidden_psi":psi[local].cpu().numpy(),
                        "couplings":a[local].cpu().numpy(),"logits":u[local].cpu().numpy(),"continuous_answer":continuous[local].cpu().numpy(),
                        "prediction":prediction[local].cpu().numpy(),"continuous_field":field[local].cpu().numpy(),"flip_delta":delta[local].cpu().numpy()}
                np.savez_compressed(path,**arrays)
                record={"status":"replayed_exactly","utc":S.utc(),"post_outcome_diagnostic":True,"dataset_index":index,"original_batch_start":lo,
                        "seed":seed,"eta":eta,"gpu":gpu,"native_saved_batch_parity":True,"hidden_steps":int(hidden_steps[local]),
                        "hidden_good":bool(hidden_good[local]),"hidden_absolute_residual":float(hidden_abs[local]),"answer_steps":int(used[local]),
                        "stopped":bool(stopped[local]),"two_cycle_detected":cycle,"minimum_flip_delta":float(delta[local].min()),
                        "negative_flip_count":int((delta[local]<0).sum()),"discrete_energy":float(model.energy(t,prediction,a)[local]),
                        "oracle_label_energy":float(model.energy(t,target,a)[local]),"history":history,"states_sha256":S.sha(path),
                        "script_sha256":S.sha(__file__),"checkpoint_sha256":S.sha(S.OUT/f"models/seed{seed}.pt"),
                        "dataset_sha256":S.sha(source/"loops20.npz"),"original_output_sha256":S.sha(source/f"runs/loops20-s{seed}/eta{eta}.npz")}
                S.save(dest/f"s{seed}-eta{eta}.json",record);S.emit("failure_replayed",seed=seed,eta=eta,steps=record["answer_steps"],minimum_flip_delta=record["minimum_flip_delta"],cycle=cycle)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",type=int,choices=[0,1],required=True);main(parser.parse_args().gpu)
