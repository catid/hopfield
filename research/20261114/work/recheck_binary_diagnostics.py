"""Recompute retained diagnostics in original batches; no answer trajectories."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import numpy as np
import study as S


def main(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid","--format=csv,noheader"],text=True).strip();owners=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();dest=S.OUT/"binary_relaxation";pp=S.HERE/"binary_relaxation-protocol.json";protocol=json.loads(pp.read_text());rows=[];started=time.perf_counter()
    strengths=protocol["continuous_intervention"]["strengths"]
    for job in protocol["jobs"]:
        if job["gpu"]!=gpu:continue
        run=dest/f"runs/{job['id']}";record=json.loads((run/"result.json").read_text());assert S.sha(run/"outputs.npz")==record["outputs_sha256"]
        with np.load(run/"outputs.npz") as archive:out={key:archive[key] for key in archive.files}
        data=np.load(dest/f"shard{job['shard']:02d}.npz");model=S.model_for(job["seed"],90,"original","cuda");errors={key:0. for key in ["energy","minimum_original_flip_margin","original_continuous_energy","modified_continuous_energy","continuous_regularizer","modified_field_residual"]}
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(data["tokens"][lo:lo+64].astype(np.int64),device="cuda");psi,_=model.settle(t,tol=1e-6,max_steps=4000,check=8);a=model.couplings(t,psi);free=model.free(t)
                for v,strength in enumerate(strengths):
                    binary=torch.tensor(out["predictions"][v,lo:lo+64],device="cuda",dtype=a.dtype);u=torch.tensor(out["final_logits"][v,lo:lo+64],device="cuda");p=torch.sigmoid(u)
                    e=model.energy(t,binary,a);m=model.flip_delta(t,binary,a).masked_fill(~free,float("inf")).flatten(1).amin(1)
                    original=model.energy(t,p,a);regularizer=strength*(free.to(p.dtype)*p*(1-p)).flatten(1).sum(1)
                    field=model.field(t,p,a)+strength*free.to(p.dtype)*(2*p-1);residual=(field-u).abs().masked_fill(~free,0).flatten(1).amax(1)
                    values={"energy":e,"minimum_original_flip_margin":m,"original_continuous_energy":original,"modified_continuous_energy":original+regularizer,"continuous_regularizer":regularizer,"modified_field_residual":residual}
                    assert bool((strength*(free.to(binary.dtype)*binary*(1-binary)).flatten(1).sum(1)==0).all())
                    for key,value in values.items():
                        actual=value.cpu().numpy();expected=out[key][v,lo:lo+64];assert np.array_equal(np.isfinite(actual),np.isfinite(expected))
                        finite=np.isfinite(actual);error=float(np.abs(actual[finite]-expected[finite]).max()) if finite.any() else 0.;errors[key]=max(errors[key],error)
                        assert np.allclose(actual,expected,rtol=0,atol=1e-5,equal_nan=True),(job,key,error)
        rows.append({"job":job["id"],"model_input_diagnostics":1024,"maximum_errors":errors,"outputs_sha256":record["outputs_sha256"]});S.emit("binary_diagnostic_rechecked",gpu=gpu,job=job["id"],max_error=max(errors.values()))
    result={"status":"passed","utc":S.utc(),"gpu":gpu,"rows":rows,"model_input_diagnostics":sum(r["model_input_diagnostics"] for r in rows),"new_answer_rollouts":0,"protocol_sha256":S.sha(pp),"script_sha256":S.sha(__file__),"seconds":time.perf_counter()-started,"method":"Original hidden readout recomputed in identical64-input batches; frozen native field/energy/flip functions plus independently written compensated term recompute every retained final diagnostic. No new answer update or rescue."}
    S.save(dest/f"diagnostic-recheck-gpu{gpu}.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",type=int,choices=[0,1],required=True);main(parser.parse_args().gpu)
