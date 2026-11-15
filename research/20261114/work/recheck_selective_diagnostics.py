"""Recompute native, always-lambda and queued-fallback diagnostics without answer updates."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import numpy as np
import study as S


def main(gpu):
    lock=(S.ROOT/f"artifacts/gpu{gpu}.lock").open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True);assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    torch,_,_=S.imports();dest=S.OUT/"selective_fallback";pp=S.HERE/"selective_fallback-protocol.json";p=json.loads(pp.read_text());started=time.perf_counter();rows=[];queues={s:[] for s in range(3)}
    def check(model,t,a,out,strength):
        binary=torch.tensor(out['predictions'],device='cuda',dtype=a.dtype);u=torch.tensor(out['final_logits'],device='cuda');soft=torch.sigmoid(u);free=model.free(t)
        energy=model.energy(t,binary,a);margin=model.flip_delta(t,binary,a).masked_fill(~free,float('inf')).flatten(1).amin(1)
        original=model.energy(t,soft,a);reg=strength*(free.to(soft.dtype)*soft*(1-soft)).flatten(1).sum(1)
        field=model.field(t,soft,a)+strength*free.to(soft.dtype)*(2*soft-1);res=(field-u).abs().masked_fill(~free,0).flatten(1).amax(1)
        values={'energy':energy,'minimum_original_flip_margin':margin,'original_continuous_energy':original,'modified_continuous_energy':original+reg,'continuous_regularizer':reg,'modified_field_residual':res};errors={}
        assert bool((strength*(free.to(binary.dtype)*binary*(1-binary)).flatten(1).sum(1)==0).all())
        for key,value in values.items():
            actual=value.cpu().numpy();expected=out[key];assert np.array_equal(np.isfinite(actual),np.isfinite(expected));finite=np.isfinite(actual)
            errors[key]=float(np.abs(actual[finite]-expected[finite]).max()) if finite.any() else 0.
            assert np.allclose(actual,expected,rtol=0,atol=1e-5,equal_nan=True),(key,errors[key])
        return errors
    with torch.no_grad():
        for job in p['jobs']:
            if job['gpu']!=gpu:continue
            run=dest/f"runs/{job['id']}";record=json.loads((run/'result.json').read_text());assert record['outputs_sha256']==S.sha(run/'outputs.npz')
            with np.load(run/'outputs.npz') as x:out={k:x[k] for k in x.files}
            with np.load(dest/f"shard{job['shard']:02d}.npz") as x:data={k:x[k] for k in ['tokens','indices']}
            model=S.model_for(job['seed'],120,'original','cuda');errors={}
            for lo in range(0,256,64):
                t=torch.tensor(data['tokens'][lo:lo+64].astype(np.int64),device='cuda');psi,_=model.settle(t,tol=1e-6,max_steps=4000,check=8);a=model.couplings(t,psi)
                for v in [0,1]:
                    sample={k:x[v,lo:lo+64] for k,x in out.items() if k not in ['native_accepted','hidden_good','hidden_steps']}
                    got=check(model,t,a,sample,float(v));errors={k:max(errors.get(k,0.),x) for k,x in got.items()}
                reject=np.flatnonzero(~out['native_accepted'][lo:lo+64])
                if len(reject):
                    idx=torch.tensor(reject,device='cuda');queues[job['seed']].append((t[idx],a[idx],data['indices'][lo:lo+64][reject]))
            rows.append({'job':job['id'],'diagnostics_checked':512,'maximum_errors':errors,'outputs_sha256':record['outputs_sha256']});S.emit('selective_diagnostics_rechecked',gpu=gpu,job=job['id'],max_error=max(errors.values()))
        for seed,parts in queues.items():
            if not parts:continue
            model=S.model_for(seed,120,'original','cuda');tokens=torch.cat([x[0] for x in parts]);couplings=torch.cat([x[1] for x in parts]);indices=np.concatenate([x[2] for x in parts]);path=dest/f'fallback-gpu{gpu}-seed{seed}.npz'
            with np.load(path) as x:out={k:x[k] for k in x.files}
            assert np.array_equal(indices,out['indices']);errors={}
            for lo in range(0,len(indices),64):
                sample={k:x[lo:lo+64] for k,x in out.items() if k not in ['indices','hidden_good']};got=check(model,tokens[lo:lo+64],couplings[lo:lo+64],sample,1.);errors={k:max(errors.get(k,0.),x) for k,x in got.items()}
            rows.append({'job':f'fallback-gpu{gpu}-seed{seed}','diagnostics_checked':len(indices),'maximum_errors':errors,'outputs_sha256':S.sha(path)})
    S.save(dest/f'diagnostic-recheck-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'gpu':gpu,'rows':rows,'diagnostics_checked':sum(r['diagnostics_checked'] for r in rows),'protocol_sha256':S.sha(pp),'script_sha256':S.sha(__file__),'seconds':time.perf_counter()-started,'new_answer_trajectories':0,'method':'Original64-input hidden readouts and exact queued subset order reconstructed; independent compensated energy/field expression recomputes all retained diagnostics.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],required=True);main(parser.parse_args().gpu)
