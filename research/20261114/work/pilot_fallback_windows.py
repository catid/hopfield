"""Prior-cohort timing replay for fixed-window planning; no fresh confirmation data."""
import argparse
import json
import time
import numpy as np
import study as S
import larger_relaxation as L
from selective_fallback import acquire,cpu_answer


def main(gpu):
    lock,inventory=acquire(gpu);torch,_,_=S.imports();source=S.OUT/'selective_fallback';dest=S.OUT/'window_fallback';dest.mkdir(parents=True,exist_ok=True);p=json.loads((S.HERE/'selective_fallback-protocol.json').read_text());models={s:S.model_for(s,120,'original','cuda') for s in range(3)};queues={s:[] for s in range(3)};counts={s:0 for s in range(3)};reference={}
    started=time.perf_counter()
    for seed in range(3):
        with np.load(source/f'fallback-gpu{gpu}-seed{seed}.npz') as x:
            for i,index in enumerate(x['indices']):reference[(seed,int(index))]={k:x[k][i] for k in ['predictions','answer_steps','stopped','finite']}
    with torch.no_grad():
        for job in p['jobs']:
            if job['gpu']!=gpu:continue
            seed=job['seed'];model=models[seed]
            with np.load(source/f"runs/{job['id']}/outputs.npz") as x:accept=x['native_accepted']
            with np.load(source/f"shard{job['shard']:02d}.npz") as x:tokens=x['tokens'];indices=x['indices']
            for lo in range(0,256,64):
                reject=np.flatnonzero(~accept[lo:lo+64])
                if not len(reject):continue
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');psi,_=model.settle(t,tol=1e-6,max_steps=4000,check=8);a=model.couplings(t,psi);idx=torch.tensor(reject,device='cuda')
                queues[seed].append((t[idx],a[idx],indices[lo:lo+64][reject],counts[seed]+lo+reject))
            counts[seed]+=256
    pools={s:(torch.cat([x[0] for x in q]),torch.cat([x[1] for x in q]),np.concatenate([x[2] for x in q]),np.concatenate([x[3] for x in q])) for s,q in queues.items()};preparation=time.perf_counter()-started;rows=[]
    for rep in range(3):
        for width in ([256,1024,8192] if rep%2==0 else [8192,1024,256]):
            seconds=0.;batches=0;n=0;maximum=0
            for seed,(t,a,ids,positions) in pools.items():
                for window in np.unique(positions//width):
                    keep=np.flatnonzero(positions//width==window);idx=torch.tensor(keep,device='cuda');torch.cuda.synchronize();begin=time.perf_counter();qt=t[idx];qa=a[idx];u=torch.full_like(qa[:,0],3.);output=L.solve(models[seed],qt,qa,u,8192,1.);cpu=cpu_answer(output);torch.cuda.synchronize();seconds+=time.perf_counter()-begin
                    for i,j in enumerate(keep):
                        ref=reference[(seed,int(ids[j]))]
                        for key in ref:assert np.array_equal(cpu[key][i],ref[key]),(width,seed,int(ids[j]),key)
                    batches+=1;n+=len(keep);maximum=max(maximum,len(keep))
            rows.append({'repetition':rep,'window':width,'fallback_seconds_including_gather_init_and_output':seconds,'batches':batches,'rejected_model_inputs':n,'maximum_batch':maximum,'exact_board_step_flag_parity':True});S.emit('window_pilot_result',gpu=gpu,**rows[-1])
    S.save(dest/f'prior-cohort-pilot-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'gpu':gpu,'inventory':inventory,'rows':rows,'preparation_seconds':preparation,'worker_seconds':time.perf_counter()-started,'source_audit_sha256':S.sha(source/'audit.json'),'script_sha256':S.sha(__file__),'evidence_role':'Post-outcome performance-development replay of H2t rejected inputs, excluded from upcoming fresh confirmation and accuracy evidence. Fixed1024 chosen by coordinator before these timings;256/8192 shown as development comparators, no outcome-dependent tuning.','new_fresh_inputs':0,'answer_trajectories':sum(r['rejected_model_inputs'] for r in rows)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],required=True);main(parser.parse_args().gpu)
