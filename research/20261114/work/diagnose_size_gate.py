"""Retained H2al failed padding gate diagnosis; no fresh inputs or tuning."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import connectivity_batch as Q
from connectivity_transfer import PP,DEST,protocol,load_models,hidden


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();started=time.perf_counter();rows=[]
    with np.load(S.OUT/'factorial_transfer/shard00.npz') as z:original=z['tokens'][:64]
    tokens=np.pad(original,((0,0),(0,60),(0,60)));t=torch.tensor(tokens.astype(np.int64),device='cuda')
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed);a,hs,good=hidden(models[0],t);r=Q.solve(models,doubles,t,a,rotation=seed)
            with np.load(S.OUT/f'factorial_transfer/runs/shard00-s{seed}/outputs.npz') as z:prior={k:z[k] for k in ['couplings','predictions','iterations','native_final_logits']}
            for v in [0,1]:
                pred=r['outputs'][v]['prediction'].cpu().numpy()[:,:120,:120];steps=r['outputs'][v]['iterations'].cpu().numpy();diff=pred!=prior['predictions'][v,:64];ix=np.flatnonzero(np.any(diff,axis=(1,2)));rows.append({'seed':seed,'arm':Q.ARMS[v],'different_boards':len(ix),'different_cell_count':int(diff.sum()),'indices':ix.tolist(),'changed_steps_indices':np.flatnonzero(steps!=prior['iterations'][v,:64]).tolist(),'old_steps_at_differences':prior['iterations'][v,:64][steps!=prior['iterations'][v,:64]].tolist(),'new_steps_at_differences':steps[steps!=prior['iterations'][v,:64]].tolist(),'all_stopped':bool(r['outputs'][v]['stopped'].all()),'all_finite':bool(r['outputs'][v]['finite'].all())})
    S.save(DEST/'failed-padding-gate-diagnosis.json',{'status':'diagnosed_failed_gate','protocol_sha256':S.sha(PP),'rows':rows,'inventory':inventory,'seconds':time.perf_counter()-started,'fresh_inputs':0,'source_sha256':S.sha(__file__)});print(json.dumps(rows))


if __name__=='__main__':main()
