"""Recheck every H2aa readout/finalenergy/margin without new search paths."""
import argparse
import copy
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
from binary_coordinate_study import PP,DEST,ARMS,protocol


def main(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter();rows=[];models={};doubles={}
    for seed in range(3):
        models[seed]=[];doubles[seed]=[]
        for arm in ARMS[:2]:
            model=S.model_for(seed,120,'original','cuda');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location='cuda',weights_only=True));models[seed].append(model);doubles[seed].append(copy.deepcopy(model).double())
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256']
        with np.load(run/'outputs.npz') as x:out={k:x[k] for k in ['predictions','couplings','minimum_flip_margin64','energy64','native_minimum_flip_margin32','binary_cache_margin']}
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as x:tokens=x['tokens']
        errors={k:0. for k in ['couplings','energy64','minimum_flip_margin64','native_minimum_flip_margin32','binary_cache_margin']}
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');base=models[job['seed']][0];psi,_=base.settle(t,tol=1e-6,max_steps=4000,check=8);a=base.couplings(t,psi);assert np.array_equal(a.cpu().numpy(),out['couplings'][lo:lo+64]);ad=a.double()
                for v in range(3):
                    model=doubles[job['seed']][0 if v==0 else 1];board=torch.tensor(out['predictions'][v,lo:lo+64],dtype=torch.float64,device='cuda');values={'energy64':model.energy(t,board,ad),'minimum_flip_margin64':model.flip_delta(t,board,ad).masked_fill(t!=1,float('inf')).flatten(1).amin(1)}
                    if v<2:values['native_minimum_flip_margin32']=models[job['seed']][v].flip_delta(t,board.float(),a).masked_fill(t!=1,float('inf')).flatten(1).amin(1)
                    else:
                        cache=C.BinaryQuadratic(model,t,ad);values['binary_cache_margin']=cache.delta(board).masked_fill(t!=1,float('inf')).flatten(1).amin(1)
                    for key,value in values.items():
                        actual=value.cpu().numpy();expected=out[key][lo:lo+64] if key=='binary_cache_margin' else out[key][v,lo:lo+64];assert np.array_equal(actual,expected),(job,key,float(np.max(abs(actual-expected))));errors[key]=max(errors[key],float(np.max(abs(actual-expected))))
        rows.append({'job':job['id'],'model_input_diagnostics':768,'maximum_errors':errors});S.emit('binary_diagnostics_rechecked',job=job['id'],gpu=gpu)
    S.save(DEST/f'diagnostic-recheck-gpu{gpu}.json',{'status':'passed','utc':S.utc(),'protocol_sha256':S.sha(PP),'rows':rows,'model_input_diagnostics':sum(r['model_input_diagnostics'] for r in rows),'new_answer_trajectories':0,'seconds':time.perf_counter()-started,'inventory':inventory,'script_sha256':S.sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);main(p.parse_args().gpu)
