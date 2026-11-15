"""Read-only H2af source identities and complete cached/reference replays."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as G
import binary_coordinate as C
from mine_leaf_negatives import PP,DEST,TRAIN,PRIOR,protocol


def main(gpu):
    lock,inventory=G.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter();rows=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens']
    gate=json.loads((S.OUT/'second_negative_transfer/cast-gate.json').read_text())
    for seed in range(3):
        if seed%2!=gpu:continue
        original=S.model_for(seed,30,'original','cuda');model=S.model_for(seed,30,'original','cuda');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
        for key,value in model.state_dict().items():
            if key not in ['T','b','kappa']:assert torch.equal(value,original.state_dict()[key]),key
        original_theta=np.concatenate([original.T.cpu().numpy().ravel(),original.b.cpu().numpy(),original.kappa.cpu().numpy()]);candidate=np.load(PRIOR/f'seed{seed}/candidate-theta.npy');scale=float(np.max(abs(original_theta)).astype(np.float64)/np.max(abs(candidate)));cast=np.asarray(candidate*scale,np.float32);expected=np.load(S.ROOT/p['coefficient_arrays'][str(seed)]['path']);actual=np.concatenate([model.T.cpu().numpy().ravel(),model.b.cpu().numpy(),model.kappa.cpu().numpy()]);assert np.array_equal(cast,expected) and np.array_equal(actual,expected) and scale==gate['records'][seed]['positive_scale']
        double=S.model_for(seed,30,'original','cuda',torch.float64);double.load_state_dict(model.state_dict())
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        receipt=json.loads((DEST/f'mined-seed{seed}.json').read_text());assert S.sha(DEST/f'mined-seed{seed}.npz')==receipt['outputs_sha256']
        with np.load(DEST/f'mined-seed{seed}.npz') as z:saved={k:z[k] for k in z.files}
        timings={'cached':0.,'reference':0.};max_margin_error=0.
        with torch.no_grad():
            for lo in range(0,1000,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');psi,_=model.settle(t,tol=1e-6,max_steps=4000,check=8);aa=model.couplings(t,psi);assert np.array_equal(aa.cpu().numpy(),attention[lo:lo+64]);a=aa.double();cache=C.BinaryQuadratic(double,t,a)
                class Reference:
                    free=cache.free;colors=cache.colors
                    def initial(self):return cache.initial()
                    def delta(self,b):return double.flip_delta(t,b,a)
                for name,obj in [('cached',cache),('reference',Reference())]:
                    torch.cuda.synchronize();before=time.perf_counter();out=C.solve(obj);torch.cuda.synchronize();timings[name]+=time.perf_counter()-before
                    for key in ['prediction','sweeps','stopped','finite','flip_count']:assert np.array_equal(out[key].cpu().numpy(),saved[key][lo:lo+64]),(seed,lo,name,key)
                    error=float(abs(out['minimum_flip_margin'].cpu().numpy()-saved['minimum_flip_margin'][lo:lo+64]).max());assert error<1e-10;max_margin_error=max(max_margin_error,error)
        rows.append({'seed':seed,'source_inputs':1000,'all_scaling_and_cast_values_exact':True,'hidden_state_exactly_original':True,'recomputed_readouts_exact':True,'saved_boards_sweeps_flags_flips_exact_for_both_replays':True,'maximum_reference_margin_error':max_margin_error,'timings_seconds':timings,'source_output_sha256':receipt['outputs_sha256'],'checkpoint_sha256':p['models'][str(seed)]['sha256']})
    S.save(DEST/f'source-replay-gpu{gpu}.json',{'status':'passed','inventory':inventory,'rows':rows,'seconds':time.perf_counter()-started,'new_independent_test_inputs':0,'source_replay_trajectories':2*sum(r['source_inputs'] for r in rows),'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'scope':'Full same-input cached and independent upstream flip-delta reference replays, plus exact checkpoint scaling/hidden/readout identities. Read-only verification; no refitting or new scientific inputs.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True,choices=[0,1]);main(p.parse_args().gpu)
