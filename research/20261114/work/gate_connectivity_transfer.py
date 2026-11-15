"""H2al source-only wall-padding size gate before new180maze generation."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import connectivity_batch as Q
from connectivity_transfer import PP,DEST,protocol,load_models,hidden
from gate_factorial_transfer import literal_check
from audit_connectivity_screen import path_valid


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();DEST.mkdir(exist_ok=True);started=time.perf_counter();rows=[];max_delta=0.;checks=0
    with np.load(S.OUT/'factorial_transfer/shard00.npz') as z:original=z['tokens'][:64]
    tokens=np.pad(original,((0,0),(0,60),(0,60)));t=torch.tensor(tokens.astype(np.int64),device='cuda')
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed);a,hs,good=hidden(models[0],t);assert good.all()
            with np.load(S.OUT/f'factorial_transfer/runs/shard00-s{seed}/outputs.npz') as z:prior_a=z['couplings'][:64];prior_pred=z['predictions'][:2,:64];prior_steps=z['iterations'][:2,:64]
            inside=a[:,:,:120,:120].cpu().numpy();mask=np.repeat((original>0)[:,None],5,axis=1);assert np.array_equal(inside[mask],prior_a[mask]);r=Q.solve(models,doubles,t,a,rotation=seed)
            for v in [0,1]:assert np.array_equal(r['outputs'][v]['prediction'].cpu().numpy()[:,:120,:120],prior_pred[v]) and np.array_equal(r['outputs'][v]['iterations'].cpu().numpy(),prior_steps[v])
            simple=np.array([path_valid(tok,b) for tok,b in zip(tokens,r['prefixes'][1]['prediction'].cpu().numpy().astype(bool))]);assert np.array_equal(simple,r['blocks'][2]['initial_simple_path'])
            for v in range(4):
                board=r['outputs'][v]['prediction'];assert torch.equal(board[t!=1],(t>=2).to(board.dtype)[t!=1]);margin=doubles[Q.MODEL_INDEX[v]].flip_delta(t,board.double(),a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1);assert (margin[r['outputs'][v]['stopped']]>=-1e-6).all()
            assert torch.equal(r['outputs'][2]['prediction'][simple],r['prefixes'][1]['prediction'][simple])
            for v in [2,3]:
                raw=r['blocks'][v]['raw'];ix=r['blocks'][v]['valid_prefix_indices']
                if raw is None:assert len(ix)==0;continue
                assert np.array_equal(raw['states_before'][0],r['prefixes'][1]['prediction'].cpu().numpy()[ix])
                for record,states in zip(raw['trace'],raw['states_before']):
                    items=[(x['local_input'],x['cells']) for x in record['candidates']];values=[x['delta'] for x in record['candidates']]
                    if items:max_delta=max(max_delta,literal_check(doubles[1],t[ix],a[ix].double(),torch.tensor(states,device='cuda',dtype=torch.float64),items,values));checks+=len(items)
            rows.append({'seed':seed,'wall_padded_seen_inputs':64,'exact_free_readouts_bp_binary_prefix_and_iteration_parity':True,'skip_path_identity':True,'all_arm_clamps_certificates':True,'timing':r['timing']});assert time.perf_counter()-started<120
    S.save(DEST/'numerical-gate.json',{'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'rows':rows,'candidate_literal_checks':checks,'maximum_literal_error':max_delta,'seconds':time.perf_counter()-started,'new_independent_inputs':0,'scope':'Previouslyseen120mazes wallpaddedto180, numerical/sizegate only, no outcomes used for tuning.','script_sha256':S.sha(__file__)});S.emit('connectivity_transfer_size_gate_passed',seconds=time.perf_counter()-started,checks=checks)


if __name__=='__main__':main()
