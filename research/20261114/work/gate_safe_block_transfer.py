"""H2an same120 objective/source output identity gate."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import safe_block_batch as Q
from safe_block_transfer import PP,DEST,protocol,load_models,hidden
from gate_factorial_transfer import literal_check
from verify_graph_block_source import connected


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();DEST.mkdir(exist_ok=True);started=time.perf_counter();rows=[];checks=0;max_error=0.
    with np.load(S.OUT/'factorial_transfer/shard00.npz') as z:tokens=z['tokens'][:64]
    t=torch.tensor(tokens.astype(np.int64),device='cuda')
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed);a,_,good=hidden(models[0],t);assert good.all();r=Q.solve(models,doubles,t,a,rotation=seed)
            with np.load(S.OUT/f'factorial_transfer/runs/shard00-s{seed}/outputs.npz') as z:readout=z['couplings'][:64];pred=z['predictions'][[0,1,3],:64];steps=z['iterations'][[0,1,3],:64];stop=z['stopped'][[0,1,3],:64]
            assert np.array_equal(readout,a.cpu().numpy())
            for v in range(3):
                for key,old in [('prediction',pred[v]),('iterations',steps[v]),('stopped',stop[v])]:assert np.array_equal(r['outputs'][v][key].cpu().numpy(),old)
            with np.load(S.OUT/f'connectivity_screen/veto/shard00-s{seed}/outputs.npz') as z:
                for key in ['prediction','iterations','stopped','finite']:assert np.array_equal(r['outputs'][3][key].cpu().numpy(),z[key][:64])
            for v in range(4):
                board=r['outputs'][v]['prediction'];assert torch.equal(board[t!=1],(t>=2).to(board.dtype)[t!=1]);margin=doubles[Q.MODEL_INDEX[v]].flip_delta(t,board.double(),a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1);assert (margin[r['outputs'][v]['stopped']]>=-1e-6).all()
            for v in [2,3]:
                raw=r['blocks'][v]['raw'];ix=r['blocks'][v]['valid_prefix_indices'];assert len(ix)==64 and np.array_equal(raw['states_before'][0],r['prefixes'][1]['prediction'].cpu().numpy())
                for record,boards in zip(raw['trace'],raw['states_before']):
                    items=[(z['local_input'],z['cells']) for z in record['candidates']];values=[z['delta'] for z in record['candidates']]
                    if items:max_error=max(max_error,literal_check(doubles[1],t,a.double(),torch.tensor(boards,device='cuda',dtype=torch.float64),items,values));checks+=len(items)
                    if v==3:
                        for i,cells in items:changed=boards[i].copy();changed.flat[cells]=False;assert connected(tokens[i],changed)
            rows.append({'seed':seed,'seen_inputs':64,'exact_archived_fourarm_boards_iterations_flags':True,'exact_readouts_prefixreuse_clamps_certificates':True});assert time.perf_counter()-started<120
    result={'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'rows':rows,'candidate_literal_checks':checks,'maximum_literal_error':max_error,'seconds':time.perf_counter()-started,'new_independent_inputs':0,'script_sha256':S.sha(__file__)};S.save(DEST/'numerical-gate.json',result);S.emit('safe_block_source_gate_passed',seconds=result['seconds'])


if __name__=='__main__':main()
