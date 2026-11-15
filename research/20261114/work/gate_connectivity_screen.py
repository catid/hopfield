"""H2ak independent reachability gate and fixed seen120 source intervention."""
from collections import deque
import itertools
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import connected_block_moves as B
import graph_block_moves as OLD
from connectivity_screen import PP,DEST,PARENT,protocol
from gate_factorial_transfer import literal_check


def bfs(t,b):
    start=tuple(np.argwhere(t==2)[0]);goal=tuple(np.argwhere(t==3)[0]);q=deque([start]);seen={start};h,w=t.shape
    if not b[start] or not b[goal]:return False
    while q:
        r,c=q.popleft()
        for rr,cc in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
            if 0<=rr<h and 0<=cc<w and b[rr,cc] and (rr,cc) not in seen:seen.add((rr,cc));q.append((rr,cc))
    return goal in seen


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();DEST.mkdir(exist_ok=True);started=time.perf_counter();fixture=np.array([[2,1,0],[1,1,3],[0,1,1]]);pos=np.argwhere(fixture==1);checks=0
    for bits in itertools.product([0,1],repeat=len(pos)):
        board=fixture>=2
        for (r,c),bit in zip(pos,bits):board[r,c]=bit
        assert B.reachable(fixture,board)==bfs(fixture,board);raw,_=OLD.candidates(fixture,board);kept,_=B.candidates(fixture,board);expected=[]
        for cells in raw:
            z=board.copy();z.flat[list(cells)]=False;answer=bfs(fixture,z);assert answer==B.reachable(fixture,board,cells);checks+=1
            if answer:expected.append(cells)
        assert kept==expected
    rows=[];error=0.
    for seed in range(3):
        model=S.model_for(seed,120,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
        assert S.sha(S.ROOT/p['datasets'][0]['path'])==p['datasets'][0]['sha256'];source=PARENT/f'runs/shard00-s{seed}/outputs.npz';assert S.sha(source)==p['parent_results'][f'shard00-s{seed}']['outputs_sha256']
        with np.load(S.ROOT/p['datasets'][0]['path']) as z:tokens=z['tokens'][:64]
        with np.load(source) as z:initial=z['predictions'][1,:64];attention=z['couplings'][:64]
        t=torch.tensor(tokens.astype(np.int64),device='cuda');a=torch.tensor(attention,device='cuda',dtype=torch.float64);board=torch.tensor(initial,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,t,a);out=B.solve(cache,board)
        for record,states in zip(out['trace'],out['states_before']):
            items=[(r['local_input'],r['cells']) for r in record['candidates']];values=[r['delta'] for r in record['candidates']]
            for i,cells in items:z=states[i].copy();z.flat[cells]=False;assert bfs(tokens[i],z);checks+=1
            if items:error=max(error,literal_check(model,t,a,torch.tensor(states,device='cuda',dtype=torch.float64),items,values))
        for tok,initial_board,final in zip(tokens,initial,out['prediction'].cpu().numpy()):
            if B.simple_path(tok,initial_board):assert np.array_equal(initial_board,final)
        rows.append({'seed':seed,'seen_inputs':64,'moves':int(out['block_moves'].sum()),'unchanged_initial_simple_paths':True});assert time.perf_counter()-started<120
    S.save(DEST/'numerical-gate.json',{'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'rows':rows,'independent_reachability_checks':checks,'maximum_literal_delta_error':error,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'fresh_inputs':0});S.emit('connectivity_screen_gate_passed',checks=checks,seconds=time.perf_counter()-started)


if __name__=='__main__':main()
