"""H2ai bounded source-only chain geometry, clamp and literal energy gate."""
import itertools
import json
import signal
import time
import numpy as np
import study as S
import binary_coordinate as C
import graph_block_moves as B
from graph_block_study import PP,DEST,protocol


def main():
    started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120CPU-second gate')));signal.alarm(120);p=protocol();torch,_,_=S.imports();rows=[]
    fixture=np.array([[2,1,0],[1,1,3],[0,1,1]],np.int64);positions=np.argwhere(fixture==1);boards=[]
    for bits in itertools.product([0,1],repeat=len(positions)):
        b=fixture>=2
        for (r,c),bit in zip(positions,bits):b[r,c]=bit
        boards.append(b)
    for seed in range(3):
        m=S.model_for(seed,30,'original','cpu',torch.float64);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cpu',weights_only=True));rng=np.random.default_rng(2026111515+seed);a=rng.random((len(boards),5,3,3)).astype(np.float32);a/=a.sum(1,keepdims=True);t=torch.tensor(np.repeat(fixture[None],len(boards),axis=0));b=torch.tensor(np.stack(boards),dtype=torch.float64);aa=torch.tensor(a,dtype=torch.float64);cache=C.BinaryQuadratic(m,t,aa);items=[]
        for i,board in enumerate(boards):
            blocks,_=B.candidates(fixture,board)
            for cells in blocks:assert 2<=len(cells)<=256 and all(fixture.flat[k]==1 and board.flat[k] for k in cells);items.append((i,cells))
        values=B.scores(cache,b,items).numpy();error=0.
        for (i,cells),value in zip(items,values):
            z=b[i:i+1].clone();z.flatten()[list(cells)]=0;literal=float(m.energy(t[i:i+1],z,aa[i:i+1])-m.energy(t[i:i+1],b[i:i+1],aa[i:i+1]));error=max(error,abs(value-literal))
        assert error<1e-10;rows.append({'seed':seed,'source_binary_boards':32,'candidate_moves':len(items),'maximum_local_formula_vs_literal_full_energy_error':error})
    # Independent geometry invariants: after deleting protected junctions/clamps,
    # each eligible induced component has maximum degree2 and is a path or cycle.
    fixtures=[np.array([[2,1,1,1,3],[0,0,1,0,0],[0,0,1,0,0]]),np.array([[2,0,0,0,3],[0,1,1,0,0],[0,1,1,0,0]])]
    counts=[]
    for t in fixtures:
        b=t>0;blocks,_=B.candidates(t,b);assert all(all(t.flat[i]==1 for i in cells) for cells in blocks);counts.append([len(cells) for cells in blocks])
    assert counts==[[2],[4]],counts
    signal.alarm(0);S.save(DEST/'numerical-gate.json',{'status':'passed','protocol_sha256':S.sha(PP),'rows':rows,'explicit_branch_and_closed_cycle_sizes':counts,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'helper_sha256':S.sha(B.__file__),'new_independent_inputs':0,'scope':'All32binary source fixtures/model, every generated block delta checked against literal full original energy; allclamps protected. Explicit branch and disconnected freecycle geometry cases. Source-only, no target, no fitting.'})


if __name__=='__main__':main()
