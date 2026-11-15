"""H2ao independent articulation geometry and literal boundary-energy gate."""
import itertools
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import pendant_block_moves as B
from verify_pendant_geometry import geometry
from verify_graph_block_source import connected
from gate_factorial_transfer import literal_check
from pendant_rescue import PP,DEST,PARENT,protocol


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();DEST.mkdir(exist_ok=True);started=time.perf_counter();fixture=np.array([[2,1,0],[1,1,3],[0,1,1]]);pos=np.argwhere(fixture==1);fixtures=[]
    for bits in itertools.product([0,1],repeat=len(pos)):
        b=fixture>=2
        for (r,c),bit in zip(pos,bits):b[r,c]=bit
        fixtures.append((fixture,b))
    rng=np.random.default_rng(2026111519);t=np.ones((5,5),np.uint8);t[0,0]=2;t[4,4]=3
    for i in range(128):b=(rng.random((5,5))<.65)|(t>=2);fixtures.append((t,b))
    raw_checks=0
    for t,b in fixtures:
        observed,stats=B.pendant_components(t,b);expected,independent=geometry(t,b);assert observed==expected and stats==independent;kept,_=B.candidates(t,b);want=[]
        for cells in expected:
            z=b.copy();z.flat[list(cells)]=False
            if connected(t,z):want.append(cells)
        assert kept==want;raw_checks+=len(expected)
    with np.load(S.ROOT/p['datasets'][0]['path']) as z:tokens=z['tokens'][:4]
    with np.load(PARENT/'runs/shard00-s0/outputs.npz') as z:fixture_boards=z['predictions'][3,:4].copy()
    fixture_boards[:,:16,:16]|=tokens[:,:16,:16]>0
    t=torch.tensor(tokens.astype(np.int64),device='cuda');boards=torch.tensor(fixture_boards,device='cuda',dtype=torch.float64);rows=[];maximum=0.;candidate_count=0;items=[]
    for i,b in enumerate(fixture_boards):
        raw,stats=B.pendant_components(tokens[i],b);expected,independent=geometry(tokens[i],b);assert raw==expected and stats==independent
        for cells in B.candidates(tokens[i],b)[0]:changed=b.copy();changed.flat[list(cells)]=False;assert connected(tokens[i],changed);items.append((i,cells))
    with torch.no_grad():
        for seed in range(3):
            model=S.model_for(seed,120,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
            with np.load(PARENT/f'runs/shard00-s{seed}/outputs.npz') as z:readout=z['couplings'][:4]
            a=torch.tensor(readout,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,t,a)
            if items:values=B.scores(cache,boards,items).cpu().numpy();error=literal_check(model,t,a,boards,items,values);maximum=max(maximum,error)
            small_t=torch.tensor(np.stack([x[0] for x in fixtures[:32]]).astype(np.int64),device='cuda');small_b=torch.tensor(np.stack([x[1] for x in fixtures[:32]]),device='cuda',dtype=torch.float64);rr=np.random.default_rng(2026111519+seed).random((32,5,3,3)).astype(np.float32);rr/=rr.sum(1,keepdims=True);small_a=torch.tensor(rr,device='cuda',dtype=torch.float64);small_items=[]
            for i,(tt,bb) in enumerate(fixtures[:32]):small_items.extend((i,cells) for cells in B.candidates(tt,bb)[0])
            assert small_items;small_values=B.scores(C.BinaryQuadratic(model,small_t,small_a),small_b,small_items).cpu().numpy();maximum=max(maximum,literal_check(model,small_t,small_a,small_b,small_items,small_values));candidate_count+=len(items)+len(small_items);rows.append({'seed':seed,'seen120_boards_with_fixed16corner_open_mask':4,'large_candidate_literal_checks':len(items),'small_candidate_literal_checks':len(small_items),'all_candidate_components_independently_reconstructed':True});assert time.perf_counter()-started<120
    result={'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'structural_fixtures':len(fixtures),'raw_fixture_candidates':raw_checks,'rows':rows,'literal_candidate_checks':candidate_count,'maximum_literal_error':maximum,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'scope':'No new sourcepolicy trajectories. Exhaustive5freecell plus128fixedrandomclamped graphmasks; first4retained120outputs with fixed16x16corner open mask exerciseactualouterboundaries/internalwalls, exactarticulationcomponents and fulloriginalenergydeletions. No labelselection.'};S.save(DEST/'numerical-gate.json',result);S.emit('pendant_numerical_gate_passed',seconds=result['seconds'],candidates=candidate_count)


if __name__=='__main__':main()
