"""H2ap nonempty hand-enumerated motifs and original-energy boundary gate."""
import itertools
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import pendant_block_moves as B
import connected_block_moves as CHAIN
from verify_pendant_geometry import geometry
from verify_graph_block_source import connected
from gate_factorial_transfer import literal_check
from pendant_rescue_v2 import PP,DEST,PARENT,protocol
from pendant_gate_fixtures import motifs


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();started=time.perf_counter();small=np.array([[2,1,0],[1,1,3],[0,1,1]],np.uint8);pos=np.argwhere(small==1);fixtures=[]
    for bits in itertools.product([0,1],repeat=len(pos)):
        b=small>=2
        for (r,c),bit in zip(pos,bits):b[r,c]=bit
        fixtures.append((small,b))
    rng=np.random.default_rng(2026111519);t=np.ones((5,5),np.uint8);t[0,0]=2;t[4,4]=3
    for i in range(128):b=(rng.random((5,5))<.65)|(t>=2);fixtures.append((t,b))
    declared=motifs();fixtures.extend((r['tokens'],r['board']) for r in declared);raw_checks=0;motif_rows=[]
    for t,b in fixtures:
        observed,stats=B.pendant_components(t,b);expected,independent=geometry(t,b);assert observed==expected and stats==independent;kept,_=B.candidates(t,b);want=[]
        for cells in expected:
            z=b.copy();z.flat[list(cells)]=False
            if connected(t,z):want.append(cells)
        assert kept==want;raw_checks+=len(expected)
    for r in declared:
        raw,stats=B.pendant_components(r['tokens'],r['board']);assert stats['witnesses']==r['expected_witnesses'] and len(raw)==r['expected_candidate_count'] and stats['duplicate_component_candidates']==r['expected_duplicate_count'];assert B.candidates(r['tokens'],r['board'])[0]==raw
        chain=CHAIN.candidates(r['tokens'],r['board'])[0];new=[c for c in raw if c not in chain];motif_rows.append({'name':r['name'],'expected_witnesses':r['expected_witnesses'],'candidate_count':len(raw),'old_chain_candidates':[list(c) for c in chain],'new_family_only_candidates':[list(c) for c in new],'duplicate_count':stats['duplicate_component_candidates']})
        if r['name'].startswith(('fork','cycle')):assert new
    with np.load(S.ROOT/p['datasets'][0]['path']) as z:tokens=z['tokens'][:4]
    with np.load(PARENT/'runs/shard00-s0/outputs.npz') as z:large=z['predictions'][3,:4].copy()
    large[:,:16,:16]|=tokens[:,:16,:16]>0;groups=[('random5',fixtures[32:160]),('explicit9',fixtures[160:]),('seen120',list(zip(tokens,large)))];rows=[];maximum=0.;candidate_count=0
    # Reconstruct geometry once before scoring any model; never use shortest labels.
    prepared=[]
    for name,group in groups:
        items=[]
        for i,(tok,board) in enumerate(group):
            raw,stats=B.pendant_components(tok,board);expected,independent=geometry(tok,board);assert raw==expected and stats==independent
            items.extend((i,cells) for cells in B.candidates(tok,board)[0])
        assert items,name
        prepared.append((name,np.stack([x[0] for x in group]),np.stack([x[1] for x in group]),items))
    with torch.no_grad():
        for seed in range(3):
            model=S.model_for(seed,120,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
            for name,tok,board,items in prepared:
                if name=='seen120':
                    with np.load(PARENT/f'runs/shard00-s{seed}/outputs.npz') as z:readout=z['couplings'][:4]
                else:
                    readout=np.random.default_rng(2026111519+seed).random((len(tok),5,*tok.shape[-2:])).astype(np.float32);readout/=readout.sum(1,keepdims=True)
                tt=torch.tensor(tok.astype(np.int64),device='cuda');bb=torch.tensor(board,device='cuda',dtype=torch.float64);aa=torch.tensor(readout,device='cuda',dtype=torch.float64);values=B.scores(C.BinaryQuadratic(model,tt,aa),bb,items).cpu().numpy();err=literal_check(model,tt,aa,bb,items,values);maximum=max(maximum,err);candidate_count+=len(items);rows.append({'seed':seed,'fixture_group':name,'literal_candidate_checks':len(items),'maximum_error':err});assert time.perf_counter()-started<120
        # Known E=sum(h*p): two identical length2 pendants have exact delta=-2.
        r=next(r for r in declared if r['name']=='multiple-rotation0');tt=torch.tensor(r['tokens'][None].astype(np.int64),device='cuda');bb=torch.tensor(r['board'][None],device='cuda',dtype=torch.float64);aa=torch.full((1,5,9,9),.2,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,tt,aa);cache.h.fill_(-1);cache.h[:,0:8,:]=1
        for weight in cache.weights.values():weight.zero_()
        raw=B.solve(cache,bb);first=raw['trace'][0];assert len(first['candidates'])==2 and all(c['delta']==-2 for c in first['candidates']);assert first['chosen']==[0] and first['candidates'][0]['cells']==r['expected_witnesses'][0]['cells'];assert raw['stopped'].all() and np.array_equal(raw['prediction'][0].cpu().numpy(),np.indices((9,9))[0]==8)
        tie={'energy':'Synthetic E=sum(h*p),h=+1abovebackbone/-1onbackbone,zeroquadraticweights','candidate_deltas':[-2.,-2.],'chosen_cells':first['candidates'][0]['cells'],'exact_lexicographic_tie_passed':True,'original_coefficients':False,'new_scientific_input_searches':0}
    result={'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'structural_fixtures':len(fixtures),'raw_fixture_candidates':raw_checks,'motifs':motif_rows,'rows':rows,'literal_candidate_checks':candidate_count,'maximum_literal_error':maximum,'synthetic_tie_gate':tie,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'scope':'Predeclaredgeometry+originalcoefficient literalenergy gate; synthetic known-energy tie test separately labeled;no H2an intervention trajectories or new scientificinputs.'};assert result['seconds']<120;S.save(DEST/'numerical-gate.json',result);S.emit('pendant_v2_numerical_gate_passed',seconds=result['seconds'],candidates=candidate_count)


if __name__=='__main__':main()
