"""Complete selector-change preflight; original cost/geometry gate reused by hash."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import cycle_cut_moves as ENERGY
import lexical_cycle_cut_moves as LEX
from verify_cycle_cut_geometry_checked import geometry,cycle_rank
from cycle_cut_rescue_checked import PP,protocol


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();started=time.perf_counter();a=(S.HERE/'cycle_cut_moves.py').read_text().split('import numpy',1)[1];b=(S.HERE/'lexical_cycle_cut_moves.py').read_text().split('import numpy',1)[1];assert b==a.replace('j=min(ids,key=lambda j:(float(values[j]),items[j][1]))','j=min(ids,key=lambda j:items[j][1])')
    prior=p['checked_source_preflight'];assert S.sha(S.ROOT/prior['path'])==prior['sha256'];r=json.loads((S.ROOT/prior['path']).read_text());assert r['status']=='passed';model=S.model_for(0,120,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models']['0']['path'],map_location='cuda',weights_only=True));t=np.zeros((9,13),np.uint8);t[8,:]=1;t[8,0]=2;t[8,12]=3
    for c in [2,9]:
        for x in [(7,c),(6,c),(5,c),(4,c),(3,c),(3,c+1),(4,c+1)]:t[x]=1
    rows=[]
    with torch.no_grad():
        for rotation in range(4):
            tok=np.rot90(t,rotation).copy();board=tok>0;expected,stats=geometry(tok,board);assert len(expected)==2 and all(len(c)==3 for c in expected);assert LEX.candidates(tok,board)==(expected,stats);tt=torch.tensor(tok[None].astype(np.int64),device='cuda');bb=torch.tensor(board[None],device='cuda',dtype=torch.float64);aa=torch.full((1,5,*tok.shape),.2,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,tt,aa);cache.h.fill_(-1);cache.h[0].flatten()[list(expected[0])]=-2
            for w in cache.weights.values():w.zero_()
            outcomes=[]
            for name,helper,wanted,cost in [('energy',ENERGY,expected[1],3.),('lexical',LEX,expected[0],6.)]:
                raw=helper.solve(cache,bb);assert raw['block_cap'].tolist()==[True] and not raw['stopped'].any() and raw['block_moves'].tolist()==[16];assert np.array_equal(raw['prediction'][0].cpu().numpy(),board)
                for trace,states in zip(raw['trace'],raw['states_before']):
                    assert len(trace['chosen'])==1;chosen=trace['candidates'][trace['chosen'][0]];assert chosen['cells']==list(wanted) and chosen['delta']==cost;changed=states[0].copy();changed.flat[list(wanted)]=False;assert cycle_rank(states[0])==2 and cycle_rank(changed)==1
                outcomes.append({'selector':name,'chosen_cells':list(map(int,wanted)),'known_original_delta':cost,'moves':16,'native_restores_cycle':True,'cap_retained_failure':True})
            rows.append({'rotation':rotation,'outcomes':outcomes})
    out={'status':'passed','kind':'preflight_not_scientific_study','reference_protocol_sha256':S.sha(PP),'original_cost_geometry_preflight':prior,'selector_only_source_change_verified':True,'source_sha256':S.sha(__file__),'lexical_helper_sha256':S.sha(S.HERE/'lexical_cycle_cut_moves.py'),'rows':rows,'seconds':time.perf_counter()-started,'inventory':inventory,'scope':'Same complete previously verified original cost/geometry gate hash-reused; four two-cycle known-energy fixtures force lexical +6 versus energy +3, including16native-restoration/cap rounds. No cohort outcomes or new scientific searches.'};assert out['seconds']<120;dest=S.OUT/'lexical_cycle_cut_preflight';dest.mkdir(exist_ok=True);S.save(dest/'numerical-gate.json',out);S.emit('lexical_selector_preflight_passed',seconds=out['seconds'])


if __name__=='__main__':main()
