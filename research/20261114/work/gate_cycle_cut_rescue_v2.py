"""H2ar old-energy costs, cycle-rank admission and positive-acceptance gate."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import cycle_cut_moves as B
from verify_cycle_cut_geometry_v2 import geometry,cycle_rank
from gate_factorial_transfer import literal_check
from pendant_gate_fixtures import motifs
from cycle_cut_rescue_v2 import PP,DEST,PARENT,protocol


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();started=time.perf_counter();declared=motifs();rng=np.random.default_rng(2026111519);t=np.ones((5,5),np.uint8);t[0,0]=2;t[4,4]=3;random=[]
    for i in range(128):b=(rng.random((5,5))<.65)|(t>=2);random.append((t,b))
    with np.load(S.ROOT/p['datasets'][0]['path']) as z:tokens=z['tokens'][:4]
    with np.load(PARENT/'runs/shard00-s0/outputs.npz') as z:large=z['predictions'][3,:4].copy()
    large[:,:16,:16]|=tokens[:,:16,:16]>0;groups=[('random5',random),('explicit9',[(r['tokens'],r['board']) for r in declared]),('seen120',list(zip(tokens,large)))];prepared=[];structural=0
    for name,group in groups:
        items=[]
        for i,(tok,board) in enumerate(group):
            actual,stats=B.candidates(tok,board);expected,independent=geometry(tok,board);assert actual==expected and stats==independent;assert B.cycle_rank(board)==cycle_rank(board);structural+=1;items.extend((i,cells) for cells in actual)
        prepared.append((name,np.stack([x[0] for x in group]),np.stack([x[1] for x in group]),items))
    rows=[];maximum=0.;candidate_count=0
    with torch.no_grad():
        for seed in range(3):
            model=S.model_for(seed,120,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
            for name,tok,board,items in prepared:
                if name=='seen120':
                    with np.load(PARENT/f'runs/shard00-s{seed}/outputs.npz') as z:readout=z['couplings'][:4]
                else:readout=np.random.default_rng(2026111519+seed).random((len(tok),5,*tok.shape[-2:])).astype(np.float32);readout/=readout.sum(1,keepdims=True)
                tt=torch.tensor(tok.astype(np.int64),device='cuda');bb=torch.tensor(board,device='cuda',dtype=torch.float64);aa=torch.tensor(readout,device='cuda',dtype=torch.float64)
                if items:values=B.scores(C.BinaryQuadratic(model,tt,aa),bb,items).cpu().numpy();err=literal_check(model,tt,aa,bb,items,values);maximum=max(maximum,err)
                else:err=0.
                candidate_count+=len(items);rows.append({'seed':seed,'fixture_group':name,'literal_candidate_checks':len(items),'maximum_error':err});assert time.perf_counter()-started<120
        positive=[]
        for r in declared:
            if not r['name'].startswith('cycle'):continue
            tt=torch.tensor(r['tokens'][None].astype(np.int64),device='cuda');bb=torch.tensor(r['board'][None],device='cuda',dtype=torch.float64);aa=torch.full((1,5,9,9),.2,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,tt,aa);cache.h.fill_(-1)
            for weight in cache.weights.values():weight.zero_()
            raw=B.solve(cache,bb);assert raw['block_cap'].tolist()==[True] and raw['stopped'].tolist()==[False] and raw['block_moves'].tolist()==[16] and raw['finite'].all();assert np.array_equal(raw['prediction'][0].cpu().numpy(),r['board']);assert len(raw['trace'])==16
            for record,before in zip(raw['trace'],raw['states_before']):
                assert len(record['chosen'])==1;c=record['candidates'][record['chosen'][0]];assert c['delta']==3 and len(c['cells'])==3;changed=before[0].copy();changed.flat[c['cells']]=False;assert cycle_rank(before[0])==1 and cycle_rank(changed)==0;assert np.array_equal(before[0],r['board']);assert record['native_results'][0]['stopped']
            positive.append({'fixture':r['name'],'delta':3.,'immediate_rank_change':[1,0],'native_restores_all3deleted_vertices_and_cycle':True,'positive_moves':16,'block_cap':True,'reported_success':False})
    assert candidate_count>0 and len(positive)==4;result={'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'structural_fixtures':structural,'rows':rows,'literal_candidate_checks':candidate_count,'maximum_literal_error':maximum,'synthetic_positive_acceptance':{'energy':'Known E=-sumfreep withzeroquadraticweights;gateonly,notoriginalcoefficients','rows':positive,'new_scientific_search_inputs':0},'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'scope':'Oldcoefficient costs+independent cycleadmission;syntheticpositivegate actuallyexecutesnewacceptanceandrecordsrestoration/caps;no sourcecohortintervention beforegate.'};assert result['seconds']<120;S.save(DEST/'numerical-gate.json',result);S.emit('cycle_cut_gate_passed',seconds=result['seconds'],candidate_checks=candidate_count)


if __name__=='__main__':main()
