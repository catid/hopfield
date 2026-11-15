"""H2aw complete large-component costs, size boundaries and selector preflight."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import large_cycle_cut_moves as B
from verify_large_cycle_geometry import geometry,raw_chains,cycle_rank
from gate_factorial_transfer import literal_check
from verify_training_feasibility import energy,LD
PARENT=S.OUT/'cycle_transfer_180'
DEST=S.OUT/'large_cycle_cut_preflight'


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();started=time.perf_counter();p=json.loads((PARENT/'protocol.json').read_text());audit=json.loads((PARENT/'audit.json').read_text());manifest=json.loads((PARENT/'datasets.json').read_text());assert audit['status']=='passed';rows=[];checks=0;maximum=maximum_ld=0.;all_sizes=[]
    old=(S.HERE/'cycle_cut_moves.py').read_text().split('def scores(',1)[1];new=(S.HERE/'large_cycle_cut_moves.py').read_text().split('def scores(',1)[1];expected=old.replace('def solve(cache,initial,watchdog=None):','def solve(cache,initial,watchdog=None,selector="lexical"):').replace('j=min(ids,key=lambda j:(float(values[j]),items[j][1]))','j=min(ids,key=(lambda j:items[j][1]) if selector=="lexical" else (lambda j:(float(values[j]),items[j][1])))');assert new==expected
    with torch.no_grad():
        for seed in range(3):
            model=S.model_for(seed,180,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['old_binary']['path'],map_location='cuda',weights_only=True));tokens=[];boards=[];readouts=[];ids=[]
            for row in audit['rows']:
                job=row['job']
                if job['seed']!=seed:continue
                ix=np.flatnonzero(~np.array(row['success'][4],bool))
                if not len(ix):continue
                with np.load(S.ROOT/manifest['datasets'][job['shard']]['path']) as z:tokens.extend(z['tokens'][ix])
                with np.load(PARENT/'runs'/job['id']/'outputs.npz') as z:boards.extend(z['predictions'][4,ix]);readouts.extend(z['couplings'][ix])
                ids.extend(job['shard']*256+int(i) for i in ix)
            tok=np.stack(tokens);board=np.stack(boards);a=np.stack(readouts);items=[]
            for i in range(len(tok)):
                actual,stats=B.candidates(tok[i],board[i]);want,reference=geometry(tok[i],board[i],512);assert actual==want and stats==reference;assert not geometry(tok[i],board[i],256)[0];assert actual;items.extend((i,c) for c in actual);all_sizes.extend(len(c) for c in actual)
            tt=torch.tensor(tok.astype(np.int64),device='cuda');bb=torch.tensor(board,device='cuda',dtype=torch.float64);aa=torch.tensor(a,device='cuda',dtype=torch.float64);values=B.scores(C.BinaryQuadratic(model,tt,aa),bb,items).cpu().numpy();maximum=max(maximum,literal_check(model,tt,aa,bb,items,values));checks+=len(items);theta=np.concatenate([model.T.cpu().numpy().ravel(),model.b.cpu().numpy(),model.kappa.cpu().numpy()]).astype(LD);mu=np.array(model.mu_offsets);ao=np.array(model.aoffs)
            for j,(owner,cells) in enumerate(items):
                changed=board[owner:owner+1].copy();changed[0].flat[list(cells)]=False;literal=energy(tok[owner:owner+1],changed,a[owner:owner+1].astype(LD),theta,mu,ao)-energy(tok[owner:owner+1],board[owner:owner+1],a[owner:owner+1].astype(LD),theta,mu,ao);err=abs(float(literal[0])-float(values[j]));assert err<1e-8;maximum_ld=max(maximum_ld,err)
            boundary=[]
            for count in [1,2,256,257,512,513]:
                tok=np.ones((1,count+2),np.uint8);tok[0,0]=2;tok[0,-1]=3;b=tok>0;actual,stats=B.chain_candidates(tok,b);wanted,independent=raw_chains(tok,b,512);assert actual==wanted and stats==independent;assert len(actual)==int(2<=count<=512)
                if actual:
                    tt=torch.tensor(tok[None].astype(np.int64),device='cuda');bb=torch.tensor(b[None],device='cuda',dtype=torch.float64);aa=torch.full((1,5,*tok.shape),.2,device='cuda',dtype=torch.float64);items=[(0,actual[0])];v=B.scores(C.BinaryQuadratic(model,tt,aa),bb,items).cpu().numpy();maximum=max(maximum,literal_check(model,tt,aa,bb,items,v));checks+=1
                boundary.append({'free_chain_size':count,'raw_admitted':bool(actual),'protected_endpoints':True})
            rows.append({'seed':seed,'actual180_parent_cases':len(ids),'indices':ids,'actual_large_candidate_checks':len(values),'size_boundary_cases':boundary});assert time.perf_counter()-started<120
        tok=np.zeros((9,13),np.uint8);tok[8,:]=1;tok[8,0]=2;tok[8,12]=3
        for c in [2,9]:
            for x in [(7,c),(6,c),(5,c),(4,c),(3,c),(3,c+1),(4,c+1)]:tok[x]=1
        board=tok>0;wanted,_=geometry(tok,board,512);assert len(wanted)==2 and all(len(c)==3 for c in wanted);tt=torch.tensor(tok[None].astype(np.int64),device='cuda');bb=torch.tensor(board[None],device='cuda',dtype=torch.float64);aa=torch.full((1,5,*tok.shape),.2,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,tt,aa);cache.h.fill_(-1);cache.h[0].flatten()[list(wanted[0])]=-2
        for weight in cache.weights.values():weight.zero_()
        selector_rows=[]
        for selector,cells,cost in [('lexical',wanted[0],6.),('energy',wanted[1],3.)]:
            raw=B.solve(cache,bb,selector=selector);assert raw['block_cap'].tolist()==[True] and not raw['stopped'].any() and raw['block_moves'].tolist()==[16];assert np.array_equal(raw['prediction'][0].cpu().numpy(),board)
            for trace in raw['trace']:
                chosen=trace['candidates'][trace['chosen'][0]];assert chosen['cells']==list(cells) and chosen['delta']==cost
            selector_rows.append({'selector':selector,'chosen_cells':list(map(int,cells)),'positive_cost':cost,'native_restoration_and16round_cap_verified':True})
    assert sum(r['actual180_parent_cases'] for r in rows)==84 and min(all_sizes)==259 and max(all_sizes)==385;result={'status':'passed','kind':'preflight_not_scientific_study','parent_protocol_sha256':S.sha(PARENT/'protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'inventory':inventory,'rows':rows,'literal_candidate_checks':checks,'actual_large_candidate_sizes':[min(all_sizes),max(all_sizes)],'maximum_gpu_literal_error':maximum,'maximum_independent_longdouble_error':maximum_ld,'selector_synthetic_checks':selector_rows,'source_sha256':S.sha(__file__),'helper_sha256':S.sha(S.HERE/'large_cycle_cut_moves.py'),'verifier_sha256':S.sha(S.HERE/'verify_large_cycle_geometry.py'),'seconds':time.perf_counter()-started,'new_scientific_searches':0,'scope':'All84failedparent180modeloutputs/all168new259..385cellcandidates, exactrawcapboundaries1/2/256/257/512/513, originalliteralenergies inclindependentlongdouble, andbothselectorspositivechoice/native-capfixture. No new parent-cohort trajectories.'};assert result['seconds']<120;DEST.mkdir(exist_ok=True);S.save(DEST/'numerical-gate.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inventory']},indent=2))


if __name__=='__main__':main()
