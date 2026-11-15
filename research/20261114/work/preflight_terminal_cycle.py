"""H2ba source geometry/cost gate and explicit terminal-at-round-cap flag fixture."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import lexical_cycle_cut_moves as LEX
import terminal_cycle_moves as T
from verify_large_cycle_geometry import geometry
from local_factor_delta import deltas
from verify_training_feasibility import LD
from gate_factorial_transfer import literal_check
PARENT=S.OUT/'density_transfer';AUDIT=S.OUT/'local_factor_audit';DEST=S.OUT/'terminal_cycle_preflight'


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();started=time.perf_counter();p=json.loads((PARENT/'protocol.json').read_text());audit=json.loads((AUDIT/'audit.json').read_text());assert audit['status']=='passed';diagnosis=json.loads((PARENT/'residual-cap-diagnosis.json').read_text());assert diagnosis['model_cap_states']==204 and diagnosis['new_lexical_cap_states']==194;manifest=json.loads((PARENT/'datasets.json').read_text());rows=[];count=terminal_count=checks=0;max_literal=max_local=0.
    with torch.no_grad():
        for seed in range(3):
            model=S.model_for(seed,180,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['old_binary']['path'],map_location='cuda',weights_only=True));tokens=[];boards=[];readouts=[];source_indices=[]
            for job in p['jobs']:
                if job['seed']!=seed:continue
                run=PARENT/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256'];ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
                with np.load(run/'outputs.npz') as z:eligible=z['block_cap'][1]&~z['block_cap'][0];ix=np.flatnonzero(eligible);boards.extend(z['predictions'][3,ix]);readouts.extend(z['couplings'][ix])
                with np.load(S.ROOT/ds['path']) as z:tokens.extend(z['tokens'][ix])
                source_indices.extend(job['shard']*256+int(i) for i in ix)
            tok=np.stack(tokens);board=np.stack(boards);aa=np.stack(readouts);terminal,records=T.terminal_checks(tok,board,np.ones(len(tok),bool));items=[]
            for record in records:
                i=record['local_input'];cells,stats=geometry(tok[i],board[i],256);assert cells==[tuple(c) for c in record['candidates']] and record['geometry']==stats;assert terminal[i]==(len(cells)==0);items.extend((i,c) for c in cells)
            tt=torch.tensor(tok.astype(np.int64),device='cuda');bb=torch.tensor(board,device='cuda',dtype=torch.float64);a=torch.tensor(aa,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,tt,a);values=LEX.scores(cache,bb,items).cpu().numpy();max_literal=max(max_literal,literal_check(model,tt,a,bb,items,values));theta=np.concatenate([model.T.cpu().numpy().ravel(),model.b.cpu().numpy(),model.kappa.cpu().numpy()]).astype(LD);local=deltas(tok,board,aa.astype(LD),theta,np.array(model.mu_offsets),np.array(model.aoffs),items);err=float(abs(local-values).max());assert err<1e-8;max_local=max(max_local,err);assert cache.delta(bb).masked_fill(~cache.free,float('inf')).min()>0;count+=len(tok);terminal_count+=int(terminal.sum());checks+=len(items);rows.append({'seed':seed,'eligible_lexical_caps':len(tok),'terminal_without_new_move':int(terminal.sum()),'source_indices':source_indices,'candidate_costs':len(items)});assert time.perf_counter()-started<120
        # Final-state certification at a16-move cap: terminal, nonterminal,
        # native-failed, nonfinite, and negative-margin cases stay distinct.
        tok=np.zeros((5,9,13),np.uint8);tok[:,8,:]=1;tok[:,8,0]=2;tok[:,8,12]=3
        for x in [(7,2),(6,2),(5,2),(4,2),(3,2),(3,3),(4,3)]:tok[1][x]=1
        board=tok>0;raw={'prediction':torch.tensor(board,device='cuda',dtype=torch.float64),'stopped':np.zeros(5,bool),'finite':np.array([1,1,1,0,1],bool),'native_cap':np.array([0,0,1,0,0],bool),'block_cap':np.ones(5,bool),'minimum_flip_margin':np.array([1.,1.,1.,1.,-1.]),'block_moves':np.full(5,16),'native_sweeps':np.full(5,16)};before=raw['prediction'].clone();result=T.certify_final(tok,raw);assert torch.equal(before,result['prediction']);assert result['terminal_certified'].tolist()==[True,False,False,False,False];assert result['stopped'].tolist()==[True,False,False,False,False] and result['block_cap'].tolist()==[False,True,True,True,True];assert np.all(result['block_moves']==16) and np.all(result['native_sweeps']==16)
    assert count==194 and terminal_count==27;r={'status':'passed','kind':'source_geometry_cost_and_terminal_flag_gate_before_science_freeze','source_protocol_sha256':S.sha(PARENT/'protocol.json'),'source_audit_sha256':S.sha(AUDIT/'audit.json'),'diagnosis_sha256':S.sha(PARENT/'residual-cap-diagnosis.json'),'helper_sha256':S.sha(S.HERE/'terminal_cycle_moves.py'),'script_sha256':S.sha(__file__),'inventory':inventory,'rows':rows,'eligible_lexical_caps':count,'terminal_only_cases':terminal_count,'source_candidate_literal_checks':checks,'maximum_gpu_literal_error':max_literal,'maximum_independent_local_factor_error':max_local,'synthetic_terminal_at_round16_cases':5,'synthetic_board_iterations_and_move_counts_unchanged':True,'new_source_trajectories':0,'seconds':time.perf_counter()-started};DEST.mkdir(exist_ok=True);S.save(DEST/'numerical-gate.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['rows','inventory']},indent=2))


if __name__=='__main__':main()
