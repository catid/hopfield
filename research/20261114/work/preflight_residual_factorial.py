"""H2bb source eligibility/terminal and original-cost gate; zero continuations."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import lexical_cycle_cut_moves as L
import terminal_cycle_moves as T
from verify_large_cycle_geometry import geometry
from local_factor_delta import deltas
from verify_training_feasibility import LD
from gate_factorial_transfer import literal_check
PARENT=S.OUT/'terminal_cycle';ORIGIN=S.OUT/'density_transfer';DEST=S.OUT/'residual_factorial_preflight'


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();started=time.perf_counter();p=json.loads((PARENT/'protocol.json').read_text());audit=json.loads((PARENT/'audit.json').read_text());assert audit['status']=='passed' and audit['joint_success']['terminal_then_extra16']==4086;rows=[];counts={'veto':0,'lexical':0};terms={'veto':0,'lexical':0};inputs={'veto':set(),'lexical':set()};checks=0;max_literal=max_local=0.
    with torch.no_grad():
        for seed in range(3):
            model=S.model_for(seed,180,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True));tokens=[];boards=[];readouts=[];families=[]
            for job in p['jobs']:
                if job['seed']!=seed:continue
                run=PARENT/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
                with np.load(run/'outputs.npz') as z:ix=np.flatnonzero(z['block_cap'][1]);assert z['finite'][1,ix].all() and not z['native_cap'][1,ix].any();boards.extend(z['predictions'][1,ix])
                with np.load(ORIGIN/'runs'/job['id']/'outputs.npz') as z:origin=z['block_cap'][0,ix];readouts.extend(z['couplings'][ix])
                with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:tokens.extend(z['tokens'][ix])
                for i,veto in zip(ix,origin):family='veto' if veto else 'lexical';families.append(family);counts[family]+=1;inputs[family].add(job['shard']*256+int(i))
            tok=np.stack(tokens);board=np.stack(boards);aa=np.stack(readouts);term,records=T.terminal_checks(tok,board,np.ones(len(tok),bool));items=[]
            for record,family in zip(records,families):
                i=record['local_input'];cells,stats=geometry(tok[i],board[i],256);assert record['geometry']==stats and record['candidates']==[list(c) for c in cells];terms[family]+=int(term[i]);items.extend((i,c) for c in cells)
            tt=torch.tensor(tok.astype(np.int64),device='cuda');bb=torch.tensor(board,device='cuda',dtype=torch.float64);a=torch.tensor(aa,device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,tt,a);assert cache.delta(bb).masked_fill(~cache.free,float('inf')).min()>0;values=L.scores(cache,bb,items).cpu().numpy();max_literal=max(max_literal,literal_check(model,tt,a,bb,items,values));theta=np.concatenate([model.T.cpu().numpy().ravel(),model.b.cpu().numpy(),model.kappa.cpu().numpy()]).astype(LD);local=deltas(tok,board,aa.astype(LD),theta,np.array(model.mu_offsets),np.array(model.aoffs),items);err=float(abs(local-values).max());assert err<1e-8;max_local=max(max_local,err);checks+=len(items);rows.append({'seed':seed,'residual_states':len(tok),'candidate_checks':len(items)})
    assert counts=={'veto':10,'lexical':15} and terms=={'veto':1,'lexical':0} and not(inputs['veto']&inputs['lexical']);prior=S.OUT/'terminal_cycle_preflight/numerical-gate.json';gate=json.loads(prior.read_text());assert gate['status']=='passed' and gate['helper_sha256']==S.sha(S.HERE/'terminal_cycle_moves.py');r={'status':'passed','new_source_trajectories':0,'parent_protocol_sha256':S.sha(PARENT/'protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'terminal_helper_sha256':S.sha(S.HERE/'terminal_cycle_moves.py'),'prior_terminal_gate_sha256':S.sha(prior),'script_sha256':S.sha(__file__),'inventory':inventory,'rows':rows,'source_caps_by_family':counts,'known_terminal_by_family':terms,'joint_input_ids_by_family':{k:sorted(v) for k,v in inputs.items()},'families_disjoint_by_input':True,'candidate_checks':checks,'maximum_gpu_literal_error':max_literal,'maximum_independent_local_error':max_local,'seconds':time.perf_counter()-started,'scope':'All25 source cap states independentgeometry/cost/native-stability. One veto terminal state known bookkeeping,24nonterminal futuresearch starts. Earlier5 terminal-at-round16 synthetic cases reusedbyhash because helper unchanged. No source continuations/freshinputs.'};assert r['seconds']<120;DEST.mkdir(exist_ok=True);S.save(DEST/'numerical-gate.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['rows','inventory']},indent=2))


if __name__=='__main__':main()
