"""Posthoc no-trajectory decomposition of verified H2ay block-cap failures."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_large_cycle_geometry import geometry,raw_chains
from verify_graph_block_source import connected
from verify_training_feasibility import flip,LD
from local_factor_delta import deltas
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'artifacts/research/20261114/work/density_transfer';A=ROOT/'artifacts/research/20261114/work/local_factor_audit'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def one(job):
    run=D/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert sha(run/'outputs.npz')==receipt['outputs_sha256'];datasets=json.loads((D/'datasets.json').read_text())['datasets'];ds=ROOT/datasets[job['shard']]['path'];assert sha(ds)==datasets[job['shard']]['sha256']
    with np.load(ds) as z:t=z['tokens'];distance=z['shortest_edges']
    with np.load(run/'outputs.npz') as z:out={k:z[k] for k in ['predictions','stopped','finite','block_cap','block_native_cap','hidden_good','couplings','minimum_flip_margin64']}
    ix=np.flatnonzero(~out['stopped'][3]);rows=[]
    with np.load(D/f'model-seed{job["seed"]}-old_binary.npz') as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    if len(ix):actual=flip(t[ix],out['predictions'][3,ix],out['couplings'][ix].astype(LD),theta,mu,ao);margin=np.where(t[ix]==1,actual,np.inf).reshape(len(ix),-1).min(1);assert abs(margin-out['minimum_flip_margin64'][3,ix]).max()<1e-9
    for j,i in enumerate(ix):
        tok=t[i];b=out['predictions'][3,i];assert out['block_cap'][1,i] and out['finite'][3,i] and not out['block_native_cap'][1,i] and out['hidden_good'][i];valid=path_valid(tok,b);shortest=valid and b.sum()==distance[i]+1;c256,_=geometry(tok,b,256);c512,_=geometry(tok,b,512);items=[(0,c) for c in c256];values=deltas(tok[None],b[None],out['couplings'][i:i+1].astype(LD),theta,mu,ao,items) if items else np.array([]);raw,_=raw_chains(tok,b,256);veto=[]
        for cells in raw:
            changed=b.copy();changed.flat[list(cells)]=False
            if connected(tok,changed):veto.append((0,cells))
        vv=deltas(tok[None],b[None],out['couplings'][i:i+1].astype(LD),theta,mu,ao,veto) if veto else np.array([]);rows.append({'input_index':int(job['shard']*256+i),'seed':job['seed'],'inherited_veto_cap':bool(out['block_cap'][0,i]),'simple_path':bool(valid),'valid_shortest':bool(shortest),'native_margin':float(margin[j]),'cycle256_candidates':len(c256),'cycle512_candidates':len(c512),'cycle256_minimum_cost':float(values.min()) if len(values) else None,'negative_veto256_candidates':int((vv<-1e-8).sum())})
    return {'job':job,'source_outputs_sha256':receipt['outputs_sha256'],'rows':rows}


def main():
    started=time.perf_counter();audit=json.loads((A/'audit.json').read_text());assert audit['status']=='passed' and audit['hypothesis']=='H2ay';p=json.loads((D/'protocol.json').read_text())
    with ProcessPoolExecutor(max_workers=8) as pool:jobs=list(pool.map(one,p['jobs']))
    rows=[r for job in jobs for r in job['rows']];assert len(rows)==204;result={'status':'passed','kind':'posthoc_source_diagnosis_zero_trajectories','source_audit_sha256':sha(A/'audit.json'),'protocol_sha256':sha(D/'protocol.json'),'model_cap_states':len(rows),'joint_inputs':len({r['input_index'] for r in rows}),'inherited_veto_cap_states':sum(r['inherited_veto_cap'] for r in rows),'new_lexical_cap_states':sum(not r['inherited_veto_cap'] for r in rows),'simple_path_states':sum(r['simple_path'] for r in rows),'valid_shortest_states':sum(r['valid_shortest'] for r in rows),'no_cycle256_candidate_states':sum(r['cycle256_candidates']==0 for r in rows),'remaining_cycle256_candidate_states':sum(r['cycle256_candidates']>0 for r in rows),'remaining_cycle512_candidate_states':sum(r['cycle512_candidates']>0 for r in rows),'minimum_native_margin':min(r['native_margin'] for r in rows),'rows':jobs,'workers':8,'seconds':time.perf_counter()-started,'source_sha256':sha(Path(__file__)),'scope':'All204 inherited/new lexical256 block-cap states. Independent native margins, path/shortest status and old/new-cap candidate geometry/costs only. Distinguish already-terminal-at-cap from still-cyclic and upstream veto cap. No certification flags changed; no continuation or fresh input. No claim that further rounds rescue the remaining states.'};assert result['seconds']<60;(D/'residual-cap-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
