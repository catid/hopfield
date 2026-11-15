"""H2aw preflight: all old lexical256 states have no allowed continuation."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import study as S
from verify_training_feasibility import flip,LD
from verify_large_cycle_geometry import geometry
from audit_connectivity_screen import path_valid
PARENT=S.OUT/'cycle_transfer_180'
DEST=S.OUT/'large_cycle_cut_preflight'


def one(args):
    job,manifest,models=args;run=PARENT/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert r['protocol_sha256']==S.sha(PARENT/'protocol.json') and S.sha(run/'outputs.npz')==r['outputs_sha256'];ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:t=z['tokens']
    with np.load(run/'outputs.npz') as z:b=z['predictions'][4];a=z['couplings'].astype(LD);stored=z['minimum_flip_margin64'][4];assert z['finite'][4].all() and z['stopped'][4].all() and not z['block_cap'][2].any() and not z['block_native_cap'][2].any()
    model=next(x for x in models['records'] if x['seed']==job['seed'] and x['arm']=='old_binary');path=PARENT/model['path'];assert S.sha(path)==model['sha256']
    with np.load(path) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    margins=np.where(t==1,flip(t,b,a,theta,mu,ao),np.inf).reshape(256,-1).min(1);assert margins.min()>=-1e-8;error=float(abs(margins-stored).max());assert error<1e-9;simple=np.array([path_valid(tok,board) for tok,board in zip(t,b)]);candidates=0
    for tok,board,is_simple in zip(t,b,simple):
        # Simple paths are exact identity by the declared old policy.
        if is_simple:continue
        cells,stats=geometry(tok,board,256);candidates+=len(cells);assert not cells
    return {'job':job['id'],'source_outputs_sha256':r['outputs_sha256'],'simple_identity_cases':int(simple.sum()),'nonsimple_terminal_cases':int((~simple).sum()),'admitted_old256_candidates':candidates,'minimum_native_margin':float(margins.min()),'maximum_reported_margin_error':error,'no_caps_nonfinite_or_unstopped':True}


def main():
    started=time.perf_counter();p=json.loads((PARENT/'protocol.json').read_text());a=json.loads((PARENT/'audit.json').read_text());assert a['status']=='passed';manifest=json.loads((PARENT/'datasets.json').read_text());models=json.loads((PARENT/'model-extraction.json').read_text())
    with ProcessPoolExecutor(max_workers=16) as pool:rows=list(pool.map(one,[(j,manifest,models) for j in p['jobs']]))
    r={'status':'passed','kind':'preflight_not_scientific_study','parent_protocol_sha256':S.sha(PARENT/'protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'source_sha256':S.sha(__file__),'geometry_verifier_sha256':S.sha(S.HERE/'verify_large_cycle_geometry.py'),'model_inputs':12288,'simple_identity_cases':sum(x['simple_identity_cases'] for x in rows),'nonsimple_terminal_cases':sum(x['nonsimple_terminal_cases'] for x in rows),'admitted_old256_candidates':sum(x['admitted_old256_candidates'] for x in rows),'minimum_native_margin':min(x['minimum_native_margin'] for x in rows),'maximum_reported_margin_error':max(x['maximum_reported_margin_error'] for x in rows),'rows':rows,'seconds':time.perf_counter()-started,'new_searches':0,'scope':'Alloldlexical256prefixes: independently recomputed native margins/no caps or nonfinite; simplepathidentity and every remaining oldcandidate set empty. More unchanged-policy rounds cannot move any output. No new trajectories.'};assert r['seconds']<120;DEST.mkdir(exist_ok=True);S.save(DEST/'old256-fixed-point-control.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))


if __name__=='__main__':main()
