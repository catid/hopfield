"""H2bg explicit-limit geometry and archived-prefix parity gates."""
import argparse
import json
import time
import numpy as np
import study as S
from continuation_study import DEST,SOURCE,PP,protocol


def synthetic():
    from preflight_veto_geometry import fixtures
    from verify_large_cycle_geometry import geometry
    from continuation_geometry import candidates
    started=time.perf_counter();counts={};checks=0
    for family,t,b in fixtures():
        for limit in [256,512]:assert candidates(t,b,limit)==geometry(t,b,limit),(family,limit);checks+=1
        counts[family]=counts.get(family,0)+1
    assert time.perf_counter()-started<120
    S.save(DEST/'synthetic-gate.json',{'status':'passed','counts':counts,'geometry_checks':checks,'limits':[256,512],'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['continuation_geometry.py','bridge_cycle_geometry.py','verify_large_cycle_geometry.py','preflight_continuation.py']},'new_trajectories':0})


def source(gpu):
    import selective_fallback as F
    import binary_coordinate as C
    import continuation_moves as M
    from loop80_transfer import load_models,protocol as source_protocol
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter();data=json.loads((SOURCE/'datasets.json').read_text())['datasets'][0];assert S.sha(S.ROOT/data['path'])==data['sha256']
    with np.load(S.ROOT/data['path']) as z:tokens=z['tokens']
    rows=[]
    def watch():assert time.perf_counter()-started<120,'Fixed source gate budget'
    with torch.no_grad():
        for seed in ([0,2] if gpu==0 else [1]):
            _,models=load_models(source_protocol(),seed);run=SOURCE/'runs'/f'shard00-s{seed}';r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256']
            with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','couplings','block_moves','post_block_sweeps','block_native_cap','block_cap','stopped','finite']}
            cases=rounds=0
            for lo in range(0,256,32):
                item=next(q for q in r['trace_archives'] if q['offset']==lo and q['arm']=='unified48');path=run/item['path'];assert S.sha(path)==item['sha256']
                with np.load(path) as z:ix=z['valid_prefix_indices'];states=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());prestop=z['pre_terminal_stopped'];precap=z['pre_terminal_block_cap']
                if not len(ix):continue
                glob=lo+ix;t=torch.tensor(tokens[glob].astype(np.int64),device='cuda');a=torch.tensor(old['couplings'][glob],device='cuda').double();cache=C.BinaryQuadratic(models[1],t,a);initial=torch.tensor(old['predictions'][4,glob],device='cuda').double();raw=M.certify_final(tokens[glob],M.solve(cache,initial,watch,max_rounds=32,max_cells=256),256)
                assert np.array_equal(raw['prediction'].cpu().numpy(),old['predictions'][5,glob]);assert payload['trace']==raw['trace'] and payload['final_terminal_checks']==raw['terminal_check_records'];assert np.array_equal(states,raw['states_before']) and np.array_equal(prestop,raw['pre_terminal_stopped']) and np.array_equal(precap,raw['pre_terminal_block_cap'])
                for key,oldkey in [('block_moves','block_moves'),('native_sweeps','post_block_sweeps'),('native_cap','block_native_cap'),('block_cap','block_cap')]:assert np.array_equal(raw[key],old[oldkey][3,glob])
                for key in ['stopped','finite']:assert np.array_equal(raw[key],old[key][5,glob])
                cases+=len(ix);rounds+=len(raw['trace']);watch()
            rows.append({'seed':seed,'source_inputs':256,'eligible_archived_states':cases,'trace_rounds':rounds,'all_boards_flags_choices_traces_exact':True,'source_outputs_sha256':r['outputs_sha256']})
    S.save(DEST/f'source-gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(PP),'gpu':gpu,'inventory':inventory,'seconds':time.perf_counter()-started,'rows':rows,'scope':'Exact replay of already archived unified16-to48 prefixes. Zero new scientific trajectories or accuracy samples.'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['synthetic','source']);ap.add_argument('--gpu',type=int,choices=[0,1]);a=ap.parse_args();source(a.gpu) if a.command=='source' else synthetic()
