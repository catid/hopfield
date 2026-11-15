"""No-trajectory H2ba residual cap lineage and remaining native/graph certificate."""
import json
import time
import numpy as np
import study as S
from verify_large_cycle_geometry import geometry
from audit_connectivity_screen import path_valid
from terminal_cycle import PP,DEST,PARENT


def main():
    started=time.perf_counter();p=json.loads(PP.read_text());audit=json.loads((DEST/'audit.json').read_text());assert audit['status']=='passed';rows=[]
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
        with np.load(run/'outputs.npz') as z:out={k:z[k] for k in ['predictions','block_cap','native_cap','stopped','finite','minimum_flip_margin64']}
        with np.load(PARENT/'runs'/job['id']/'outputs.npz') as z:source_veto=z['block_cap'][0]
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:t=z['tokens'];distance=z['shortest_edges']
        for i in np.flatnonzero(out['block_cap'][1]):
            board=out['predictions'][1,i];cells,stats=geometry(t[i],board,256);valid=path_valid(t[i],board);rows.append({'input_index':job['shard']*256+int(i),'seed':job['seed'],'inherited_veto_cap':bool(source_veto[i]),'valid_simple_path':bool(valid),'valid_shortest':bool(valid and board.sum()==distance[i]+1),'remaining_cycle256_candidates':len(cells),'cycle_rank':stats['cycle_rank'],'native_margin':float(out['minimum_flip_margin64'][1,i]),'finite':bool(out['finite'][1,i]),'native_cap':bool(out['native_cap'][1,i]),'stopped':bool(out['stopped'][1,i])})
    assert len(rows)==25 and all(r['finite'] and not r['native_cap'] and not r['stopped'] and r['native_margin']>=-1e-8 for r in rows);r={'status':'passed','kind':'posthoc_no_new_trajectories','protocol_sha256':S.sha(PP),'audit_sha256':S.sha(DEST/'audit.json'),'remaining_model_caps':25,'remaining_joint_inputs':len({r['input_index'] for r in rows}),'inherited_veto_model_caps':sum(r['inherited_veto_cap'] for r in rows),'new_lexical_model_caps':sum(not r['inherited_veto_cap'] for r in rows),'valid_shortest_caps':sum(r['valid_shortest'] for r in rows),'new_lexical_caps_with_candidates':sum(not r['inherited_veto_cap'] and r['remaining_cycle256_candidates']>0 for r in rows),'rows':rows,'source_sha256':S.sha(__file__),'seconds':time.perf_counter()-started};S.save(DEST/'residual-cap-diagnosis.json',r);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
