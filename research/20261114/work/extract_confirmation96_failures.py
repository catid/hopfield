"""Publish every failed96 model state as an explicitly selected failure supplement."""
import json
import numpy as np
import study as S
import fast_npz as Z
from confirmation96_study import PP,DEST,ARMS,REPORTED
from verify_cycle_cut_geometry import cycle_rank

a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';ds=json.loads((DEST/'datasets.json').read_text());target=DEST/'failure-fixtures';target.mkdir(exist_ok=True);rows=[]
for row in a['rows']:
    failed=np.flatnonzero(~np.array(row['success'][7],bool))
    if not len(failed):continue
    job=row['job'];run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256'];data=ds['datasets'][job['shard']];assert S.sha(S.ROOT/data['path'])==data['sha256']
    with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','minimum_flip_margin64','energy64','iterations','stopped','finite','couplings','hidden_good','block_cap','block_native_cap','block_moves']}
    with np.load(S.ROOT/data['path']) as z:tokens=z['tokens'];labels=z['labels']
    theta=[]
    for arm in ['pl','old_binary','old_binary','old_binary']:
        with np.load(DEST/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
    for i in failed:
        arrays={k:old[k][list(REPORTED),i:i+1] for k in ['predictions','minimum_flip_margin64','energy64','iterations','stopped','finite']};arrays.update(tokens=tokens[i:i+1],labels=labels[i:i+1],couplings=old['couplings'][i:i+1],hidden_good=old['hidden_good'][i:i+1],theta=np.stack(theta),mu_offsets=mu,aoffs=ao,block_cap=np.concatenate([np.zeros((1,1),bool),old['block_cap'][[3,4,5],i:i+1]]),native_cap=np.concatenate([np.zeros((1,1),bool),old['block_native_cap'][[3,4,5],i:i+1]]));index=job['shard']*256+int(i);path=target/f'i{index}-s{job["seed"]}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_outputs_sha256':r['outputs_sha256'],'expected_success_by_arm':[row['success'][v][int(i)] for v in REPORTED],'cycle_ranks_by_arm':[cycle_rank(old['predictions'][v,i]) for v in REPORTED],'total_cycle_moves_by_arm':[0,int(old['block_moves'][1:4,i].sum()),int(old['block_moves'][1:5,i].sum()),int(old['block_moves'][1:6,i].sum())]})
S.save(target/'manifest.json',{'selection':'Every failedtotal96model state, no failuresexcluded; supplemental failureanalysis only, not an unbiasedsubset or additionalaccuracy sample.','arms':[ARMS[v] for v in REPORTED],'protocol_sha256':S.sha(PP),'source_audit_sha256':S.sha(DEST/'audit.json'),'source_sha256':S.sha(__file__),'fixtures':rows})
