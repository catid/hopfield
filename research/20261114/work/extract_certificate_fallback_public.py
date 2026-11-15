"""Portable full routing and all routed H2bj case evidence, no new inference."""
import json
import shutil
import numpy as np
import study as S
import fast_npz as Z
from certificate_fallback_study import PP, DEST, SOURCE

p=json.loads(PP.read_text());a=json.loads((DEST/'audit.json').read_text())
assert a['status']=='passed' and a['protocol_sha256']==S.sha(PP)
target=DEST/'public-fixtures';target.mkdir(exist_ok=True)
r=json.loads((DEST/'routing.json').read_text());fields={}
for row in r['jobs']:
    path=DEST/row['file'];assert S.sha(path)==row['sha256'];job=row['job']
    with np.load(path) as z:
        for k in z.files:
            if k not in fields:fields[k]=np.empty((3,4096),z[k].dtype)
            fields[k][job['seed'],job['shard']*256:(job['shard']+1)*256]=z[k]
Z.save(target/'routing-fields.npz',**fields)
shutil.copyfile(SOURCE/'audit.json',target/'source-audit.json')
fixtures=[]
mapping={'prediction':'predictions','iterations':'iterations','stopped':'stopped','finite':'finite','final_logits':'native_final_logits'}
for row in r['rows']:
    seed=row['model_seed'];i=row['batch_index'];job=next(j for j in p['source_jobs'] if j['id']==row['job_id'])
    batch=SOURCE/'runs'/row['job_id']/f'batch{row["batch"]:02d}.npz';assert S.sha(batch)==job['batch_sha256'][batch.name]
    with np.load(batch) as z:
        ref={k:z[v][0] for k,v in mapping.items()}
        context={k:z[k][7,i:i+1] for k in ['predictions','iterations','stopped','finite','minimum_flip_margin64','energy64']}
        context.update(block_cap=z['block_cap'][5,i:i+1],native_cap=z['block_native_cap'][5,i:i+1])
    Z.save(target/f'source-bp-batch-seed{seed}.npz',**ref)
    ds=p['datasets'][row['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:
        context.update(tokens=z['tokens'][row['local_index']:row['local_index']+1],labels=z['labels'][row['local_index']:row['local_index']+1])
    theta=[]
    for arm in ['pl','old_binary']:
        with np.load(SOURCE/f'model-seed{seed}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
    context.update(theta=np.stack(theta),mu_offsets=mu,aoffs=ao)
    Z.save(target/f'context-seed{seed}.npz',**context)
    for folder,prefix in [('gates','gate'),('fallback','replay'),('diagnostics','diagnostics')]:
        name=f'seed{seed}' if folder=='gates' else f'i{row["input_index"]}-s{seed}'
        shutil.copyfile(DEST/folder/f'{name}.npz',target/f'{prefix}-seed{seed}.npz')
        if folder!='diagnostics':shutil.copyfile(DEST/folder/f'{name}.json',target/f'{prefix}-seed{seed}.json')
    fixtures.append({**row,'source_batch_sha256':S.sha(batch),'source_outputs_sha256':job['outputs_sha256']})
S.save(target/'manifest.json',{'protocol_sha256':S.sha(PP),'audit_sha256':S.sha(DEST/'audit.json'),
    'source_audit_sha256':S.sha(SOURCE/'audit.json'),'model_inputs':12288,'routed_model_inputs':3,
    'fixtures':fixtures,'files':{q.name:S.sha(q) for q in sorted(target.iterdir()) if q.is_file() and q.name!='manifest.json'},
    'extractor_sha256':S.sha(__file__),
    'scope':'All numerical routing fields, all three routed source and replay cases, and96 source-batch parity states. Full original source archives were hash-verified on Work. Unchanged accepted-state correctness is inherited from the complete published H2bh audit, not re-estimated from a subset.'})
