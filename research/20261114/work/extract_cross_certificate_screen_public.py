"""Portable full scalar seal and complete H2bl negative/retention-loss fixtures."""
import csv
import json
import shutil
import numpy as np
import study as S
import fast_npz as Z
from cross_certificate_screen_study import PP,DEST,SOURCE


def main():
    p=json.loads(PP.read_text());a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed'
    with (DEST/'per-model-input.csv').open() as stream:table=list(csv.DictReader(stream))
    bycase={(int(r['input_index']),int(r['model_seed'])):r for r in table}
    target=DEST/'public-fixtures';target.mkdir(exist_ok=True);fixed=[];supp=[];scalar=[]
    shutil.copyfile(SOURCE/'audit.json',target/'source-audit.json')
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];path=run/'margins.npz';shutil.copyfile(path,target/f'{job["id"]}-margins.npz');scalar.append({'job':job,'file':f'{job["id"]}-margins.npz','sha256':S.sha(path)})
        with np.load(path) as z:measured={k:z[k] for k in z.files}
        source=SOURCE/'runs'/job['id']/'outputs.npz';assert S.sha(source)==job['outputs_sha256']
        with np.load(source) as z:boards=z['predictions'][0];couplings=z['couplings']
        ds=p['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens'];labels=z['labels']
        theta=[]
        for arm in ['pl','old_binary']:
            with np.load(SOURCE/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
        selected=[i for i in range(256) if bycase[job['shard']*256+i,job['seed']]['graph_valid_shortest']=='0' or bycase[job['shard']*256+i,job['seed']]['dual_accept']=='0']
        for category,indices,rows in [('fixed',[0],fixed),('negative-or-loss',selected,supp)]:
            for i in indices:
                index=job['shard']*256+i;arrays={k:v[i:i+1] for k,v in measured.items()};arrays.update(tokens=tokens[i:i+1],labels=labels[i:i+1],board=boards[i:i+1],couplings=couplings[i:i+1],theta=np.stack(theta),mu_offsets=mu,aoffs=ao)
                path=target/f'{category}-i{index}-s{job["seed"]}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_outputs_sha256':job['outputs_sha256'],'scalar_file':f'{job["id"]}-margins.npz','local_index':i})
    S.save(target/'manifest.json',{'protocol_sha256':S.sha(PP),'audit_sha256':S.sha(DEST/'audit.json'),'source_audit_sha256':S.sha(SOURCE/'audit.json'),
        'scalar_archives':scalar,'fixed_cases':fixed,'negative_or_retention_loss_cases':supp,
        'fixed_selection':'First input in every source shard and model, independent of outcome.',
        'supplement_selection':'Every BP graph-shortest failure and every rejected graph-shortest success; include successful rejections as well as false acceptances. Explicit source outcome-selected supplement.',
        'extractor_sha256':S.sha(__file__),'scope':'All scalar margins/flags, fixed48 cases and complete negative/retention-loss supplement. All underlying source states and both objective margins were independently checked on Work.'})
    print(json.dumps({'fixed_cases':len(fixed),'negative_or_loss_cases':len(supp),'scalar_archives':len(scalar)}))


if __name__=='__main__':main()
