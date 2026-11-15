"""First input per shard/model, all four arms, without outcome selection."""
import json
import numpy as np
import study as S
import fast_npz as Z
from connectivity_transfer_180 import PP,DEST,ARMS


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);rows=[]
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256']
        with np.load(run/'outputs.npz') as z:arrays={k:z[k][:,:1] for k in ['predictions','minimum_flip_margin64','energy64','stopped','finite']};arrays['couplings']=z['couplings'][:1]
        with np.load(DEST/f'shard{job["shard"]:02d}.npz') as z:arrays['tokens']=z['tokens'][:1];arrays['labels']=z['labels'][:1];index=int(z['indices'][0])
        theta=[]
        for arm in [ARMS[k] for k in [0,1,1,1]]:
            with np.load(DEST/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);arrays['mu_offsets']=z['mu_offsets'];arrays['aoffs']=z['aoffs']
        arrays['theta']=np.stack(theta);path=target/f'{job["id"]}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_outputs_sha256':r['outputs_sha256'],'checkpoint_sha256_by_arm':r['checkpoint_sha256_by_arm']})
    S.save(target/'manifest.json',{'selection':'Firstinput eachshard/allmodels/all4arms,48files192states; nooutcomeselection.','protocol_sha256':S.sha(PP),'arms':list(ARMS),'source_script_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
