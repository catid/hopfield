"""H2au deterministic first-input-per-shard/model public state fixtures."""
import json
import numpy as np
import study as S
import fast_npz as Z
from cycle_transfer import PP,DEST,ARMS


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);manifest=json.loads((DEST/'datasets.json').read_text());rows=[]
    for job in p['jobs']:
        folder=job['id'];arrays={}
        with np.load(DEST/'runs'/folder/'outputs.npz') as z:
            for k in ['predictions','minimum_flip_margin64','energy64']:arrays[k]=z[k][:,:1]
            arrays['couplings']=z['couplings'][:1]
        with np.load(S.ROOT/manifest['datasets'][job['shard']]['path']) as z:arrays['tokens']=z['tokens'][:1];arrays['labels']=z['labels'][:1];index=int(z['indices'][0])
        theta=[]
        for arm in ['pl','old_binary','old_binary','old_binary','old_binary']:
            with np.load(DEST/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);arrays['mu_offsets']=z['mu_offsets'];arrays['aoffs']=z['aoffs']
        arrays['theta']=np.stack(theta);path=target/f'{folder}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_outputs_sha256':S.sha(DEST/'runs'/folder/'outputs.npz')})
    S.save(target/'manifest.json',{'selection':'Firstinput each32shards/all3models, fivearms,480states96labels; nooutcomeselection.','arms':list(ARMS),'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
