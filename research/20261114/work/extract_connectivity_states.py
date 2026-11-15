"""H2ak deterministic first-input public fixtures, five retained policies."""
import json
import numpy as np
import study as S
import fast_npz as Z
from connectivity_screen import PP,DEST,PARENT


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);rows=[]
    for job in p['jobs']:
        folder=job['id'];arrays={}
        with np.load(PARENT/'runs'/folder/'outputs.npz') as z:pred=[z['predictions'][k,:1] for k in [0,1,3]];margin=[z['minimum_flip_margin64'][k,:1] for k in [0,1,3]];energies=[z['energy64'][k,:1] for k in [0,1,3]];arrays['couplings']=z['couplings'][:1]
        with np.load(DEST/'skip'/folder/'outputs.npz') as z:pred.append(z['predictions'][:1]);margin.append(z['minimum_flip_margin64'][:1]);energies.append(z['energy64'][:1])
        with np.load(DEST/'veto'/folder/'outputs.npz') as z:pred.append(z['prediction'][:1]);margin.append(z['minimum_flip_margin'][:1]);energies.append(z['energy'][:1])
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:arrays['tokens']=z['tokens'][:1];arrays['labels']=z['labels'][:1];index=int(z['indices'][0])
        theta=[]
        for arm in ['pl','old_binary','old_binary','old_binary','old_binary']:
            with np.load(PARENT/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);arrays['mu_offsets']=z['mu_offsets'];arrays['aoffs']=z['aoffs']
        arrays.update(theta=np.stack(theta),predictions=np.stack(pred),minimum_flip_margin64=np.stack(margin),energy64=np.stack(energies));path=target/f'{folder}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_parent_outputs_sha256':p['parent_results'][folder]['outputs_sha256'],'source_skip_outputs_sha256':S.sha(DEST/'skip'/folder/'outputs.npz'),'source_veto_outputs_sha256':S.sha(DEST/'veto'/folder/'outputs.npz')})
    S.save(target/'manifest.json',{'selection':'Firstinput eachshard/allmodels, fivearms,240states48labels; nooutcomeselection.','arms':['pl','old_binary','old_blocks','skip_simple_path','connectivity_veto'],'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
