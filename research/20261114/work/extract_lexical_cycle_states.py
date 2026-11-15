"""H2at deterministic first-input public fixtures; reused comparisons explicit."""
import json
import numpy as np
import study as S
import fast_npz as Z
from lexical_cycle_cut import PP,DEST,PARENT
ENERGY=S.OUT/'cycle_cut_rescue_checked'


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);rows=[]
    for job in p['jobs']:
        folder=job['id'];arrays={}
        with np.load(PARENT/'runs'/folder/'outputs.npz') as z:pred=[z['predictions'][k,:1] for k in [0,1,3]];margin=[z['minimum_flip_margin64'][k,:1] for k in [0,1,3]];energies=[z['energy64'][k,:1] for k in [0,1,3]];arrays['couplings']=z['couplings'][:1]
        with np.load(ENERGY/'runs'/folder/'outputs.npz') as z:pred.append(z['prediction'][:1]);margin.append(z['minimum_flip_margin'][:1]);energies.append(z['energy'][:1])
        with np.load(DEST/'runs'/folder/'outputs.npz') as z:pred.append(z['prediction'][:1]);margin.append(z['minimum_flip_margin'][:1]);energies.append(z['energy'][:1])
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:arrays['tokens']=z['tokens'][:1];arrays['labels']=z['labels'][:1];index=int(z['indices'][0])
        theta=[]
        for arm in ['pl','old_binary','old_binary','old_binary','old_binary']:
            with np.load(PARENT/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);arrays['mu_offsets']=z['mu_offsets'];arrays['aoffs']=z['aoffs']
        arrays.update(theta=np.stack(theta),predictions=np.stack(pred),minimum_flip_margin64=np.stack(margin),energy64=np.stack(energies));path=target/f'{folder}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_parent_outputs_sha256':p['source_outputs'][folder]['sha256'],'source_energy_ranked_outputs_sha256':S.sha(ENERGY/'runs'/folder/'outputs.npz'),'source_lexical_outputs_sha256':S.sha(DEST/'runs'/folder/'outputs.npz')})
    S.save(target/'manifest.json',{'selection':'Firstinput eachshard/allmodels, fivearms,240states48labels; nooutcomeselection. Allfirstfourarmsareexactarchivedcomparisons.','arms':['pl','old_binary','connectivity_veto','energy_ranked_cycle_cut','lexical_cycle_cut'],'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
