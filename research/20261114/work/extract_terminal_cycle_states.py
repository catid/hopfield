"""H2ba public first-input plus every frozen lexical-cap state, no new-outcome selection."""
import json
import numpy as np
import study as S
import fast_npz as Z
from terminal_cycle import PP,DEST,PARENT


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);rows=[]
    for job in p['jobs']:
        jid=job['id']
        with np.load(PARENT/'runs'/jid/'outputs.npz') as z:old={k:z[k] for k in ['predictions','minimum_flip_margin64','energy64','couplings','block_cap']}
        with np.load(DEST/'runs'/jid/'outputs.npz') as z:new={k:z[k] for k in ['predictions','minimum_flip_margin64','energy64']}
        selected=sorted({0,*np.flatnonzero(old['block_cap'][1]).tolist()})
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:data={k:z[k] for k in ['tokens','labels','indices']}
        theta=[]
        for arm in ['pl','old_binary','old_binary','old_binary','old_binary']:
            with np.load(PARENT/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
        for i in selected:
            arrays={k:data[k][i:i+1] for k in ['tokens','labels']};arrays.update(theta=np.stack(theta),mu_offsets=mu,aoffs=ao,couplings=old['couplings'][i:i+1])
            for key in ['predictions','minimum_flip_margin64','energy64']:arrays[key]=np.stack([old[key][v,i:i+1] for v in [0,1,3]]+[new[key][v,i:i+1] for v in [0,1]])
            path=target/f'{jid}-i{i:03d}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':int(data['indices'][i]),'model_seed':job['seed'],'source_lexical_cap':bool(old['block_cap'][1,i]),'source_veto_cap':bool(old['block_cap'][0,i]),'source_parent_outputs_sha256':p['source_outputs'][jid]['sha256'],'source_new_outputs_sha256':S.sha(DEST/'runs'/jid/'outputs.npz')})
    S.save(target/'manifest.json',{'selection':'First input per shard/model union every204 frozen lexical256 capped state, including10 inheritedveto caps. No selection on new outcomes.','arms':['pl','old_binary','inherited_lexical256','terminal_only','terminal_then_extra16'],'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
