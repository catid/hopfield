"""H2aw first-input and all frozen-source nonsimple fixtures; no new-outcome selection."""
import json
import numpy as np
import study as S
import fast_npz as Z
from large_cycle_cut import PP,DEST,PARENT


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);rows=[]
    for job in p['jobs']:
        folder=job['id']
        with np.load(PARENT/'runs'/folder/'outputs.npz') as z:old={k:z[k] for k in ['predictions','minimum_flip_margin64','energy64','couplings']}
        new={}
        for policy in ['lexical512','energy512']:
            with np.load(DEST/policy/'runs'/folder/'outputs.npz') as z:new[policy]={k:z[k] for k in z.files}
        selected=sorted({0,*np.flatnonzero(~new['lexical512']['initial_simple_path']).tolist()})
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:data={k:z[k] for k in ['tokens','labels','indices']}
        theta=[]
        for arm in ['pl','old_binary','old_binary','old_binary','old_binary']:
            with np.load(PARENT/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
        for i in selected:
            arrays={k:data[k][i:i+1] for k in ['tokens','labels']};arrays.update(theta=np.stack(theta),mu_offsets=mu,aoffs=ao,couplings=old['couplings'][i:i+1])
            for output,source,key in [('predictions','predictions','prediction'),('minimum_flip_margin64','minimum_flip_margin64','minimum_flip_margin'),('energy64','energy64','energy')]:
                arrays[output]=np.stack([old[source][j,i:i+1] for j in [0,1,4]]+[new[policy][key][i:i+1] for policy in ['lexical512','energy512']])
            path=target/f'{folder}-i{i:03d}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':int(data['indices'][i]),'model_seed':job['seed'],'selection_reason':'first_input_and_frozen_source_nonsimple' if i==0 and not new['lexical512']['initial_simple_path'][i] else 'first_input' if i==0 else 'frozen_source_nonsimple','source_parent_outputs_sha256':p['source_outputs'][folder]['sha256'],'source_new_outputs_sha256':{policy:S.sha(DEST/policy/'runs'/folder/'outputs.npz') for policy in new}})
    S.save(target/'manifest.json',{'selection':'Union of first input per shard/model and every frozen H2av lexical256 nonsimple input; no selection using new outcomes. Original BP/binary/lexical256 states inherited.','arms':['pl','old_binary','inherited_lexical256','lexical512_from_lexical256','energy512_from_lexical256'],'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
