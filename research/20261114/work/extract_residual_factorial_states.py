"""H2bb six-arm public fixtures; source-first plus every25 residualcap state."""
import json
import numpy as np
import study as S
import fast_npz as Z
from residual_factorial import PP,DEST,PARENT,ORIGIN


def main():
    p=json.loads(PP.read_text());target=DEST/'state-fixtures';target.mkdir(exist_ok=True);rows=[]
    for job in p['jobs']:
        jid=job['id']
        with np.load(PARENT/'runs'/jid/'outputs.npz') as z:base={k:z[k][1] for k in ['predictions','minimum_flip_margin64','energy64','block_cap']}
        with np.load(ORIGIN/'runs'/jid/'outputs.npz') as z:original={k:z[k] for k in ['predictions','minimum_flip_margin64','energy64','couplings']}
        new={}
        for policy in ['lexical_family','veto_family']:
            with np.load(DEST/policy/'runs'/jid/'outputs.npz') as z:new[policy]={k:z[k] for k in ['predictions','minimum_flip_margin64','energy64','eligible']}
        assert not np.any(new['lexical_family']['eligible']&new['veto_family']['eligible']);selected=sorted({0,*np.flatnonzero(base['block_cap']).tolist()})
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:data={k:z[k] for k in ['tokens','labels','indices']}
        theta=[]
        for arm in ['pl','old_binary','old_binary','old_binary','old_binary','old_binary']:
            with np.load(ORIGIN/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
        for i in selected:
            arrays={k:data[k][i:i+1] for k in ['tokens','labels']};arrays.update(theta=np.stack(theta),mu_offsets=mu,aoffs=ao,couplings=original['couplings'][i:i+1]);family='veto_family' if new['veto_family']['eligible'][i] else 'lexical_family'
            for key in ['predictions','minimum_flip_margin64','energy64']:arrays[key]=np.stack([original[key][v,i:i+1] for v in [0,1]]+[base[key][i:i+1]]+[new[policy][key][i:i+1] for policy in ['lexical_family','veto_family']]+[new[family][key][i:i+1]])
            path=target/f'{jid}-i{i:03d}.npz';Z.save(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':int(data['indices'][i]),'model_seed':job['seed'],'source_cap':bool(base['block_cap'][i]),'joint_source_family':family,'source_parent_outputs_sha256':p['source_outputs'][jid]['sha256'],'source_branch_outputs_sha256':{policy:S.sha(DEST/policy/'runs'/jid/'outputs.npz') for policy in new}})
    S.save(target/'manifest.json',{'selection':'First input per shard/model union all25 frozenH2ba capstates; no new-outcome selection. Joint arm composed perstate from disjointbranches with no duplicateinference.','arms':['pl','old_binary','inherited_H2ba','lexical_family_only','veto_family_only','both_families_composed'],'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
