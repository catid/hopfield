"""Export deterministic H2aa binary-energy fixtures for public verification."""
import numpy as np
import json
import study as S
from binary_coordinate_study import PP,DEST,ARMS,protocol


def main():
    p=protocol();torch,_,_=S.imports();target=DEST/'state-fixtures';target.mkdir(exist_ok=True);models={};rows=[]
    for seed in range(3):
        models[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cpu');model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)][arm]['path'],map_location='cpu',weights_only=True));models[seed].append(model)
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
        with np.load(run/'outputs.npz') as z:arrays={k:z[k][:,:1] for k in ['predictions','minimum_flip_margin64','energy64']};arrays['couplings']=z['couplings'][:1]
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as z:arrays['tokens']=z['tokens'][:1];arrays['labels']=z['labels'][:1];index=int(z['indices'][0])
        landscape=DEST/'landscape'/f"{job['id']}.npz"
        with np.load(landscape) as z:
            for key in ['label_energy64','label_minimum_flip64','label_minus_output64']:arrays[key]=z[key][:1]
        mm=models[job['seed']];arrays['theta']=np.stack([np.concatenate([m.T.numpy().ravel(),m.b.numpy(),m.kappa.numpy()]) for m in mm]);arrays['mu_offsets']=np.array(mm[0].mu_offsets);arrays['aoffs']=np.array(mm[0].aoffs)
        path=target/f"{job['id']}.npz";np.savez_compressed(path,**arrays);rows.append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'source_outputs_sha256':receipt['outputs_sha256'],'checkpoint_sha256_by_arm':receipt['checkpoint_sha256_by_arm'],'landscape_sha256':S.sha(landscape)})
    S.save(target/'manifest.json',{'selection':'Firstinput eachshard/allmodels/allarms:48fixturefiles,144states, nooutcomeselection.','protocol_sha256':S.sha(PP),'arms':list(ARMS),'source_script_sha256':S.sha(__file__),'fixtures':rows})


if __name__=='__main__':main()
