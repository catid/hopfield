"""H2bk fixed public subset and every routed or failed reported model case."""
import json
import numpy as np
import study as S
import fast_npz as Z
from fallback_confirmation_study import PP,DEST,REPORTED


def main():
    p=json.loads(PP.read_text());a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';manifest=json.loads((DEST/'datasets.json').read_text());byjob={r['job']['id']:r for r in a['rows']}
    targets={name:DEST/name for name in ['state-fixtures','routed-failure-fixtures']};rows={name:[] for name in targets}
    for target in targets.values():target.mkdir(exist_ok=True)
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256'] and S.sha(run/'hybrid-outputs.npz')==receipt['hybrid_outputs_sha256']
        with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','iterations','finite','stopped','minimum_flip_margin64','energy64','couplings','hidden_good','block_cap','block_native_cap','native_final_logits']}
        with np.load(run/'hybrid-outputs.npz') as z:h={k:z[k] for k in z.files}
        ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens'];labels=z['labels']
        theta=[]
        for arm in ['pl','old_binary']:
            with np.load(DEST/f'model-seed{job["seed"]}-{arm}.npz') as z:theta.append(z['theta']);mu=z['mu_offsets'];ao=z['aoffs']
        success=np.array(byjob[job['id']]['success'],bool);supplement=np.flatnonzero(~success.all(0)|~h['lp_accepted']);subsets={}
        for name,indices in [('state-fixtures',[0]),('routed-failure-fixtures',supplement)]:
            for i in indices:
                i=int(i);routed=not h['lp_accepted'][i];index=job['shard']*256+i;chosen=0 if routed else 1
                if routed:
                    batch=i//32
                    if batch not in subsets:
                        path=run/f'batch{batch:02d}-subset.npz'
                        with np.load(path) as z:subsets[batch]={k:z[k] for k in z.files}
                    sub=subsets[batch];j=int(np.flatnonzero(sub['indices']==i%32)[0]);hybrid_energy=sub['own_energy'][j]
                else:hybrid_energy=old['energy64'][7,i]
                arrays={'tokens':tokens[i:i+1],'labels':labels[i:i+1],'couplings':old['couplings'][i:i+1],
                    'theta':np.stack([theta[0],theta[1],theta[chosen],theta[chosen]]),'mu_offsets':mu,'aoffs':ao,
                    'predictions':np.stack([old['predictions'][0,i:i+1],old['predictions'][7,i:i+1],h['prediction'][i:i+1],h['prediction'][i:i+1]]),
                    'minimum_flip_margin64':np.array([[old['minimum_flip_margin64'][0,i]],[old['minimum_flip_margin64'][7,i]],[h['output_own_margin'][i]],[h['output_own_margin'][i]]]),
                    'energy64':np.array([[old['energy64'][0,i]],[old['energy64'][7,i]],[hybrid_energy],[hybrid_energy]]),
                    'hidden_good':old['hidden_good'][i:i+1],'hybrid_output_model':np.array([1 if routed else 0],np.uint8),
                    'source_lp_margin':h['source_lp_margin'][i:i+1],'hybrid_lp_margin':h['output_lp_margin'][i:i+1],
                    'lp_block_cap':old['block_cap'][5,i:i+1],'lp_native_cap':old['block_native_cap'][5,i:i+1],
                    'lp_accepted':h['lp_accepted'][i:i+1],'accepted_main':h['accepted_main'][i:i+1],'accepted_dual':h['accepted_dual'][i:i+1]}
                for k in ['iterations','finite','stopped']:arrays[k]=np.stack([old[k][0,i:i+1],old[k][7,i:i+1],h[k][i:i+1],h[k][i:i+1]])
                arrays['bp_control_final_logits']=old['native_final_logits'][0,i:i+1]
                if routed:
                    arrays.update(subset_final_logits=sub['final_logits'][j:j+1],subset_minimum_margin32=sub['minimum_margin32'][j:j+1])
                else:arrays.update(subset_final_logits=np.zeros((0,180,180),np.float32),subset_minimum_margin32=np.zeros(0,np.float32))
                path=targets[name]/f'i{index}-s{job["seed"]}.npz';Z.save(path,**arrays)
                rows[name].append({'file':path.name,'sha256':S.sha(path),'input_index':index,'model_seed':job['seed'],'routed':bool(routed),
                    'expected_success_by_arm':success[:,i].astype(int).tolist(),'source_outputs_sha256':receipt['outputs_sha256'],'source_hybrid_outputs_sha256':receipt['hybrid_outputs_sha256'],
                    'bp_checkpoint_sha256':p['models'][str(job['seed'])]['pl']['sha256'],'lp_checkpoint_sha256':p['models'][str(job['seed'])]['old_binary']['sha256']})
    for name,target in targets.items():
        S.save(target/'manifest.json',{'selection':'First input of every shard/model, independent of outcomes.' if name=='state-fixtures' else 'Every routed case or model-input failing any reported arm, including BP controls; no failed cases omitted. Explicit outcome-selected supplement.',
            'arms':list(REPORTED),'protocol_sha256':S.sha(PP),'source_audit_sha256':S.sha(DEST/'audit.json'),'fixtures':rows[name],'extractor_sha256':S.sha(__file__)})
    print(json.dumps({'fixed_cases':len(rows['state-fixtures']),'routed_or_failed_cases':len(rows['routed-failure-fixtures'])}))


if __name__=='__main__':main()
