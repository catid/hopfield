"""Post-outcome dense-basin diagnostic, independent sampled energy-flip checks."""
import csv
import json
import numpy as np
import study as S
from contrastive_training import DEST,PP,ARMS
from gate_contrastive import independent_energy


def main():
    torch,_,_=S.imports();rows=[];receipts=[]
    for gpu in range(2):
        receipt=json.loads((DEST/f'landscape-gpu{gpu}.json').read_text());assert receipt['status']=='complete' and receipt['csv_sha256']==S.sha(DEST/f'landscape-gpu{gpu}.csv');receipts.append(receipt);rows.extend(list(csv.DictReader((DEST/f'landscape-gpu{gpu}.csv').open())))
    structure_receipt=json.loads((DEST/'structures.json').read_text());assert structure_receipt['status']=='complete' and structure_receipt['csv_sha256']==S.sha(DEST/'structures.csv');structures=list(csv.DictReader((DEST/'structures.csv').open()));lookup={(int(r['input_index']),int(r['model_seed']),r['arm']):r for r in rows};summary=[]
    for arm in ['pl','contrastive','unchanged']:
        r=[x for x in rows if x['arm']==arm];z=[x for x in structures if x['arm']==arm]
        def quant(items,key):
            values=np.array([float(x[key]) for x in items]);return dict(zip(['min','p05','median','p95','max'],map(float,np.quantile(values,[0,.05,.5,.95,1]))))
        summary.append({'arm':arm,'model_inputs':len(r),'canonical_label_flip_stable':sum(float(x['label_minimum_flip_margin'])>=-1e-6 for x in r),'labels_with_negative_add_flip':sum(int(x['label_negative_add_flips'])>0 for x in r),'labels_with_negative_remove_flip':sum(int(x['label_negative_remove_flips'])>0 for x in r),'allopen_lower_than_label':sum(float(x['allopen_minus_label_energy'])<0 for x in r),'allopen_flip_stable':sum(float(x['allopen_minimum_flip_margin'])>=-1e-6 for x in r),'branched_outputs':sum(int(x['branch_nodes'])>0 for x in z),'disconnected_outputs':sum(int(x['components'])>1 for x in z),'landscape_quantiles':{k:quant(r,k) for k in ['label_minimum_flip_margin','label_minimum_add_margin','label_minimum_remove_margin','label_negative_add_flips','label_negative_remove_flips','allopen_minus_label_energy','allopen_minimum_flip_margin']},'structure_quantiles':{k:quant(z,k) for k in ['selected_nodes','components','branch_nodes','isolated_nodes','selected_free_fraction']}})
    # First input of each shard and each fixed model/arm, independent of outcome.
    maximum_energy_error=maximum_flip_error=0.;checks=0;case={'selection':'Post-outcome descriptive first input(index0), model0; no primarydecision change.','variants':[]}
    for seed in range(3):
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cpu',torch.float64)
            if arm!='unchanged':model.load_state_dict(torch.load(DEST/f'fits/s{seed}-{arm}/checkpoint.pt',map_location='cpu',weights_only=True))
            for shard in range(16):
                with np.load(DEST/f'shard{shard:02d}.npz') as x:token=x['tokens'][0];label=x['labels'][0]
                with np.load(DEST/f'runs/shard{shard:02d}-s{seed}/outputs.npz') as x:attention=x['couplings'][0];prediction=x['predictions'][ARMS.index(arm),0]
                t=torch.tensor(token.astype(np.int64))[None];y=torch.tensor(label,dtype=torch.float64)[None];a=torch.tensor(attention,dtype=torch.float64)[None]
                with torch.no_grad():
                    d=model.flip_delta(t,y,a).masked_fill(t!=1,float('inf'));j=int(d.flatten().argmin());flipped=y.clone();flipped.flatten()[j]=1-flipped.flatten()[j]
                    directflip=independent_energy(torch,model,t,flipped,a)-independent_energy(torch,model,t,y,a);directgap=independent_energy(torch,model,t,torch.ones_like(y),a)-independent_energy(torch,model,t,y,a)
                expected=lookup[(shard*256,seed,arm)];fe=abs(float(directflip)-float(expected['label_minimum_flip_margin']));ee=abs(float(directgap)-float(expected['allopen_minus_label_energy']));assert fe<1e-7 and ee<1e-7;maximum_energy_error=max(ee,maximum_energy_error);maximum_flip_error=max(fe,maximum_flip_error);checks+=1
                if seed==shard==0:
                    case['tokens']=token.tolist();case['canonical_label']=label.astype(int).tolist();case['variants'].append({'arm':arm,'prediction':prediction.astype(int).tolist(),'landscape':expected,'structure':next(z for z in structures if int(z['input_index'])==0 and int(z['model_seed'])==0 and z['arm']==arm)})
    rows.sort(key=lambda r:(int(r['input_index']),int(r['model_seed']),r['arm']))
    with (DEST/'landscape.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'validated','post_outcome_descriptive':True,'utc':S.utc(),'rows':summary,'receipts':receipts,'structure_receipt':structure_receipt,'independent_direct_energy_checks':checks,'maximum_allopen_gap_error':maximum_energy_error,'maximum_selected_label_flip_error':maximum_flip_error,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'new_independent_inputs':0,'new_answer_trajectories':0,'scope':'ReusedH2vcohort/fixedreadouts. Canonical shortestlabels only, not allalternative shortestpaths. Denseallopencomparison is a new diagnostic negative, never fed into H2vtraining or inference. Positive label flipmargin does not guarantee that nativeu3reaches its basin.'};S.save(DEST/'landscape-summary.json',result);S.save(DEST/'dense-case.json',case);S.emit('contrastive_landscape_summarized',summary=summary,checks=checks,max_energy_error=maximum_energy_error,max_flip_error=maximum_flip_error)


if __name__=='__main__':main()
