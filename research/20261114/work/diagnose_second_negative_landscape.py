"""Post-outcome H2ae label/output energies and label stability; no search."""
import argparse
import csv
import json
import time
import numpy as np
import study as S
import selective_fallback as F
from second_negative_transfer import PP,DEST,protocol

D=DEST/'landscape'


def main(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();D.mkdir(exist_ok=True);started=time.perf_counter();models={};rows=[];receipts=[]
    for seed in range(3):
        m=S.model_for(seed,120,'original','cuda',torch.float64);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['second_negative_binary']['path'],map_location='cuda',weights_only=True));models[seed]=m
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
        with np.load(run/'outputs.npz') as x:couplings=x['couplings'];pred=x['predictions'][2];expected_energy=x['energy64'][2]
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as x:tokens=x['tokens'];labels=x['labels'];indices=x['indices']
        model=models[job['seed']];energies=[];margins=[]
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');a=torch.tensor(couplings[lo:lo+64],device='cuda',dtype=torch.float64);y=torch.tensor(labels[lo:lo+64],device='cuda',dtype=torch.float64);b=torch.tensor(pred[lo:lo+64],device='cuda',dtype=torch.float64)
                actual=model.energy(t,b,a).cpu().numpy();assert np.array_equal(actual,expected_energy[lo:lo+64])
                energies.append(model.energy(t,y,a).cpu().numpy());margins.append(model.flip_delta(t,y,a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy())
        le=np.concatenate(energies);lm=np.concatenate(margins);path=D/f"{job['id']}.npz";np.savez_compressed(path,label_energy64=le,label_minimum_flip64=lm,output_energy64=expected_energy,label_minus_output64=le-expected_energy)
        for i,index in enumerate(indices):rows.append({'input_index':int(index),'model_seed':job['seed'],'label_energy64':float(le[i]),'output_energy64':float(expected_energy[i]),'label_minus_output64':float(le[i]-expected_energy[i]),'label_minimum_flip64':float(lm[i]),'label_selected_cells':int(labels[i].sum())})
        receipts.append({'job':job['id'],'source_outputs_sha256':receipt['outputs_sha256'],'diagnostic_sha256':S.sha(path)});S.emit('second_negative_landscape_job',gpu=gpu,job=job['id'])
    csvpath=D/f'per-state-gpu{gpu}.csv'
    with csvpath.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    S.save(D/f'gpu{gpu}.json',{'status':'complete','utc':S.utc(),'inventory':inventory,'source_states':len(rows),'new_answer_trajectories':0,'seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'per_state_csv_sha256':S.sha(csvpath),'jobs':receipts,'scope':'Post-outcome fixed-state originalbinaryenergy andlabelstability diagnostics; allcanonical labels evaluation-only. No decoderchanges, targetselection, repair, newindependentinputs or newprimaryinference.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],required=True);main(parser.parse_args().gpu)
