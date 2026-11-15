"""Post-outcome saved-readout landscape and topology diagnostics; no fits/solves."""
import argparse
import copy
import csv
import json
import time
import numpy as np
import study as S
import selective_fallback as F
from contrastive_training import DEST,PP,ARMS,protocol


def landscape(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter();rows=[];models={}
    for seed in range(3):
        models[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cuda',torch.float64)
            if arm!='unchanged':model.load_state_dict(torch.load(DEST/f'fits/s{seed}-{arm}/checkpoint.pt',map_location='cuda',weights_only=True))
            models[seed].append(model)
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as x:tokens=x['tokens'];labels=x['labels'];indices=x['indices']
        with np.load(DEST/f"runs/{job['id']}/outputs.npz") as x:attention=x['couplings']
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');y=torch.tensor(labels[lo:lo+64],device='cuda',dtype=torch.float64);a=torch.tensor(attention[lo:lo+64],device='cuda',dtype=torch.float64);free=t==1;on=free&y.bool();off=free&~y.bool();allopen=torch.ones_like(y)
                for arm,model in zip(ARMS,models[job['seed']]):
                    delta=model.flip_delta(t,y,a);minimum=delta.masked_fill(~free,float('inf')).flatten(1).amin(1);remove=delta.masked_fill(~on,float('inf')).flatten(1).amin(1);add=delta.masked_fill(~off,float('inf')).flatten(1).amin(1);negadd=((delta<-1e-6)&off).sum((1,2));negremove=((delta<-1e-6)&on).sum((1,2));gap=model.energy(t,allopen,a)-model.energy(t,y,a);openmargin=model.flip_delta(t,allopen,a).masked_fill(~free,float('inf')).flatten(1).amin(1)
                    values=[v.cpu().numpy() for v in [minimum,remove,add,negadd,negremove,gap,openmargin]]
                    for i,index in enumerate(indices[lo:lo+64]):rows.append({'input_index':int(index),'model_seed':job['seed'],'arm':arm,'label_minimum_flip_margin':float(values[0][i]),'label_minimum_remove_margin':float(values[1][i]),'label_minimum_add_margin':float(values[2][i]),'label_negative_add_flips':int(values[3][i]),'label_negative_remove_flips':int(values[4][i]),'allopen_minus_label_energy':float(values[5][i]),'allopen_minimum_flip_margin':float(values[6][i])})
        S.emit('contrastive_landscape_diagnostic',gpu=gpu,job=job['id'])
    with (DEST/f'landscape-gpu{gpu}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    S.save(DEST/f'landscape-gpu{gpu}.json',{'status':'complete','post_outcome_descriptive':True,'utc':S.utc(),'records':len(rows),'new_fits':0,'new_answer_trajectories':0,'inventory':inventory,'seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'csv_sha256':S.sha(DEST/f'landscape-gpu{gpu}.csv'),'scope':'Retained sharedfloat32readouts and trainedweights promoted exactly to float64. One canonicalshortestlabelperinput, not all alternative shortestpaths. No decoder changes.'})


def structures():
    from scipy.ndimage import label as components
    from diagnose_restart import classify
    started=time.perf_counter();p=json.loads(PP.read_text());rows=[];cross=np.array([[0,1,0],[1,1,1],[0,1,0]])
    checks=0
    for job in p['jobs']:
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as x:tokens=x['tokens'];indices=x['indices']
        with np.load(DEST/f"runs/{job['id']}/outputs.npz") as x:pred=x['predictions'];stopped=x['stopped']
        for v,arm in enumerate(ARMS):
            for i,board in enumerate(pred[v]):
                deg=np.zeros(board.shape,np.uint8);deg[1:]+=board[:-1];deg[:-1]+=board[1:];deg[:,1:]+=board[:,:-1];deg[:,:-1]+=board[:,1:];ncomp=int(components(board,cross)[1]);branches=int((board&(deg>2)).sum());isolated=int((board&(deg==0)).sum());selected=int(board.sum())
                row={'input_index':int(indices[i]),'model_seed':job['seed'],'arm':arm,'certified':int(stopped[v,i]),'selected_nodes':selected,'components':ncomp,'branch_nodes':branches,'isolated_nodes':isolated,'selected_free_fraction':float((board&(tokens[i]==1)).sum()/(tokens[i]==1).sum()),'endpoint_only':int(selected==2 and board[tokens[i]>=2].all())};rows.append(row)
                if i==0:
                    reference=classify(tokens[i],board);assert all(row[k]==reference[k] for k in ['selected_nodes','components','branch_nodes','isolated_nodes','endpoint_only']);checks+=1
        S.emit('contrastive_structure_diagnostic',job=job['id'])
    with (DEST/'structures.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    S.save(DEST/'structures.json',{'status':'complete','post_outcome_descriptive':True,'rows':len(rows),'independent_set_based_classifier_checks':checks,'seconds':time.perf_counter()-started,'csv_sha256':S.sha(DEST/'structures.csv'),'source_sha256':S.sha(__file__),'new_answer_trajectories':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['landscape','structures']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='landscape':landscape(args.gpu)
    else:structures()
