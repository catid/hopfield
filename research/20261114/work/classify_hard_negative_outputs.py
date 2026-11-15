"""Post-outcome H2ac graph structures and saved-counter algebra."""
import csv
import json
import time
import numpy as np
from scipy.ndimage import label as components
import study as S
from diagnose_restart import classify
from hard_negative_transfer import PP,DEST


def main():
    started=time.perf_counter();p=json.loads(PP.read_text());rows=[];cache={};checks=0;cross=np.array([[0,1,0],[1,1,1],[0,1,0]])
    with (DEST/'per-model-input.csv').open() as f:outcomes={(int(r['input_index']),int(r['model_seed'])):r for r in csv.DictReader(f) if r['arm']=='hard_negative_binary'}
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
        with np.load(run/'outputs.npz') as x:pred=x['predictions'][2];flips=x['binary_flip_count'][1]
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as x:tokens=x['tokens'];labels=x['labels'];indices=x['indices']
        for i,(tok,board,y,index) in enumerate(zip(tokens,pred,labels,indices)):
            key=(int(index),np.packbits(board).tobytes())
            if key not in cache:
                deg=np.zeros(board.shape,np.uint8);deg[1:]+=board[:-1];deg[:-1]+=board[1:];deg[:,1:]+=board[:,:-1];deg[:,:-1]+=board[:,1:]
                cc,ncomp=components(board,cross);s=int(cc[tok==2][0]);g=int(cc[tok==3][0]);main=cc==s;off=[k for k in range(1,ncomp+1) if k!=s]
                z={'selected_nodes':int(board.sum()),'components':int(ncomp),'branch_nodes':int((board&(deg>2)).sum()),'isolated_nodes':int((board&(deg==0)).sum()),'endpoint_only':int(board.sum()==2),'start_goal_connected':int(s==g),'main_component_selected':int(main.sum()),'main_component_branch_nodes':int((main&(deg>2)).sum()),'off_start_components':len(off),'off_start_cycle_components':sum(bool(np.all(deg[cc==k]==2)) for k in off),'missed_label_cells':int((y&~board).sum()),'extra_selected_cells':int((board&~y).sum()),'contains_canonical_label':int(np.all(board[y]))}
                z['main_component_is_canonical_label']=int(np.array_equal(main,y));cache[key]=z
            row=dict(cache[key],input_index=int(index),model_seed=job['seed'],success=int(outcomes[(int(index),job['seed'])]['success']),recorded_flip_count=int(flips[i]),initial_selected_nodes=int((tok>0).sum()))
            difference=row['recorded_flip_count']-(row['initial_selected_nodes']-row['selected_nodes']);assert difference>=0 and difference%2==0
            row['additions_implied_by_counter']=difference//2;row['removals_implied_by_counter']=row['recorded_flip_count']-difference//2
            assert row['recorded_flip_count']==int(outcomes[(int(index),job['seed'])]['binary_flip_count'])
            rows.append(row)
            if i==0:
                reference=classify(tok,board);assert all(row[k]==reference[k] for k in ['selected_nodes','components','branch_nodes','isolated_nodes','endpoint_only']);checks+=1
        S.emit('hard_negative_structure_job',job=job['id'])
    path=DEST/'structures.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups={}
    for label,part in [('all',rows),('wrong',[r for r in rows if not r['success']]),('solved',[r for r in rows if r['success']])]:
        groups[label]={'n':len(part),'disconnected':sum(r['components']>1 for r in part),'branched':sum(r['branch_nodes']>0 for r in part),'start_goal_connected':sum(r['start_goal_connected'] for r in part),'contains_canonical_label':sum(r['contains_canonical_label'] for r in part),'main_component_is_canonical_label':sum(r['main_component_is_canonical_label'] for r in part),'has_off_start_cycle':sum(r['off_start_cycle_components']>0 for r in part),'positive_inferred_additions':sum(r['additions_implied_by_counter']>0 for r in part),'metrics':{k:{'min':min(r[k] for r in part),'median':float(np.median([r[k] for r in part])),'max':max(r[k] for r in part)} for k in ['components','branch_nodes','extra_selected_cells','missed_label_cells']}}
    S.save(DEST/'structures.json',{'status':'complete','groups':groups,'source_states':12288,'independent_set_classifier_checks':checks,'unique_graph_checks':len(cache),'counter_scope':'Everyper-inputtotalcounter checked. Forbinarytransitions F=R+A and S0-Sf=R-A, so A=[F-(S0-Sf)]/2. Equalityimplieszeroadditions assumingrecordedcountercorrect. Per-phase targeteventtraces notretained; no independenttargeteventreplayclaim. Sourceeventcounterchecks separate.','new_answer_trajectories':0,'seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'csv_sha256':S.sha(path)})


if __name__=='__main__':main()
