"""Descriptive post-hoc H2aj failure topology; no target fitting or policy change."""
import csv
import json
import time
import numpy as np
import study as S
from verify_graph_block_source import connected
from factorial_transfer import PP,DEST,ARMS


def main():
    started=time.perf_counter();p=json.loads(PP.read_text());a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated';lookup={}
    with (DEST/'per-model-input.csv').open() as f:
        for r in csv.DictReader(f):lookup[(int(r['input_index']),int(r['model_seed']),r['arm'])]=int(r['success'])
    records=[];different_binary=different_block=0
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text())
        with np.load(DEST/f'shard{job["shard"]:02d}.npz') as z:tok=z['tokens'];labels=z['labels'];indices=z['indices']
        with np.load(run/'outputs.npz') as z:pred=z['predictions'];energies=z['energy64'];moves=z['block_moves']
        different_binary+=int(np.any(pred[1]!=pred[2],axis=(1,2)).sum());different_block+=int(np.any(pred[3]!=pred[4],axis=(1,2)).sum())
        for v in [3,4]:
            arm=ARMS[v];previous=ARMS[v-2];changed={}
            for i,index in enumerate(indices):
                before=lookup[(int(index),job['seed'],previous)];after=lookup[(int(index),job['seed'],arm)]
                if before==after:continue
                changed[i]={'input_index':int(index),'model_seed':job['seed'],'arm':arm,'change':'gain' if after else 'loss','selected_cells_before':int(pred[v-2,i].sum()),'selected_cells_after':int(pred[v,i].sum()),'canonical_label_cells':int(labels[i].sum()),'endpoint_connected_before':connected(tok[i],pred[v-2,i]),'endpoint_connected_after':connected(tok[i],pred[v,i]),'energy_change':float(energies[v,i]-energies[v-2,i]),'block_moves':int(moves[v-3,i]),'moves':[]}
            for item in receipt['trace_archives']:
                if item['arm']!=arm:continue
                lo=item['offset']
                if not any(lo<=i<lo+64 for i in changed):continue
                with np.load(run/item['path']) as z:valid=z['valid_prefix_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())['trace']
                for ri,round_ in enumerate(trace):
                    for j in round_['chosen']:
                        c=round_['candidates'][j];local=c['local_input'];i=lo+int(valid[local])
                        if i not in changed:continue
                        before=history[ri,local];after=before.copy();after.flat[c['cells']]=False
                        changed[i]['moves'].append({'round':ri,'cells':c['cells'],'delta':c['delta'],'removed_canonical_cells':int(labels[i].flat[c['cells']].sum()),'connected_before':connected(tok[i],before),'connected_after_deletion':connected(tok[i],after)})
            records.extend(changed.values())
    rows=[]
    for arm in ARMS[3:]:
        for kind in ['gain','loss']:
            part=[r for r in records if r['arm']==arm and r['change']==kind];rows.append({'arm':arm,'change':kind,'model_input_cases':len(part),'distinct_inputs':len({r['input_index'] for r in part}),'final_disconnected':sum(not r['endpoint_connected_after'] for r in part),'first_move_disconnects':sum(bool(r['moves']) and not r['moves'][0]['connected_after_deletion'] for r in part),'any_move_disconnects':sum(any(not m['connected_after_deletion'] for m in r['moves']) for r in part),'all_energies_strictly_decrease':all(r['energy_change']<0 for r in part)})
    result={'status':'complete','protocol_sha256':S.sha(PP),'audit_sha256':S.sha(DEST/'audit.json'),'rows':rows,'old_new_binary_prediction_differences':different_binary,'old_new_block_prediction_differences':different_block,'records':records,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'scope':'Post-hoc diagnosis on alreadyobserved H2ajoutputs, no freshdata/inference/fitting, labels for diagnosis only. All gains/losses retained; not an independent test.'};S.save(DEST/'failure-diagnosis.json',result);S.emit('factorial_failures_diagnosed',rows=rows,old_new_binary_prediction_differences=different_binary,old_new_block_prediction_differences=different_block)


if __name__=='__main__':main()
