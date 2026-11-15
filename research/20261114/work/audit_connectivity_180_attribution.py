"""Post-hoc geometric limit on interpreting the H2am connectivity-veto benefit."""
import csv
import json
import time
import numpy as np
import study as S
from connectivity_transfer_180 import PP,DEST,ARMS


def main():
    started=time.perf_counter();p=json.loads(PP.read_text());rows=[];all_manhattan=[];success={}
    with (DEST/'per-model-input.csv').open() as f:
        for r in csv.DictReader(f):success[(int(r['input_index']),int(r['model_seed']),r['arm'])]=int(r['success'])
    for job in p['jobs']:
        with np.load(DEST/f'shard{job["shard"]:02d}.npz') as z:t=z['tokens'];indices=z['indices']
        dist=np.array([np.abs(np.argwhere(tok==2)[0]-np.argwhere(tok==3)[0]).sum() for tok in t]);assert np.all(dist==356);all_manhattan.extend(dist.tolist())
        with np.load(DEST/'runs'/job['id']/'outputs.npz') as z:pred=z['predictions'];simple=z['initial_simple_path'];free_cells=pred[1].sum((1,2))-2;different=np.any(pred[2]!=pred[3],axis=(1,2));energy_difference=z['energy64'][3]-z['energy64'][2]
        assert np.all(free_cells[simple]>=355) and np.all(free_cells[simple]>256);assert np.array_equal(pred[2,simple],pred[1,simple]);different_indices=[int(indices[i]) for i in np.flatnonzero(different)];outcome_differences=sum(success[(int(i),job['seed'],'skip_simple_path')]!=success[(int(i),job['seed'],'connectivity_veto')] for i in indices);rows.append({'job':job['id'],'initial_simple_paths':int(simple.sum()),'minimum_simple_path_free_cells':int(free_cells[simple].min()) if simple.any() else None,'skip_veto_board_differences':int(different.sum()),'different_board_input_indices':different_indices,'skip_veto_success_differences':outcome_differences,'energy_difference_on_different_boards':energy_difference[different].tolist()})
    result={'status':'passed','protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'rows':rows,'all_endpoint_manhattan_distances':sorted(set(all_manhattan)),'minimum_simple_path_free_cells':min(r['minimum_simple_path_free_cells'] for r in rows if r['minimum_simple_path_free_cells'] is not None),'initial_simple_paths':sum(r['initial_simple_paths'] for r in rows),'skip_veto_board_differences':sum(r['skip_veto_board_differences'] for r in rows),'skip_veto_success_differences':sum(r['skip_veto_success_differences'] for r in rows),'max_candidate_cells':256,'derivation':'Any simple endpointpath has at leastManhattanDistance+1 selectedcells. Endpointsclamped, remainingfreecells formone eligibledegree2component, ofsize>=356-1=355>256. Thus the ORIGINALunfilteredcandidate rule also cannot delete any correctsimplepath onthisdistribution. No originalblockarm was evaluated; noisolatedveto benefit claim.','scope':'Posthoc attribution boundary, reusedoutputs, zero newinference andzeroindependentinputs. Freshgain overbinary remains valid; skip/veto equality of jointsuccess is not algorithmidentity.','seconds':time.perf_counter()-started};S.save(DEST/'attribution-audit.json',result);S.emit('connectivity180_attribution',board_differences=result['skip_veto_board_differences'],success_differences=result['skip_veto_success_differences'])


if __name__=='__main__':main()
