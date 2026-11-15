"""Family-specific fresh cohorts after every source timing outcome is sealed."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import numpy as np
import torch
from common import ROOT,SRC,OUT,OLD,sha,read,save,now,frozen
from initialization import make

def main():
    torch.set_num_threads(1);p=frozen(target=True);assert not(OUT/'datasets.json').exists();gate=read(OUT/'enumeration-gate.json');assert gate['status']=='complete_verified';prior_receipt=read(OLD/'dataset.json');assert prior_receipt['protocol_sha256']==p['predecessor']['protocol_sha256']and prior_receipt['data_sha256']==sha(OLD/'data.npz');old=np.load(OLD/'data.npz');prior=np.union1d(old['all_excluded_keys'],old['test_keys']);calkeys=np.union1d(*[np.load(OUT/f'b{k}/calibration-data.npz')['keys']for k in range(2)]);all_excluded=np.union1d(prior,calkeys);outcomes={};datasets=[];prior_new=np.array([],np.int64)
    for bank in range(2):
        folder=OUT/f'b{bank}';model=read(folder/'model.json')
        if not model['source_eligible']:outcomes[str(bank)]=dict(status='source_fit_ineligible');continue
        selection=read(folder/'calibration-selection.json');outcomes[str(bank)]=dict(status=selection['status'],selection_sha256=sha(folder/'calibration-selection.json'))
        for rel,h in selection['calibration_result_sha256'].items():assert sha(ROOT/rel)==h
    # All outcomes now read before any new cohort draw.
    for bank in range(2):
        folder=OUT/f'b{bank}'
        if outcomes[str(bank)]['status']!='matched_source_median_times':continue
        selection=read(folder/'calibration-selection.json');source_manifest=read(folder/'universe.json');assert source_manifest['universe_sha256']==sha(folder/'input-universe.npz');assert gate['families'][bank]['target_solutions_sha256']==sha(folder/'solutions-n13.npy');universe=np.load(folder/'input-universe.npz');excluded=np.union1d(all_excluded,prior_new);eligible=np.setdiff1d(universe['keys'],excluded);assert len(eligible)>=4096;seed=2026112517+bank;keys=np.sort(np.random.default_rng(seed).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
        for power in[2,1,0]:
            cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
        target=np.load(folder/'solutions-n13.npy');counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
        for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
        np.savez_compressed(folder/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_search_and_calibration_keys=prior,calibration_keys=calkeys,prior_new_family_test_keys=prior_new,all_excluded_keys=excluded);receipts=[]
        for searchseed in[0,1,2]:
            initial=make(inputs,q=13,chains=20,seed=searchseed,kind='permutation');path=folder/f'initial/s{searchseed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,boards=np.packbits(initial,axis=-1));assert np.array_equal(initial,make(inputs,q=13,chains=20,seed=searchseed,kind='permutation'));receipts.append(dict(seed=searchseed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(initial.tobytes()).hexdigest()))
        dataset=dict(status='prepared_after_all_source_caps_sealed',utc=now(),bank=bank,leaper=p['families'][bank]['leaper'],source_patterns=p['families'][bank]['source_patterns'],complete_target_solutions=len(target),input_seed=seed,unique_inputs=4096,prior_search_and_calibration_keys=len(prior),new_unique_calibration_keys=len(calkeys),prior_new_family_test_keys=len(prior_new),all_excluded_keys=len(excluded),eligible_keys=len(eligible),prior_overlap=0,calibration_overlap=0,other_new_family_overlap=0,by_clues={str(k):int((inputs.sum(1)==k).sum())for k in[1,2,3]},compatible_completions=int(counts.sum()),data_sha256=sha(folder/'data.npz'),initial_states=receipts,selected_caps={'arbitrary':20000,'permutation':selection['selected_permutation_cap']},calibration_selection_sha256=sha(folder/'calibration-selection.json'),model_sha256=sha(folder/'model.npz'),predecessor_data_sha256=sha(OLD/'data.npz'),protocol_sha256=sha(SRC/'protocol.json'));save(folder/'dataset.json',dataset)
        paths=[SRC/'protocol.json',folder/'data.npz',folder/'dataset.json',folder/'model.npz',folder/'calibration-selection.json',OUT/'enumeration-gate.json']+[ROOT/r['path']for r in receipts];save(folder/'preflight.json',dict(status='passed',utc=now(),all_source_caps_sealed_before_cohorts=True,all_target_counts_verified=True,shared_initials_reconstructed=True,input_sha256={str(path.relative_to(ROOT)):sha(path)for path in paths}));datasets.append(dataset);prior_new=np.union1d(prior_new,keys);print('TARGET_READY',bank,dataset['by_clues'],dataset['compatible_completions'],flush=True)
    save(OUT/'datasets.json',dict(status='prepared',utc=now(),outcomes=outcomes,datasets=datasets,all_new_family_test_keys=prior_new.tolist(),all_prior_and_calibration_excluded_keys=all_excluded.tolist(),protocol_sha256=sha(SRC/'protocol.json')))
if __name__=='__main__':main()
