"""One common fresh target cohort, reserved after every bank's cap is sealed."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import numpy as np
import torch
from common import ROOT,SRC,OUT,OLD,BASE,sha,read,save,now,frozen
from initialization import make

def main():
    torch.set_num_threads(1);frozen(target=True);assert not(OUT/'dataset.json').exists();calkeys=np.array([],np.int64);caps={};receipts=[];selection_hashes={};models={};failures={}
    for bank in range(4):
        folder=OUT/f'b{bank}';model=read(folder/'model.json');calkeys=np.union1d(calkeys,np.load(folder/'calibration-data.npz')['keys']);models[str(bank)]=model
        if not model['source_eligible']:failures[str(bank)]='source_fit_ineligible';continue
        cal=read(folder/'calibration-selection.json');selection_hashes[str(bank)]=sha(folder/'calibration-selection.json')
        if cal['status']!='matched_source_median_times':failures[str(bank)]=cal['status'];continue
        assert cal['no_target_accuracy_used']
        for rel,h in cal['calibration_result_sha256'].items():assert sha(ROOT/rel)==h
        caps[str(bank)]={'arbitrary':cal['selected_arbitrary_cap'],'permutation':cal['selected_permutation_cap']}
    old=np.load(OLD/'data.npz');prior=np.union1d(old['all_excluded_keys'],old['test_keys']);excluded=np.union1d(prior,calkeys);universe=np.load(BASE/'h3t/input-universe.npz');eligible=np.setdiff1d(universe['keys'],excluded);keys=np.sort(np.random.default_rng(2026112516).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
    for power in[2,1,0]:
        cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
    target=np.load(BASE/'h3t/solutions-n13.npy');counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_search_and_calibration_keys=prior,calibration_keys=calkeys,all_excluded_keys=excluded)
    for seed in[0,1,2]:
        states=make(inputs,q=13,chains=20,seed=seed,kind='permutation');path=OUT/f'initial/s{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,boards=np.packbits(states,axis=-1));assert states.tobytes()==make(inputs,q=13,chains=20,seed=seed,kind='permutation').tobytes();receipts.append(dict(seed=seed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(states.tobytes()).hexdigest()))
    assert not np.isin(keys,excluded).any();dataset=dict(status='prepared_after_all_source_cap_freezes',utc=now(),input_seed=2026112516,unique_inputs=4096,prior_search_and_calibration_keys=len(prior),new_unique_calibration_keys=len(calkeys),all_excluded_keys=len(excluded),eligible_keys=len(eligible),prior_overlap=0,calibration_overlap=0,by_clues={str(k):int((inputs.sum(1)==k).sum())for k in[1,2,3]},compatible_completions=int(counts.sum()),data_sha256=sha(OUT/'data.npz'),initial_states=receipts,selected_caps=caps,failed_banks=failures,calibration_selection_sha256=selection_hashes,protocol_sha256=sha(SRC/'protocol.json'),models=models)
    save(OUT/'dataset.json',dataset);paths=[SRC/'protocol.json',OUT/'data.npz',OUT/'dataset.json',*[ROOT/r['path']for r in receipts]]
    for bank in caps:paths.extend([OUT/f'b{bank}/calibration-selection.json',OUT/f'b{bank}/model.npz'])
    save(OUT/'preflight.json',dict(status='passed',utc=now(),all_bank_caps_sealed_before_common_cohort=True,shared_initial_arrays_reconstructed=True,all_target_validity_counts_verified=True,input_sha256={str(path.relative_to(ROOT)):sha(path)for path in paths}));print('FRESH_TARGET_READY',dataset,flush=True)
if __name__=='__main__':main()
