"""Reserve fresh targets only after the source timing-selected caps are sealed."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import numpy as np
import torch
from common import ROOT,SRC,OUT,OLD,BASE,sha,read,save,now,frozen
from initialization import make

def main():
    torch.set_num_threads(1);p=frozen();cal=read(OUT/'calibration-selection.json');assert cal['status']=='matched_source_median_times'and cal['no_target_accuracy_used'];assert not(OUT/'dataset.json').exists()
    for rel,h in cal['calibration_result_sha256'].items():assert sha(ROOT/rel)==h
    old=np.load(OLD/'data.npz');prior=np.union1d(old['prior_target_test_keys'],old['test_keys']);calkeys=np.load(OUT/'calibration-data.npz')['keys'];excluded=np.union1d(prior,calkeys);universe=np.load(BASE/'h3t/input-universe.npz');eligible=np.setdiff1d(universe['keys'],excluded)
    keys=np.sort(np.random.default_rng(2026112515).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
    for power in[2,1,0]:
        cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
    target=np.load(BASE/'h3t/solutions-n13.npy');counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=prior,calibration_keys=calkeys,all_excluded_keys=excluded)
    receipts=[]
    for seed in[0,1,2]:
        states=make(inputs,q=13,chains=20,seed=seed,kind='permutation');path=OUT/f'initial/s{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,boards=np.packbits(states,axis=-1))
        assert states.tobytes()==make(inputs,q=13,chains=20,seed=seed,kind='permutation').tobytes();receipts.append(dict(seed=seed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(states.tobytes()).hexdigest()))
    assert not np.isin(keys,prior).any()and not np.isin(keys,calkeys).any()
    dataset=dict(status='prepared_after_source_cap_freeze',utc=now(),input_seed=2026112515,unique_inputs=4096,prior_search_keys=len(prior),calibration_keys=len(calkeys),eligible_keys=len(eligible),prior_overlap=0,calibration_overlap=0,
        by_clues={str(k):int((inputs.sum(1)==k).sum())for k in[1,2,3]},compatible_completions=int(counts.sum()),data_sha256=sha(OUT/'data.npz'),initial_states=receipts,selected_caps={'arbitrary':cal['selected_arbitrary_cap'],'permutation':cal['selected_permutation_cap']},
        calibration_selection_sha256=sha(OUT/'calibration-selection.json'),protocol_sha256=sha(SRC/'protocol.json'),model_sha256=sha(OLD/'model.npz'))
    save(OUT/'dataset.json',dataset);save(OUT/'preflight.json',dict(status='passed',utc=now(),caps_sealed_before_cohort=True,shared_initial_arrays_reconstructed=True,all_target_validity_counts_verified=True,
        input_sha256={str(path.relative_to(ROOT)):sha(path)for path in[SRC/'protocol.json',OUT/'calibration-selection.json',OUT/'data.npz',OUT/'dataset.json',OLD/'model.npz',*[ROOT/r['path']for r in receipts]]}))
    print('FRESH_TARGET_READY',dataset,flush=True)
if __name__=='__main__':main()
