"""Fresh cohorts and label-free query-input files after full cache sealing."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import numpy as np
import torch
from common import ROOT,SRC,OUT,OLD,sha,read,save,now,frozen
from initialization import make

def main():
    torch.set_num_threads(1);p=frozen(target=True);assert not(OUT/'datasets.json').exists();gate=read(OUT/'compilation-gate.json');assert gate['status']=='complete_verified';old=read(OLD/'datasets.json');prior=np.union1d(old['all_prior_and_calibration_excluded_keys'],old['all_new_family_test_keys']);prior_new=np.array([],np.int64);datasets=[]
    for bank,cfg in enumerate(p['models']):
        folder=OUT/f'b{bank}';comp=read(folder/'compilation.json');assert comp['cache_sha256']==sha(folder/'cache.npz')and comp['source_model_sha256']==cfg['model_sha256'];universe=np.load(OLD/f'b{bank}/input-universe.npz');excluded=np.union1d(prior,prior_new);eligible=np.setdiff1d(universe['keys'],excluded);keys=np.sort(np.random.default_rng(2026112519+bank).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
        for power in[2,1,0]:
            cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
        target=np.load(OLD/f'b{bank}/solutions-n13.npy');counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
        for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
        np.savez_compressed(folder/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_search_and_calibration_keys=prior,prior_new_family_test_keys=prior_new,all_excluded_keys=excluded);np.save(folder/'query-inputs.npy',inputs);receipts=[]
        for seed in[0,1,2]:
            initial=make(inputs,q=13,chains=20,seed=seed,kind='permutation');path=folder/f'initial/s{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,boards=np.packbits(initial,axis=-1));assert np.array_equal(initial,make(inputs,q=13,chains=20,seed=seed,kind='permutation'));receipts.append(dict(seed=seed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(initial.tobytes()).hexdigest()))
        dataset=dict(status='prepared_after_compilation_seals',utc=now(),bank=bank,family=cfg['family'],input_seed=2026112519+bank,unique_inputs=4096,complete_target_solutions=len(target),eligible_keys=len(eligible),prior_search_and_calibration_keys=len(prior),prior_new_family_test_keys=len(prior_new),all_excluded_keys=len(excluded),prior_overlap=0,other_new_family_overlap=0,by_clues={str(k):int((inputs.sum(1)==k).sum())for k in[1,2,3]},compatible_completions=int(counts.sum()),data_sha256=sha(folder/'data.npz'),query_inputs_sha256=sha(folder/'query-inputs.npy'),initial_states=receipts,selected_caps={'arbitrary':20000,'permutation':cfg['permutation_cap']},model_sha256=sha(folder/'model.npz'),compilation_sha256=sha(folder/'compilation.json'),cache_sha256=sha(folder/'cache.npz'),protocol_sha256=sha(SRC/'protocol.json'));save(folder/'dataset.json',dataset);paths=[SRC/'protocol.json',folder/'data.npz',folder/'query-inputs.npy',folder/'dataset.json',folder/'model.npz',folder/'calibration-selection.json',folder/'compilation.json',OUT/'compilation-gate.json']+[ROOT/r['path']for r in receipts];save(folder/'preflight.json',dict(status='passed',utc=now(),all_compilation_sealed_before_cohorts=True,all_validity_counts_verified=True,shared_initials_reconstructed=True,input_sha256={str(x.relative_to(ROOT)):sha(x)for x in paths}));datasets.append(dataset);prior_new=np.union1d(prior_new,keys);print('TARGET_READY',bank,dataset['by_clues'],dataset['compatible_completions'],flush=True)
    save(OUT/'datasets.json',dict(status='complete',utc=now(),datasets=datasets,all_prior_excluded_keys=prior.tolist(),all_new_family_test_keys=prior_new.tolist(),protocol_sha256=sha(SRC/'protocol.json')))
if __name__=='__main__':main()
