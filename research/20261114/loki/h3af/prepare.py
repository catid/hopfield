"""Remaining eligible N15 clue keys after an external graph-equivalence gate."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import time
import numpy as np
import torch
from common import ROOT,SRC,OUT,sha,read,save,now,frozen
from initialization import make

def main():
    torch.set_num_threads(1);p=frozen();assert not(OUT/'datasets.json').exists();assert read(OUT/'compilation-gate.json')['status']=='complete_verified'and read(OUT/'external-graph-gate.json')['status']=='passed';q=15;c=225;base=226;prior_new=np.array([],np.int64);datasets=[];start=time.perf_counter();cpu=time.process_time()
    for bank,cfg in enumerate(p['models']):
        folder=OUT/f'b{bank}';cache=np.load(folder/'cache.npz');target=cache['boards'];columns=cache['columns'];hist=np.zeros(base**3,np.int64);indices={k:np.array(list(itertools.combinations(range(q),k)))for k in[1,2,3]}
        for lo in range(0,len(columns),4096):
            cells=q*np.arange(q)[None,:]+columns[lo:lo+4096];pieces=[]
            for clues in[1,2,3]:
                ix=indices[clues];multipliers=np.array([base**(2-j)for j in range(clues)],np.int64);keys=((cells[:,ix]+1)*multipliers).sum(-1);pieces.append(keys.reshape(-1))
                if lo==0:
                    literal=np.array([sum((int(cell)+1)*base**(2-j)for j,cell in enumerate(subset))for row in cells[:17]for subset in itertools.combinations(row,clues)]);assert np.array_equal(literal,keys[:17].reshape(-1))
            hist+=np.bincount(np.concatenate(pieces),minlength=base**3)
        universe=np.flatnonzero(hist);counts=hist[universe];assert int(counts.sum())==len(target)*575;np.savez_compressed(folder/'input-universe.npz',keys=universe,counts=counts);excluded=prior_new;eligible=np.setdiff1d(universe,excluded);seed=2026112521+bank;keys=np.sort(np.random.default_rng(seed).permutation(eligible)[:4096]);inputs=np.zeros((4096,c),bool)
        for power in[2,1,0]:
            cells=keys//base**power%base;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
        matching=hist[keys]
        for given,count in zip(inputs,matching):assert target[:,given].all(1).sum()==count>0
        np.savez_compressed(folder/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=matching,all_excluded_keys=excluded);np.save(folder/'query-inputs.npy',inputs);receipts=[]
        for seed in[0,1,2]:
            initial=make(inputs,q=q,chains=20,seed=seed,kind='permutation');path=folder/f'initial/s{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,boards=np.packbits(initial,axis=-1));assert np.array_equal(initial,make(inputs,q=q,chains=20,seed=seed,kind='permutation'));receipts.append(dict(seed=seed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(initial.tobytes()).hexdigest()))
        dataset=dict(status='prepared_after_complete_cache_and_external_graph_seals',utc=now(),bank=bank,family=cfg['family'],q=q,input_seed=2026112521+bank,unique_inputs=4096,complete_target_solutions=len(target),full_unique_clue_keys=len(universe),eligible_keys=len(eligible),all_excluded_keys=len(excluded),other_new_family_overlap=0,sampling='Uniform over remaining eligible N15 clue keys after cross-family exclusion, not unrestricted family universe. N13 encoded keys are a different dimension/domain.',by_clues={str(k):int((inputs.sum(1)==k).sum())for k in[1,2,3]},compatible_completions=int(matching.sum()),data_sha256=sha(folder/'data.npz'),query_inputs_sha256=sha(folder/'query-inputs.npy'),initial_states=receipts,selected_caps={'arbitrary':20000,'permutation':cfg['permutation_cap']},model_sha256=sha(folder/'model.npz'),cache_sha256=sha(folder/'cache.npz'),protocol_sha256=sha(SRC/'protocol.json'));save(folder/'dataset.json',dataset);paths=[SRC/'protocol.json',folder/'data.npz',folder/'query-inputs.npy',folder/'dataset.json',folder/'model.npz',folder/'calibration-selection.json',folder/'compilation.json',OUT/'compilation-gate.json',OUT/'external-graph-gate.json']+[ROOT/r['path']for r in receipts];save(folder/'preflight.json',dict(status='passed',utc=now(),input_sha256={str(x.relative_to(ROOT)):sha(x)for x in paths}));datasets.append(dataset);prior_new=np.union1d(prior_new,keys);print('TARGET_READY',bank,len(target),len(universe),dataset['by_clues'],flush=True)
    save(OUT/'datasets.json',dict(status='complete',utc=now(),datasets=datasets,all_new_N15_family_test_keys=prior_new.tolist(),shared_input_preparation_wall_seconds=time.perf_counter()-start,shared_input_preparation_cpu_seconds=time.process_time()-cpu,protocol_sha256=sha(SRC/'protocol.json')))
if __name__=='__main__':main()
