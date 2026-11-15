"""Post-freeze fresh cohort, frozen model copy and shared initialization files."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from initialization import make
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa';OUT=ROOT/'artifacts/research/20261114/loki/h3aa';BASE=ROOT/'artifacts/research/20261114/loki'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());assert not(OUT/'dataset.json').exists()
    for device in['cpu','cuda']:assert json.loads((OUT/f'gate-{device}.json').read_text())['status']=='passed'
    for group in['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    u=np.load(BASE/'h3t/input-universe.npz');old=np.load(BASE/'h3w/data.npz');prior=np.union1d(old['prior_target_test_keys'],old['test_keys']);eligible=np.setdiff1d(u['keys'],prior)
    keys=np.sort(np.random.default_rng(2026112514).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
    for power in[2,1,0]:
        cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
    target=np.load(BASE/'h3t/solutions-n13.npy');counts=u['counts'][np.searchsorted(u['keys'],keys)]
    for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=prior)
    init_receipts=[]
    for kind in['arbitrary','permutation']:
        folder=OUT/f'initial/{kind}';folder.mkdir(parents=True,exist_ok=True)
        for seed in[0,1,2]:
            states=make(inputs,q=13,chains=20,seed=seed,kind=kind);path=folder/f's{seed}.npz';np.savez_compressed(path,boards=np.packbits(states,axis=-1))
            init_receipts.append(dict(kind=kind,search_seed=seed,initialization_seed=2026112600+seed,path=str(path.relative_to(ROOT)),file_sha256=sha(path),unpacked_bytes_sha256=hashlib.sha256(states.tobytes()).hexdigest(),shape=list(states.shape)))
    source=BASE/'h3w/models/bp-adamw-permutation/model.npz';assert sha(source)==p['model_sha256'];shutil.copy2(source,OUT/'model.npz')
    record=dict(status='reserved_after_protocol_and_source_gates',utc=dt.datetime.now(dt.timezone.utc).isoformat(),input_seed=2026112514,unique_inputs=4096,source_patterns=176,target_n=13,target_solutions=1932,
        prior_search_keys=len(prior),eligible_keys=len(eligible),prior_overlap=0,by_clues={str(k):int((inputs.sum(1)==k).sum())for k in[1,2,3]},compatible_completions=int(counts.sum()),
        protocol_sha256=sha(SRC/'protocol.json'),data_sha256=sha(OUT/'data.npz'),model_sha256=sha(OUT/'model.npz'),initial_states=init_receipts)
    save(OUT/'dataset.json',record);print(json.dumps(record))
if __name__=='__main__':main()
