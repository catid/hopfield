"""Reserve fresh H3u target keys; no learned fit or target calibration."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3u';OUT=ROOT/'artifacts/research/20261114/loki/h3u';OLD=ROOT/'artifacts/research/20261114/loki/h3t'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(OUT/'dataset.json').exists()
    u=dict(np.load(OLD/'input-universe.npz'));previous=np.load(OLD/'data.npz')['test_keys'];prior=np.union1d(u['prior_target_keys'],previous)
    eligible=np.setdiff1d(u['keys'],prior);keys=np.sort(np.random.default_rng(2026112511).permutation(eligible)[:4096])
    inputs=np.zeros((4096,169),bool)
    for power in [2,1,0]:
        cells=keys//170**power%170;ok=cells>0;inputs[np.flatnonzero(ok),cells[ok]-1]=True
    source=np.load(OLD/'solutions-n12.npy');target=np.load(OLD/'solutions-n13.npy');counts=u['counts'][np.searchsorted(u['keys'],keys)]
    for x,c in zip(inputs,counts):assert target[:,x].all(1).sum()==c>0
    np.savez_compressed(OUT/'training.npz',n12_all=source)
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_inputs=inputs,test_keys=keys,test_solution_counts=counts,prior_target_test_keys=prior)
    d=dict(status='reserved_before_soft_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_patterns=len(source),source_n=12,target_n=13,target_solutions=len(target),
       target_unique_inputs=4096,target_input_seed=2026112511,target_eligible_pool=len(eligible),prior_target_keys=len(prior),prior_target_input_overlap=0,
       target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),
       distribution='Uniform unique valid1/2/3-clue inputs, excluding H3j/k/l/m/n/t N13 search keys. Target universe was already enumerated in H3t.',
       input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OLD/'input-universe.npz',OLD/'data.npz',OLD/'solutions-n12.npy',OLD/'solutions-n13.npy']},
       output_sha256={name:sha(OUT/name) for name in ['training.npz','data.npz']})
    (OUT/'dataset.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':main()
