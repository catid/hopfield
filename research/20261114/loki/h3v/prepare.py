"""Reserve H3v fresh input keys without inspecting search outcomes."""
import datetime as dt,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3v';OUT=ROOT/'artifacts/research/20261114/loki/h3v';BASE=ROOT/'artifacts/research/20261114/loki'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(OUT/'dataset.json').exists();u=np.load(BASE/'h3t/input-universe.npz');old=np.load(BASE/'h3u/data.npz');prior=np.union1d(old['prior_target_test_keys'],old['test_keys']);eligible=np.setdiff1d(u['keys'],prior)
    keys=np.sort(np.random.default_rng(2026112512).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
    for power in [2,1,0]:
        c=keys//170**power%170;ok=c>0;inputs[np.flatnonzero(ok),c[ok]-1]=True
    target=np.load(BASE/'h3t/solutions-n13.npy');counts=u['counts'][np.searchsorted(u['keys'],keys)]
    for x,c in zip(inputs,counts):assert target[:,x].all(1).sum()==c>0
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=prior)
    report=dict(status='reserved_before_swap_search',utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_patterns=176,target_n=13,target_solutions=1932,unique_inputs=4096,input_seed=2026112512,prior_search_keys=len(prior),eligible_keys=len(eligible),prior_overlap=0,
       by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},compatible_completions=int(counts.sum()),distribution='Uniform unique valid1/2/3-clue inputs excluding all H3j/k/l/m/n/t/u search keys; target universe previously enumerated.',
       data_sha256=sha(OUT/'data.npz'),source_sha256=sha(__file__))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
