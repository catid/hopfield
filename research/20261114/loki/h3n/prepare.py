"""Reserve larger-source archives and fresh N13 inputs before downward-transfer fitting."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3n';OUT=ROOT/'artifacts/research/20261114/loki/h3n';BASE=ROOT/'artifacts/research/20261114/loki/h3m'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not(OUT/'dataset.json').exists();train=dict(np.load(BASE/'training.npz'));previous=dict(np.load(BASE/'data.npz'))
    n14=np.load(OUT/'solutions-n14.npy');indices=np.random.default_rng(2026112504).permutation(len(n14))[:156]
    universe_path=ROOT/'artifacts/research/20261114/loki/h3l/input-universe.npz';universe=dict(np.load(universe_path))
    excluded=np.union1d(previous['prior_target_test_keys'],previous['test_keys']);eligible=universe['keys'][~np.isin(universe['keys'],excluded)]
    keys=np.sort(np.random.default_rng(2026112505).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
    for power in [2,1,0]:
        cells=keys//170**power%170;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert previous['all_solutions'][:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'training.npz',n12_all=train['n12_all'],n14_all=n14,n14_matched=n14[indices],n14_matched_indices=indices)
    np.savez_compressed(OUT/'data.npz',all_solutions=previous['all_solutions'],test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=excluded)
    report=dict(status='reserved_before_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='superqueens:queen+knight attacks',
        transfer='N14 to N13 downward transfer; N12 to N13 comparison is upward transfer.',
        source_counts=dict(n12_full=156,n14_matched=156,n14_full=len(n14)),source_matched_seed=2026112504,
        source_controls='N12 full156 versus random N14 156 holds label count fixed; full N14 additionally changes sample count. One source subset, not a distribution over source draws.',
        target_n=13,target_solution_universe=1876,target_unique_inputs=4096,target_input_seed=2026112505,target_eligible_pool=len(eligible),
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),prior_target_input_overlap=0,
        target_distribution='Uniform unique valid1/2/3clue N13 superqueen inputs excluding all H3j/H3k/H3l/H3m evaluation inputs.',
        data_sha256=sha(OUT/'data.npz'),training_sha256=sha(OUT/'training.npz'),source_sha256=sha(__file__),
        parent_training_sha256=sha(BASE/'training.npz'),n14_solutions_sha256=sha(OUT/'solutions-n14.npy'),input_universe_sha256=sha(universe_path))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
