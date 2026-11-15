"""Reserve a fresh exposure-filter evaluation cohort disjoint from H3o and H3p."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3r';OUT=ROOT/'artifacts/research/20261114/loki/h3r';BASE=ROOT/'artifacts/research/20261114/loki/h3o'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not(OUT/'dataset.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    prior=dict(np.load(BASE/'data.npz'));universe=dict(np.load(BASE/'input-universe.npz'));confirmation=np.load(ROOT/'artifacts/research/20261114/loki/h3p/data.npz');excluded=np.union1d(prior['test_keys'],confirmation['test_keys']);eligible=universe['keys'][~np.isin(universe['keys'],excluded)]
    keys=np.sort(np.random.default_rng(2026112509).permutation(eligible)[:4096]);inputs=np.zeros((4096,225),bool)
    for power in [2,1,0]:
        cells=keys//226**power%226;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert prior['all_solutions'][:,given].all(1).sum()==count>0
    for name in ['training.npz','solutions-n15.npy','feasibility.json','input-universe.npz']:shutil.copy2(BASE/name,OUT/name)
    np.savez_compressed(OUT/'data.npz',all_solutions=prior['all_solutions'],test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=excluded)
    report=dict(status='reserved_before_exposure_scores',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='superqueens:queen+knight attacks',
        transfer='N14 to N15 upward transfer with a source-only independence-model exposure heuristic.',source_counts=dict(n14_full=5180),
        target_n=15,target_solution_universe=32516,target_unique_inputs=4096,target_input_seed=2026112509,target_eligible_pool=len(eligible),
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),prior_target_input_overlap=0,
        target_distribution='Uniform unique valid1/2/3clue N15 superqueen inputs excluding all H3o/H3p search inputs. The full target universe was already enumerated and structurally censused in H3q; fresh here means no prior search-cohort overlap, not unseen target labels.',
        new_method='Source-only marginal exposure threshold fixed before scores are computed. Inferred homogeneous lines, target q, unit and search settings stay inherited.',
        data_sha256=sha(OUT/'data.npz'),training_sha256=sha(OUT/'training.npz'),source_sha256=sha(__file__),parent_data_sha256=sha(BASE/'data.npz'),
        target_solutions_sha256=sha(OUT/'solutions-n15.npy'),input_universe_sha256=sha(OUT/'input-universe.npz'))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
