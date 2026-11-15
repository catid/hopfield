"""Reserve multisize source archives and a new N13 superqueen input cohort."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3m';OUT=ROOT/'artifacts/research/20261114/loki/h3m';BASE=ROOT/'artifacts/research/20261114/loki/h3l'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not(OUT/'dataset.json').exists();training=dict(np.load(BASE/'training.npz'));previous=dict(np.load(BASE/'data.npz'))
    n11=np.load(OUT/'solutions-n11.npy');universe=dict(np.load(BASE/'input-universe.npz'))
    excluded=np.union1d(universe['prior_target_keys'],previous['test_keys']);eligible=universe['keys'][~np.isin(universe['keys'],excluded)]
    keys=np.sort(np.random.default_rng(2026112503).permutation(eligible)[:4096]);inputs=np.zeros((4096,169),bool)
    for power in [2,1,0]:
        cells=keys//170**power%170;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert previous['all_solutions'][:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'training.npz',n11_all=n11,n12_all=training['source_solutions'],n12_base=training['training_labels'],n12_base_indices=training['training_indices'])
    np.savez_compressed(OUT/'data.npz',all_solutions=previous['all_solutions'],test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=excluded)
    report=dict(status='reserved_before_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='superqueens:queen+knight attacks',
        source_counts=dict(n12_only=128,n11_all=44,multisize_total=172,n12_full=156),
        source_breadth_caveat='Mixed-size adds44 labels; full-N12 adds28. The control is descriptive and does not perfectly match total positive-label count.',
        source_base='Identical128 N12 patterns from frozen H3l, plus the complete independently validated44-pattern N11 universe or complete156-pattern N12 universe as declared.',
        target_n=13,target_solution_universe=1876,target_unique_inputs=4096,target_input_seed=2026112503,target_eligible_pool=len(eligible),
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),prior_target_input_overlap=0,
        target_distribution='Uniform unique valid1/2/3clue N13 superqueen inputs excluding all H3j/H3k/H3l evaluation inputs.',
        data_sha256=sha(OUT/'data.npz'),training_sha256=sha(OUT/'training.npz'),source_sha256=sha(__file__),
        parent_training_sha256=sha(BASE/'training.npz'),n11_solutions_sha256=sha(OUT/'solutions-n11.npy'))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
