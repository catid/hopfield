"""Reserve superqueen source patterns and fresh target inputs before fitting."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3l';OUT=ROOT/'artifacts/research/20261114/loki/h3l'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not(OUT/'dataset.json').exists();source=np.load(OUT/'solutions-n12.npy');target=np.load(OUT/'solutions-n13.npy')
    universe=dict(np.load(OUT/'input-universe.npz'));indices=np.random.default_rng(2026112501).permutation(len(source))[:128]
    assert len(indices)==128 and len(source)==156 and len(target)==1876
    keys=np.sort(np.random.default_rng(2026112502).permutation(universe['eligible_keys'])[:4096]);inputs=np.zeros((4096,169),bool)
    for power in [2,1,0]:
        cells=keys//170**power%170;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
    assert not np.isin(keys,universe['prior_target_keys']).any()
    np.savez_compressed(OUT/'training.npz',source_solutions=source,training_indices=indices,training_labels=source[indices])
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=universe['prior_target_keys'])
    report=dict(status='reserved_before_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='superqueens:queen+knight attacks',
        source_n=12,source_solution_universe=156,source_training_patterns=128,source_seed=2026112501,
        target_n=13,target_solution_universe=1876,target_unique_inputs=4096,target_input_seed=2026112502,target_eligible_pool=len(universe['eligible_keys']),
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),prior_target_input_overlap=0,
        distribution='Uniform unique valid1/2/3clue superqueen N13 inputs excluding prior H3j/H3k ordinary-queen evaluation inputs.',
        target_use='Target labels only reserve/validate/evaluate inputs; training.npz is separate and is the only positive data read by model construction.',
        data_sha256=sha(OUT/'data.npz'),training_sha256=sha(OUT/'training.npz'),source_sha256=sha(__file__),feasibility_sha256=sha(OUT/'feasibility.json'))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
