"""Reserve camel-family source data and fresh N13 inputs before exposure inspection."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3t';OUT=ROOT/'artifacts/research/20261114/loki/h3t'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not(OUT/'dataset.json').exists();feasibility=json.loads((OUT/'feasibility.json').read_text());assert feasibility['status']=='feasible_validated_no_fit'
    source=np.load(OUT/'solutions-n12.npy');target=np.load(OUT/'solutions-n13.npy');universe=dict(np.load(OUT/'input-universe.npz'))
    keys=np.sort(np.random.default_rng(2026112510).permutation(universe['eligible_keys'])[:4096]);inputs=np.zeros((4096,169),bool)
    for power in [2,1,0]:
        cells=keys//170**power%170;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    counts=universe['counts'][np.searchsorted(universe['keys'],keys)]
    for given,count in zip(inputs,counts):assert target[:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'training.npz',n12_all=source)
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=counts,prior_target_test_keys=universe['prior_target_keys'])
    report=dict(status='reserved_before_exposure_scores',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='queens plus camel attacks (1,3)/(3,1)',
        transfer='N12 to N13 upward transfer under unchanged exposure heuristic on a new attack family.',source_counts=dict(n12_full=len(source)),
        target_n=13,target_solution_universe=len(target),target_unique_inputs=4096,target_input_seed=2026112510,target_eligible_pool=len(universe['eligible_keys']),
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),prior_target_input_overlap=0,
        target_distribution='Uniform unique valid1/2/3clue queen-plus-camel N13 inputs excluding all H3j/H3k/H3l/H3m/H3n search input keys.',
        data_sha256=sha(OUT/'data.npz'),training_sha256=sha(OUT/'training.npz'),source_sha256=sha(__file__),
        source_solutions_sha256=sha(OUT/'solutions-n12.npy'),target_solutions_sha256=sha(OUT/'solutions-n13.npy'),input_universe_sha256=sha(OUT/'input-universe.npz'))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
